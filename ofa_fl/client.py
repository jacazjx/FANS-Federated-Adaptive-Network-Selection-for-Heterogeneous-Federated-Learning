import os
from collections import OrderedDict

import torch
from torch import nn
import sys

import numpy as np
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from datetime import datetime
import socket

from hypernet.utils.communication_utils import send, recv
from hypernet.utils.common_utils import set_seed, adjust_learning_rate
from tqdm import tqdm

from hypernet.utils.evaluation import calculate_SLC_metrics, display_results
from hypernet.build_model import build_model
from hypernet.utils.metric_utils import MaskedCrossEntropyLoss, KLLoss

EPS = 1e-7


class ClientCluster():
    def __init__(self, port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # s.connect(('8.8.8.8', 80))
            # self.ip = s.getsockname()[0]
            self.ip = "127.0.0.1"
        finally:
            s.close()

        self.port = port
        self.server_ip = None
        self.clients = {}
        print('address:', (self.ip, self.port))

    def register_task(self, args, server_args, subnet_configs):
        self.subnet_configs = subnet_configs
        self.subnet_keys = {}
        self.n_class = server_args.n_class
        set_seed(server_args.seed)
        args.task = server_args.task
        args.total_clients = server_args.total_clients
        args.n_large = server_args.n_full[0]
        args.classifier_name = server_args.classifier_name
        args.finetune_epochs = server_args.finetune_epochs
        args.beta = server_args.beta
        args.temp = server_args.temp
        args.trs = server_args.trs
        args.algorithm = server_args.algorithm
        args.lr = server_args.lr
        args.momentum = server_args.momentum
        args.weight_decay = server_args.weight_decay
        args.batch_size = server_args.batch_size
        args.rounds = server_args.rounds
        args.use_scaler = server_args.use_scaler
        args.width_ratio_list = server_args.width_ratio_list
        args.dyn_alpha = server_args.dyn_alpha
        args.round_alpha = server_args.round_alpha
        args.warm_up = server_args.warm_up
        args.fed_dyn = server_args.fed_dyn

        if server_args.task.startswith('cifar'):
            from hypernet.datasets.load_cifar import load_cifar
            trainData, valData, testData = load_cifar(server_args.task, os.path.join(args.data_dir, server_args.task),
                                                      server_args.data_shares, server_args.alpha, server_args.n_full[0])
            collate_fn = None

        elif server_args.task == 'mnist':
            valData = [None] * server_args.total_clients
            from hypernet.datasets.load_mnist import load_mnist
            trainData, testData = load_mnist(os.path.join(args.data_dir, server_args.task), server_args.data_shares,
                                             server_args.alpha, server_args.n_full[0])
            collate_fn = None

        elif server_args.task == 'mnli':
            valData = [None] * server_args.total_clients
            from hypernet.datasets.load_mnli import load_mnli, collate_fn
            trainData, testData = load_mnli(os.path.join(args.data_dir, server_args.task, 'original'),
                                            server_args.data_shares, server_args.alpha, server_args.n_full[0])
            collate_fn = collate_fn
        else:
            raise ValueError('Wrong dataset.')

        return trainData, valData, testData, collate_fn

    def run(self, args):
        self.device = args.device
        # waiting for server to send request
        try:
            soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            soc.bind((self.ip, self.port))
            soc.listen(1)
            print('Start Listening...')

            while True:
                try:
                    new_socket, source_addr = soc.accept()
                    new_socket.settimeout(args.timeout)
                    if self.server_ip is not None and source_addr[0] != self.server_ip:
                        new_socket.close()
                        print(f'\033[31mReceive Unexpected Connection from {source_addr}. Connection Close.\033[0m')

                    print(f'Receive connection from {source_addr}')
                    # receive request
                    msg, status = recv(new_socket, args.buffer_size, recv_timeout=60)
                    if status == 1:
                        print(f"Receive {msg['subject'].upper()} message from {source_addr}")

                    if isinstance(msg, dict):
                        if msg['subject'] == 'register':
                            self.server_ip = source_addr[0]
                            server_args = msg['data']['args']
                            trainData, valData, testData, collate_fn = self.register_task(
                                args, server_args, msg['data']['subnet_configs'])
                            client_features = {}
                            for cid in msg['data']['ids']:
                                self.clients[cid] = Client(args, msg['data']['args'], cid, trainData[cid], valData[cid],
                                                           testData[cid], collate_fn)
                                client_features[cid] = self.clients[cid].class_distribution

                            data_byte = pickle.dumps(
                                {"subject": "register", "data": {"client_features": client_features}})
                            print("Registered. Reply to the Server.")
                            send(new_socket, data_byte, args.buffer_size)

                            del data_byte

                        elif msg['subject'] == 'train_and_eval':
                            response_data = {}
                            print("===================\n"
                                  "\t\t[TRAIN]\n"
                                  "===================\n")
                            # train
                            for cid in msg['data']['train']['ids']:
                                response_data[cid] = {"model": None, "score": None}
                                recv_weights = msg['data']['train']['model'][cid]
                                updated_weights, test_scores = self.clients[cid].local_update(args,
                                                                                              msg['data']['round'],
                                                                                              recv_weights,
                                                                                              msg['data']['subnet_configs'][cid])
                                response_data[cid]["model"] = updated_weights
                                response_data[cid]["score"] = test_scores
                                display_results(test_scores, self.clients[cid].metrics)

                            print("===================\n"
                                  "\t\t[EVAL]\n"
                                  "===================\n")
                            # eval
                            for cid in msg['data']['eval']['ids']:
                                # don't update client model
                                model_config = msg['data']['subnet_configs'][cid]
                                model = self.clients[cid].base_model.get_subnet(model_config).to(args.device)
                                recv_weights = msg['data']['eval']['model'][cid]

                                missing_keys, unexpected_keys = model.load_state_dict(recv_weights, strict=False)
                                if len(missing_keys) or len(unexpected_keys):
                                    print('Warning: missing %i missing_keys, %i unexpected_keys.' % (
                                    len(missing_keys), len(unexpected_keys)))
                                model = self.clients[cid].fine_tune(args, model)
                                test_scores, test_loss = self.clients[cid].evaluate(args, model)
                                print(f"Evaluated Client %i. Test loss = %.4f" % (cid, test_loss))
                                display_results(test_scores, self.clients[cid].metrics)
                                if cid in response_data:
                                    response_data[cid]["score"] = test_scores
                                else:
                                    response_data[cid] = {"score": test_scores}

                            # reply request
                            data_byte = pickle.dumps({"subject": "train_and_eval", "data": response_data})
                            print(f"Trained and evaluated. Send {len(data_byte) * 1e-9} Gb to the Server.")
                            new_socket.settimeout(3600)
                            send(new_socket, data_byte, args.buffer_size)

                            del data_byte
                finally:
                    new_socket.close()
                    print(f'Close Connection with {source_addr}')
        finally:
            soc.close()


class Client:
    def __init__(self, args, server_args, id, trainData, valData, testData, collate_fn):
        self.id = id
        args.epochs = server_args.epochs
        args.buffer_size = server_args.buffer_size
        set_seed(server_args.seed)
        self.task = server_args.task
        self.is_large = id >= (server_args.total_clients - server_args.n_full[0])
        self.classifier_name = server_args.classifier_name
        self.device = args.device
        self.metrics = server_args.metrics

        self.trainData = trainData
        self.valData = valData
        self.testData = testData
        self.collate_fn = collate_fn
        self.n_class = server_args.n_class

        self.base_model = build_model(server_args.width_ratio_list, server_args.task, server_args.n_class, args.device)
        # client features
        class_distribution = np.zeros(self.n_class)
        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn,
                                  num_workers=0)
        for _, labels in train_loader:
            for cls in range(self.n_class):
                class_distribution[cls] += labels.numpy().tolist().count(cls)
        self.class_distribution = class_distribution / np.sum(class_distribution)
        print(f'Client {id} class distribution:', self.class_distribution)

        self.label_mask = torch.zeros(self.n_class).to(args.device)
        self.label_mask[self.class_distribution > 0.] = 1.

        self.model_config = server_args.client_model_configs[id]
        self.alpha = 0.01
        self.prev_grads = 0.

        print(f'Client {self.id} n_train: {len(self.trainData)}, n_class: {self.n_class}')

    def train_one_batch(self, model, sample, label, optimizer, criterion):
        model.train()
        label = label.to(self.device, dtype=torch.long)
        if len(label.shape) > 1:
            label = torch.argmax(label, dim=-1)
        optimizer.zero_grad()
        t_out = self.model_fit(model, sample)
        loss = criterion(t_out, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    def finetuning_one_batch(self, args, model, sample, label, optimizer, criterion):
        model.train()
        with torch.no_grad():
            label = label.to(self.device, dtype=torch.long)
            if len(label.shape) > 1:
                label = torch.argmax(label, dim=-1)
            optimizer.zero_grad()
            t_out = self.model_fit(model, sample)
            loss = criterion(t_out, label)
            # loss.backward()
            # optimizer.step()
        return loss.item()

    def evaluate(self, args, model):
        data_loader = DataLoader(self.testData, batch_size=args.batch_size, shuffle=False, collate_fn=self.collate_fn,
                                 num_workers=1)
        criterion = nn.CrossEntropyLoss()
        y_pred = []
        y_true = []

        model.eval()
        avg_loss = []
        with torch.no_grad():
            for sample, label in data_loader:
                label = label.to(self.device, dtype=torch.float)
                if len(label.shape) == 1:
                    label = F.one_hot(label.to(torch.long), num_classes=self.n_class)

                out = self.model_fit(model.to(self.device), sample)
                avg_loss.append(criterion(out, torch.argmax(label, dim=-1)).item())
                out = torch.softmax(out, dim=-1)

                y_pred.extend(out.cpu().numpy())
                y_true.extend(label.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        test_scores = calculate_SLC_metrics(y_true, y_pred)

        return test_scores, np.mean(avg_loss)

    def local_update(self, args, curr_round, model_weights, model_config):
        self.model = self.base_model.get_subnet(model_config).to(self.device)
        if args.task == 'mnli':
            self.model.train_adapter("mnli")
        missing_keys, unexpected_keys = self.model.load_state_dict(model_weights, strict=False)
        if len(missing_keys) or len(unexpected_keys):
            print('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))

        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn,
                                  num_workers=4, pin_memory=True)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        kd_criterion = KLLoss(args.temp).to(args.device)

        local_epochs = args.epochs
        if args.warm_up and curr_round == 0:
            local_epochs = 50
        ########################################
        # FANS, SD
        ########################################
        elif args.algorithm == 'fans':

            for e in range(local_epochs):
                start_time = datetime.now()
                adjust_learning_rate(optimizer, args.lr, curr_round * args.epochs + e)
                for idx, (sample, label) in enumerate(tqdm(train_loader, total=len(train_loader))):
                    subnet_configs = self.model.get_progressive_subnet_configs()
                    # train_one_batch
                    n = len(subnet_configs) + 1
                    criterion = nn.CrossEntropyLoss()
                    self.model.train()
                    label = label.to(self.device, dtype=torch.long)
                    if len(label.shape) > 1:
                        label = torch.argmax(label, dim=-1)
                    optimizer.zero_grad()
                    t_out = self.model_fit(self.model, sample)
                    loss = criterion(t_out, label) / n
                    for i, subnet_config in enumerate(subnet_configs):
                        s_out = self.model_fit(self.model, sample, subnet_config=subnet_config)
                        kl_loss = kd_criterion(s_out, t_out.detach())
                        loss += ((kl_loss * args.beta + (1-args.beta) * criterion(s_out, label)) / n)
                    loss.backward()
                    optimizer.step()

                end_time = datetime.now()
                duration = (end_time - start_time).seconds / 60.
                print('[TRAIN] Client %i, Epoch %i, Loss, %.3f, time=%.3fmins' % (
                    self.id, curr_round * args.epochs + e, loss.item(), duration))
                # client testing
                if e == args.finetune_epochs - 1 or (e + 1) % args.epochs == 0:  # test after fine_tune_epoch
                    test_scores, _ = self.evaluate(args, self.model)
                    display_results(test_scores, self.metrics)

        updated_weights = OrderedDict({k: p.cpu() for k, p in self.model.state_dict().items()})
        del self.model
        torch.cuda.empty_cache()
        return updated_weights, test_scores

    # train model one round with only , without changing self.model value
    def fine_tune(self, args, model):
        # self.recv_params = torch.cat([p.reshape(-1) for p in model.to(self.device).parameters()])
        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn,
                                  num_workers=0)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        model.train()
        for e in range(args.finetune_epochs):
            start_time = datetime.now()
            for sample, label in tqdm(train_loader, total=len(train_loader)):
                criterion = nn.CrossEntropyLoss()
                self.finetuning_one_batch(args, model, sample, label, optimizer, criterion)
            end_time = datetime.now()
            duration = (end_time - start_time).seconds / 60.
            print('[FINE-TUNE] Client %i, time=%.3fmins' % (self.id, duration))
        return model

    def model_fit(self, model, sample, subnet_config=None, return_emb=False):
        if self.task == 'mnli':
            output = model(sample[0].to(self.device), token_type_ids=sample[1].to(self.device),
                           attention_mask=sample[2].to(self.device), output_hidden_states=True,
                           active_output_path=subnet_config)
            if return_emb:
                return output.hidden_states[-1][:, 0], output.logits
            else:
                return output.logits
        else:
            return model(sample.to(self.device), subnet_config, return_emb)
