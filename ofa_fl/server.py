import os.path
import pickle
import random
import socket
import threading
import time
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from hypernet.hypernetworks.resnet import super_resnet18, cost_budget
from hypernet.utils.common_utils import SuperWeightAveraging, convert_model_to_dict
from hypernet.utils.communication_utils import recv, send

EPS = 1e-7
class Server():
    def __init__(self, args):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()

        self.port = args.port
        self.total_clients = args.total_clients
        self.buffer_size = args.buffer_size
        self.timeout = args.timeout
        self.device = args.device
        self.metrics = args.metrics
        self.client_clusters = defaultdict(set)
        self.client_addr = {}
        self.client_features = {}
        self.logger = args.logger
        self.client_model_configs = args.client_model_configs

        if args.task == 'mnli':
            args.classifier_name = 'heads.'
        else:
            args.classifier_name = 'output_layer'

        if args.task == 'cifar10':
            self.hypernet = super_resnet18(True, args.use_scaler, args.width_ratio_list)
        elif args.task == 'cifar100':
            pass

        self.aggregate_tool = SuperWeightAveraging(self.hypernet, args.dyn_alpha, args.fed_dyn)
        self.alpha = args.round_alpha
        self.beta = args.round_beta
        assert self.alpha + self.beta == 1, "alpha, beta should sum up to 1"

    def register_client(self, id, ip, port):
        self.client_addr[id] = (ip, port)
        self.client_clusters[(ip, port)].add(id)


    def train(self, args):
        # types of messenge that server send to client
        # train: ask client to train model and return the model parameter
        # update: send the updated model to the client
        # stop: ask client to stop training and close connection
        if args.task == 'cifar10':
            unweighted = False
        else:
            unweighted = True

        self.logger.debug('---Start Registration---')
        threads = {}
        for cluster, cids in self.client_clusters.items():
            self.port = ((self.port - 1024) % (65535 - 1024)) + 1025
            subnet_configs = [self.client_model_configs[cid] for cid in cids]

            send_msg = {
                "subject": "register",
                "data": {
                    "args": args,
                    "ids": cids,
                    "subnet_configs": subnet_configs,
                }
            }

            socket_thread = SocketThread(
                addr=(self.ip, self.port),
                client_addr=cluster,
                send_msg=pickle.dumps(send_msg),
                buffer_size=args.buffer_size,
                timeout=self.timeout,
                logger=self.logger
            )
            socket_thread.start()
            threads[cluster] = socket_thread

        for cluster in threads:
            threads[cluster].join()
            self.client_features.update(threads[cluster].get_result()["client_features"])
        self.logger.debug('---Finish Registration---')

        self.all_selected_clients = set()
        # 计算每个阶段的round数
        total_rounds = args.rounds
        alpha_rounds = int(total_rounds * self.alpha)
        last_model_config = {c: f"1_1_{min(args.width_ratio_list)}" for c in self.client_model_configs.keys()}
        for r in range(args.rounds + 1):
            start_time = time.time()

            # large device always join the round
            selected_clients = sorted(np.random.permutation(list(
                self.client_addr.keys())[:args.total_clients - args.n_large])[:args.sample_clients])
            selected_clients.extend(list(range(args.total_clients - args.n_large, args.total_clients)))

            if r == args.rounds:
                selected_clients = []
            self.all_selected_clients = self.all_selected_clients | set(selected_clients)
            init_client_weights = {}
            eval_client_weights = {}
            self.logger.critical(f'Round {r} selected clients: {selected_clients}')

            threads = {}
            for cluster in self.client_clusters:
                train_clients = [c for c in selected_clients if c in self.client_clusters[cluster]]
                eval_clients = self.client_clusters[cluster] - set(train_clients)

                if r < alpha_rounds:
                    # Stage 2: 执行弹性扩大
                    self.logger.debug('Stage 1: Training with progressive model settings')
                    subnet_configs = {c: cost_budget(get_progressive_configs(last_model_config[c],
                                                                             self.client_model_configs[c][c],
                                                                             args.width_ratio_list))
                                      for c in train_clients}
                else:
                    # Stage 1: 训练最大模型设置
                    self.logger.debug('Stage 2: Training with maximum model settings')
                    subnet_configs = {c: cost_budget(random.choice(self.client_model_configs[c])) for c in
                                      train_clients}


                # train_clients
                self.hypernet.train()
                if r != 0 and args.standalone:
                    init_client_weights.update({c: OrderedDict([]) for c in train_clients})
                else:
                    init_cluster_weights = {
                        c: convert_model_to_dict(self.hypernet.get_subnet(subnet_configs[c]), args.trs)
                        for c in train_clients}
                    init_client_weights.update(init_cluster_weights)

                # eval_clients
                self.hypernet.eval()
                with torch.no_grad():
                    if r != 0 and args.standalone:
                        eval_client_weights.update({c: OrderedDict([]) for c in eval_clients})
                    else:
                        # calculate grad for regularization in server_update
                        eval_model_weights = {
                            c: convert_model_to_dict(self.hypernet.get_subnet(subnet_configs[c]), args.trs)
                            for c in eval_clients}
                        eval_client_weights.update(eval_model_weights)

                # model_weight - {global_key_idx: weight}
                send_msg = {
                    "subject": "train_and_eval",
                    "data": {
                        "round": r,
                        "train": {
                            'ids': train_clients,
                            "model": init_client_weights
                        },
                        "eval": {
                            "ids": eval_clients,
                            "model": eval_client_weights
                        },
                        "subnet_configs": subnet_configs
                    }
                }

                self.port = ((self.port - 1024) % (65535 - 1024)) + 1025

                socket_thread = SocketThread(
                    addr=(self.ip, self.port),
                    client_addr=cluster,
                    send_msg=pickle.dumps(send_msg),
                    buffer_size=args.buffer_size,
                    timeout=self.timeout,
                    logger=self.logger
                )
                socket_thread.start()
                threads[cluster] = socket_thread

            client_response = defaultdict(dict)
            for cluster in threads:
                threads[cluster].join()
                client_response.update(threads[cluster].get_result())


            model_list = []
            data_ratio_list = []
            for c, res in client_response.items():
                if c not in selected_clients:
                    continue
                if not args.standalone:
                    model_list.append(res['model'])
                    data_ratio_list.append(args.data_shares[c])
            if len(selected_clients) != 0:
                data_ratio_list = torch.tensor(data_ratio_list)
                self.aggregate_tool.aggregate(model_list, None)
                # 保存
                torch.save(self.hypernet.state_dict(), os.path.join(args.save_dir, "hypernet.pth"))

            self.logger.debug('Model Aggregation')

            end_time = time.time()
            duration = (end_time - start_time) / 60.
            avg_scores = {'small': {}, 'large': {}}

            # 打印准确率，准确率测试是在客户端进行的
            for metric in self.metrics:
                avg_scores['small'][metric] = np.average(
                    [client_response[c]['score'][metric] for c in range(args.total_clients - args.n_large)])
                avg_scores['large'][metric] = np.average([client_response[c]['score'][metric] for c in
                                                          range(args.total_clients - args.n_large, args.total_clients)])
            self.logger.critical('[TRAIN] Round %i, time=%.3fmins, ACC-small=%.4f, ACC-large=%.4f' % (
                r, duration, avg_scores['small']['ACC'], avg_scores['large']['ACC']))
            for c in client_response:
                self.logger.critical({c: {m: round(client_response[c]['score'][m], 4) for m in self.metrics}})
            if args.standalone:
                break



class SocketThread(threading.Thread):
    def __init__(self, addr, client_addr, send_msg, buffer_size=1024, timeout=10, logger=None):
        threading.Thread.__init__(self)
        self.addr = addr
        self.client_addr = client_addr
        self.send_msg = send_msg
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.logger = logger

    def run(self):
        try:
            self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.soc.bind(self.addr)
            self.soc.connect(self.client_addr)
            self.logger.debug(
                f"Run a Thread for connection with {self.client_addr}. Send {round(len(self.send_msg) * 1e-9, 4)} Gb.")
            send(self.soc, self.send_msg, self.buffer_size)

            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = f"Waiting for data from {self.client_addr}. Starting at {time_struct.tm_mday}/{time_struct.tm_mon}/{time_struct.tm_year} {time_struct.tm_hour}:{time_struct.tm_min}:{time_struct.tm_sec}"
            self.logger.debug(date_time)
            msg, status = recv(self.soc, self.buffer_size, self.timeout)
            self.received_data = msg["data"]  # model weight
            self.logger.debug(f"Receive {msg['subject'].upper()} message from {self.client_addr}")
            if status == 0:
                self.logger.debug(
                    f"Connection Closed with {self.client_addr} either due to inactivity for {self.timeout} sec or an error.")

        except BaseException as e:
            self.logger.error(f"Error Connecting to the Client {self.client_addr}: {e}")

        finally:
            self.soc.close()
            self.logger.debug(f'Close connection with {self.client_addr}.')

    def get_result(self):
        try:
            return self.received_data
        except Exception as e:
            self.logger.error(f"Error Getting Result from {self.client_addr}: {e}.")
            return None
