import os.path
import pickle
import random
import socket
import threading
import time

from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hypernet.hypernetworks.bert import super_bert_base
from hypernet.hypernetworks.resnet import super_resnet18, super_resnet101
from hypernet.hypernetworks.densenet import super_densenet121
from hypernet.utils.common_utils import SuperWeightAveraging, convert_model_to_dict, is_subnet, config_to_matrix, matrix_to_config, ModelParameterVisualizer
from hypernet.utils.communication_utils import recv, send

EPS = 1e-7
class Server():
    def __init__(self, args):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # s.connect(('8.8.8.8', 80))
            # self.ip = s.getsockname()[0]
            self.ip = "127.0.0.1"
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

        if args.task == 'cifar10':
            self.hypernet = super_resnet18(True, args.use_scaler, args.width_ratio_list, num_classes=10)
        elif args.task == 'cifar100':
            self.hypernet = super_densenet121(width_pruning_ratio_list=args.width_ratio_list, num_classes=100)
        elif args.task == "mnli":
            self.hypernet = super_bert_base(width_pruning_ratio_list=args.width_ratio_list, num_classes=3)
        elif args.task == "imagenet":
            self.hypernet = super_resnet18(True, args.use_scaler, args.width_ratio_list, num_classes=1000, input_size=(1, 3, 64, 64))
        else:
            raise ValueError('Wrong task!!!:', args.task)

        if args.algorithm == "fans":
            if os.path.exists(f"{args.save_dir}/client_config"):
                self.client_model_configs = torch.load(f"{args.save_dir}/client_config")
            else:
                all_subnet_configs = self.hypernet.generate_all_subnet_configs()
                print("Start to execute budgets...")
                all_subnet_budget = []
                all_subnet_matrix = []

                # 计算每个线程需要处理的配置数量
                num_configs = len(all_subnet_configs)
                num_threads = 10
                configs_per_thread = num_configs // num_threads

                # 将all_subnet_configs分成十个子列表
                config_chunks = [all_subnet_configs[i * configs_per_thread:(i + 1) * configs_per_thread] for i in range(num_threads)]
                if num_configs % num_threads != 0:
                    config_chunks[-1].extend(all_subnet_configs[num_threads * configs_per_thread:])

                def process_config_chunk(config_chunk, hypernet):
                    results = []
                    for config in config_chunk:
                        matrix = config_to_matrix(config, hypernet.BASE_DEPTH_LIST)
                        cost = torch.sum(torch.tensor(matrix) * hypernet.layer_cost)
                        results.append((matrix, cost))
                    return results

                # 使用ThreadPoolExecutor并行处理这些子列表
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = {executor.submit(process_config_chunk, config_chunk, self.hypernet): config_chunk for config_chunk in config_chunks}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing config chunks"):
                        for matrix, cost in future.result():
                            if cost in all_subnet_budget:
                                continue
                            all_subnet_matrix.append(matrix)
                            all_subnet_budget.append(cost)

                # 根据 all_subnet_budget 对 all_subnet_matrix 和 all_subnet_budget 进行同步排序
                sorted_indices = sorted(range(len(all_subnet_budget)), key=lambda k: all_subnet_budget[k])
                all_subnet_matrix = [all_subnet_matrix[i] for i in sorted_indices]
                all_subnet_budget = [all_subnet_budget[i] for i in sorted_indices]

                cache = {}
                for (client, budget) in self.client_model_configs.items():
                    if budget in cache.keys():
                        self.client_model_configs[client] = cache[budget]
                        continue
                    subnet_configs = []
                    for cost, matrix in zip(all_subnet_budget, all_subnet_matrix):
                        if cost <= (budget + 1e-7):
                            subnet_configs.append(matrix)
                    non_subnets = []
                    print(f"Searching largest configs for Client {client} with constraint {budget}")
                    for i, config in enumerate(subnet_configs):
                        is_sub = False
                        for j, other_config in enumerate(subnet_configs):
                            if i != j and is_subnet(config, other_config):
                                is_sub = True
                                break
                        if not is_sub:
                            non_subnets.append(config)
                    non_subnets = [matrix_to_config(config, self.hypernet.BASE_DEPTH_LIST, True if args.task != 'mnli' else False) for config in non_subnets]
                    self.client_model_configs[client] = non_subnets
                    cache[budget] = non_subnets

        torch.save(self.client_model_configs, f"{args.save_dir}/client_config")
        self.aggregate_tool = SuperWeightAveraging(self.hypernet, args.dyn_alpha, args.fed_dyn)
        self.alpha = args.round_alpha
        self.beta = args.round_beta
        # self.visualizer = ModelParameterVisualizer(self.hypernet)
        if os.path.exists(os.path.join(args.save_dir, f"hypernet_{args.alpha}.pth")):
            loaded_obj = torch.load(os.path.join(args.save_dir, f"hypernet_{args.alpha}.pth"))
            if hasattr(loaded_obj, "state_dict"):
                self.hypernet.load_state_dict(loaded_obj.state_dict())
            else:
                self.hypernet.load_state_dict(loaded_obj)


    def register_client(self, id, ip, port):
        self.client_addr[id] = (ip, port)
        self.client_clusters[(ip, port)].add(id)


    def train(self, args):
        # types of messenge that server send to client
        # train: ask client to train model and return the model parameter
        # update: send the updated model to the client
        # stop: ask client to stop training and close connection

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

        for r in range(args.rounds + 1):
            self.global_round = r
            start_time = time.time()
            # self.visualizer.record_parameters(epoch=r)
            # self.visualizer.visualize_all_layers_horizontal()

            # large device always join the round
            selected_clients = sorted(np.random.permutation(list(
                self.client_addr.keys())[:args.total_clients - args.n_full[0]])[:args.sample_clients])
            selected_clients.extend(list(range(args.total_clients - args.n_full[0], args.total_clients)))

            if r == args.rounds:
                selected_clients = []
            self.all_selected_clients = self.all_selected_clients | set(selected_clients)
            init_client_weights = {}
            eval_client_weights = {}
            self.logger.critical(f'Round {r} selected clients: {selected_clients}')

            threads = {}
            for cluster in self.client_clusters:
                train_clients = [c for c in selected_clients if c in self.client_clusters[cluster]]
                # eval_clients = self.client_clusters[cluster] - set(train_clients)

                subnet_configs = {c: random.choice(self.client_model_configs[c]) for c in train_clients}



                if r != 0 and args.standalone:
                    init_client_weights.update({c: OrderedDict([]) for c in train_clients})
                else:
                    init_cluster_weights = {
                        c: convert_model_to_dict(self.hypernet.get_subnet(subnet_configs[c]), args.trs)
                        for c in train_clients}
                    init_client_weights.update(init_cluster_weights)


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
                            "ids": [],
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
                self.aggregate_tool.aggregate(model_list, data_ratio_list)
                # 保存
                torch.save(self.hypernet, os.path.join(args.save_dir, f"hypernet_{args.alpha}.pth"))

            self.logger.debug('Model Aggregation')

            end_time = time.time()
            duration = (end_time - start_time) / 60.
            avg_scores = {'small': {}, 'large': {}}

            # 打印准确率，准确率测试是在客户端进行的

            self.logger.critical('[TRAIN] Round %i, time=%.3fmins' % (r, duration))
            for c in client_response:
                self.logger.critical({c: {m: round(client_response[c]['score'][m], 4) for m in self.metrics}})
            if args.standalone:
                break
        # 训练完成后，发送停止信号给所有客户端集群
        self.logger.debug('---Start Sending Stop Signal---')
        stop_threads = {}
        for cluster in self.client_clusters:
            send_msg = {
                "subject": "stop",
                "data": {}
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
            stop_threads[cluster] = socket_thread

        for cluster in stop_threads:
            stop_threads[cluster].join()

        self.logger.debug('---Finish Sending Stop Signal---')

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
