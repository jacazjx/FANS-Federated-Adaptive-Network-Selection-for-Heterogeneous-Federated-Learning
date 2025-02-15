import os
import random
import sys
from datetime import datetime

print(sys.path)
# sys.path.append(os.path.join(sys.path[0], "utils"))
import warnings

warnings.filterwarnings("ignore")
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
import torch
import argparse
import numpy as np

from hypernet.utils.common_utils import set_seed
from server import Server
from hypernet.utils.logger import Logger
from hypernet.utils.init_utils import gen_data_shares


def main(args):
    save_dir = os.path.join(args.save_dir, f"{args.task}/seed{args.seed}/")
    args.save_dir = save_dir
    set_seed(args.seed)


    # 生成一组设备数量以及数据分布
    data_shares = []
    device_distribution = {
        "small": args.n_small[0],
        "medium": args.n_medium[0],
        "large": args.n_large[0],
        "server": args.n_full[0]
    }
    for i in range(args.n_small[0]):
        data_shares.append(args.n_small[1]/100)
    for i in range(args.n_medium[0]):
        data_shares.append(args.n_medium[1]/100)
    for i in range(args.n_large[0]):
        data_shares.append(args.n_large[1]/100)
    for i in range(args.n_full[0]):
        data_shares.append(args.n_full[1]/100)


    args.data_shares = data_shares
    assert round(np.sum(args.data_shares), 2) == 1., args.data_shares
    assert args.total_clients == len(args.data_shares)
    assert args.finetune_epochs <= args.epochs

    args.sample_clients = min(args.total_clients, args.sample_clients)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_path = os.path.join(save_dir,
                            f"alpha{args.alpha}."
                            f"{args.algorithm}."
                            f"num_client_{args.total_clients}."
                            f"sample_client_{args.sample_clients}."
                            f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}."
                            f"log")
    args.logger = Logger(file_path=log_path).get_logger()
    args.logger.critical(log_path)
    torch.cuda.set_device(args.gpu)
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    model_configs = {
        "depthfl": {
            "small":    ["1_2_1"],
            "medium":   ["2_2_1"],
            "large":    ["3_2_1"],
            "server":   ["4_2_1"]
        },
        "heterofl": {
            "small":    ["4_2_0.125"],
            "medium":   ["4_2_0.25"],
            "large":    ["4_2_0.5"],
            "server":   ["4_2_1"]
        },
        "scalefl": {
            "small":    ["1_2_0.25"],
            "medium":   ["2_2_0.5"],
            "large":    ["3_2_0.75"],
            "server":   ["4_2_1"]
        },
        "fans": {               #DepthFL,  HeteroFL ,  ScaleFL
            "small":    0.125,  #["1_2_1", "4_2_0.25", "1_2_0.25"],
            "medium":   0.25,   #["2_2_1", "4_2_0.5",  "2_2_0.5"],
            "large":    0.5,    #["3_2_1", "4_2_0.75", "3_2_0.75"],
            "server":   1       #["4_2_1"]
        },
        "standalone": {
            "small":    ["1_2_0.25"],
            "medium":   ["2_2_0.5"],
            "large":    ["3_2_0.75"],
            "server":   ["4_2_1"]
        },
    }

    if args.task == 'cifar10':
        args.client_model_configs = {}
        # server
        idx = 0
        for name, num in device_distribution.items():
            for _ in range(num):
                args.client_model_configs[idx] = model_configs[args.algorithm][name]
                idx += 1

    elif args.task == 'cifar100':
        args.client_model_configs = {}
        # server
        idx = 0
        for name, num in device_distribution.items():
            for _ in range(num):
                args.client_model_configs[idx] = model_configs[args.algorithm][name]
                idx += 1
    elif args.task == 'mnli':
        args.client_model_configs = {}
        # server
        idx = 0
        for name, num in device_distribution.items():
            for _ in range(num):
                args.client_model_configs[idx] = model_configs[args.algorithm][name]
                idx += 1

    args.metrics = ['ACC']

    if args.task == 'cifar10':
        args.n_class = 10
    elif args.task == 'cifar100':
        args.n_class = 100
    elif args.task == 'mnist':
        args.n_class = 10
    elif args.task == 'mnli':
        args.n_class = 3
        args.epochs = 1

    args.logger.critical(args)

    server = Server(args)
    args.logger.debug('Server created.')

    for client_id, (client_ip, client_port) in client_addr.items():
        server.register_client(client_id, client_ip, client_port)

    # 初始化检查点路径
    checkpoint_path = os.path.join(save_dir, "training_checkpoint.pth")


    server.train(args)

    del server


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4321, help="random seed")
    parser.add_argument('-g', '--gpu', type=int, default="0", help="gpu id")
    # training & communication
    parser.add_argument('-p', '--port', type=int, default=12345, help="server port")
    parser.add_argument('--client_ip', type=str, help="client IP, see output of run_client.py")
    parser.add_argument('--cp', action='append', help="client ports")
    parser.add_argument('--save_dir', type=str, default="logs/")
    parser.add_argument('--device', choices=['cuda', 'cpu'], help="use cuda or cpu")
    parser.add_argument('--buffer_size', type=int, default=1048576)
    parser.add_argument('--timeout', type=int, default=7200)
    # configuration
    parser.add_argument('-t', '--task', choices=['cifar10', 'cifar100', 'mnist', 'mnli'], default='cifar10',
                        help="task name")
    # parser.add_argument('--scaling', choices=['width', 'depth', '2d'], default='width',
    #                     help="model scaling strategy. P.S. this is not used in our code")
    parser.add_argument('--n_large', type=int, help="number of large devices")
    parser.add_argument('--large_share', type=float, default=0.5, help="percentage of data on the large device")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha for dirichlet distribution")
    parser.add_argument('--temp', type=int, default=1, help="temperature for self distillation")
    parser.add_argument('--dyn_alpha', type=float, default=0.01, help="beta for self distillation of kl")
    parser.add_argument('--beta', type=float, default=0.1, help="beta for self distillation of kl")

    parser.add_argument('--total_clients', type=int, default=None, help="number of total clients")
    parser.add_argument('--sample_clients', type=int, default=10, help="number of clients join training at each round")
    parser.add_argument('-e', '--epochs', type=int, default=5, help="number of training epochs per round")
    parser.add_argument('--finetune_epochs', type=int, default=1, help="number of training epochs per round")
    parser.add_argument('-r', '--rounds', type=int, default=50, help="number of communication rounds")
    parser.add_argument('-m', '--method', type=str, choices=['s-l', 'l-s'], default='s-l',
                        help="Progressive Training way, s-l means from small to large, l-s means from large to small")
    parser.add_argument('--all_small', action='store_true', help="ALL Small client")
    parser.add_argument('--standalone', action='store_true', help="client run standalone")
    parser.add_argument('--warm_up', action='store_true', help="warm up")
    # parser.add_argument('--runtime_random', action='store_true', help="random choice model config at runtime")
    # model name
    parser.add_argument('--algorithm', type=str, choices=["scalefl", "depthfl", "fans", "heterofl", "standalone"], default="fans",
                        help="algorithms")

    # model parameter
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate of hypernet")
    parser.add_argument('--momentum', type=float, default=0.001, help="momentum of hypernet")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="weight_decay of hypernet")

    parser.add_argument('--node_hid', type=int, default=128, help="node embedding dimension")
    parser.add_argument('--trs', type=bool, default=True, help="whether track the running_mean and running_var")
    parser.add_argument('--subnet_way', type=str, default="none", choices=['none', 'l1'],
                        help="whether track the running_mean and running_var")


    parser.add_argument('--round_alpha', type=float, default=1, help="proportion of rounds for stage 1")
    parser.add_argument('--round_beta', type=float, default=0, help="proportion of rounds for stage 2")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.task == 'cifar10':
        args.rounds = 50
        args.batch_size = 128
        args.epochs = 5
        args.lr = 0.1
        args.momentum = 0.9
        args.weight_decay = 1e-4
        args.width_ratio_list = [1, 0.75, 0.5, 0.25]
    elif args.task == 'cifar100':
        args.rounds = 500
        args.batch_size = 128
        args.epochs = 5
        args.lr = 0.1
        args.momentum = 0.9
        args.weight_decay = 1e-4
        args.width_ratio_list = [1, 0.5]
    elif args.task == 'mnli':
        args.rounds = 50
        args.batch_size = 128
        args.epochs = 1
        args.lr = 3e-5
        args.momentum = 0.9
        args.weight_decay = 1e-4
        args.width_ratio_list = [1, 0.75, 0.5, 0.25]

    if args.task == 'cifar10':
        args.n_full = (1, 50)
        args.n_large = (1, 20)
        args.n_medium = (2, 10)
        args.n_small = (5, 2)
    elif args.task == 'cifar100':
        args.n_full = (1, 50)
        args.n_large = (2, 5)
        args.n_medium = (5, 2)
        args.n_small = (30, 1)
    elif args.task == 'mnli':
        args.n_full = (1, 50)
        args.n_large = (2, 10)
        args.n_medium = (4, 5)
        args.n_small = (10, 1)


    args.total_clients = args.n_full[0] + args.n_large[0] + args.n_medium[0] + args.n_small[0]

    if args.algorithm == 'heterofl':
        args.fed_dyn = False
        args.width_ratio_list = [1, 0.5, 0.25, 0.125]
        args.batch_size = 16
        args.use_scaler = True
        args.trs = False
    elif args.algorithm == 'depthfl':
        args.fed_dyn = True
        args.dyn_alpha = 0.1
        args.width_ratio_list = [0]
        args.use_scaler = False
        args.trs = True
    elif args.algorithm == 'scalefl':
        args.fed_dyn = False
        args.use_scaler = True
        args.width_ratio_list = [1, 0.5, 0.5, 0.125]
        args.round_alpha = 0.25
        args.round_beta = 0.75
        args.trs = False
    elif args.algorithm == 'fans':
        args.fed_dyn = False
        args.warm_up = False
        args.dyn_alpha = 0.1
        args.temp = 1
        args.beta = 0.6
        args.round_alpha = 0
        args.round_beta = 1
        args.use_scaler = False
        args.trs = True
    else:
        args.fed_dyn = False
        args.width_ratio_list = [1, 0.5, 0.25, 0.125]
        args.use_scaler = False
        args.trs = True

    client_clusters = [(args.client_ip, int(p)) for p in args.cp]

    client_addr = {i: client_clusters[i % len(client_clusters)] for i in range(args.total_clients - args.n_full[0])}
    for i in range(args.n_full[0]):
        client_addr[args.total_clients - args.n_full[0] + i] = client_clusters[-((i + 1) % len(client_clusters))]

    main(args)
