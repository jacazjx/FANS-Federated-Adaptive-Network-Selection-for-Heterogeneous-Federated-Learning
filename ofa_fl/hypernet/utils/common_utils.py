import copy
import random
from collections import OrderedDict
import math
from typing import List, Optional, Dict

import numpy as np

__all__ = [
    "min_divisible_value",
    "make_divisible",
    "build_activation",
    "val2list",
    "get_same_padding",
    "is_subnet",
    "config_to_matrix",
    "matrix_to_config",

    "adjust_bn_according_to_idx",

    "get_net_device",
    "sub_filter_start_end",
    "get_kv_parameter",
    "set_seed",
    "prepare_client_weights",
    "SuperWeightAveraging"
]

import torch
import torch.nn.functional as F
import torch.nn as nn

def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)

def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    elif act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hsigmoid()"


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hswish()"


def ks2list(max_ks, min_ks=1):
    assert max_ks % 2 != 0, "kernel size should be odd number"
    return list(range(min_ks, max_ks + 1, 2))


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def is_subnet(matrix_A, matrix_B):
    for a, b in zip(matrix_A, matrix_B):
        if a > b:
            return False
    return True

def config_to_matrix(config, depth_list):
    """
    将配置转换为1D向量
    """
    depth = sum(depth_list)
    matrix = np.zeros(depth)
    if isinstance(config, dict):
        start = 0
        for l in range(len(config)):
            sub_config= config[f'sublayer_{l}']
            s_l = len(sub_config)
            ratio = sub_config["block_0"]
            matrix[start:start+s_l] = ratio
            start += depth_list[l]
    else:
        matrix[:config[0]] = config[1]
    return matrix

def matrix_to_config(matrix: np.ndarray, depth_list, to_dict=True):
    """
    将1D向量转换为配置字典，并去掉值为0的块
    """
    if to_dict:
        config = {}
        start = 0
        for l in range(len(depth_list)):
            sublayer_name = f'sublayer_{l}'
            sub_config = {}
            s_l = depth_list[l]
            for block_idx in range(s_l):
                block_name = f'block_{block_idx}'
                ratio = matrix[start + block_idx]
                if ratio != 0:  # 只添加非零的块
                    sub_config[block_name] = ratio
            if sub_config:  # 只添加非空的子层
                config[sublayer_name] = sub_config
            start += s_l
    else:
        config = (np.count_nonzero(matrix), matrix[0])
    return config

def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_net_device(net):
    return net.parameters().__next__().device


def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SuperWeightAveraging:
    def __init__(self, supernet: nn.Module, alpha=1.0, fed_dyn=False):
        self.supernet = supernet
        self.fed_dyn = fed_dyn
        self.alpha = alpha
        self.h_variate = {}
        self.is_weight = {}
        self._init_counters()

    def _init_counters(self, init=True):
        for key, supernet_para in self.supernet.state_dict().items():
            self.h_variate[key] = torch.zeros_like(supernet_para.data)
            if "weight" not in key and "bias" not in key:
                self.is_weight[key] = False
            else:
                self.is_weight[key] = True


    def aggregate(self, client_models: List[OrderedDict], client_weights: Optional[List[float]]):
        # 初始化参数和计数器
        aggregate_parameters = self.supernet.state_dict()
        for key, para in aggregate_parameters.items():
            aggregate_para = torch.zeros_like(para)
            counter = torch.zeros_like(para)
            _h_variate = torch.zeros_like(para)
            for idx, client in enumerate(client_models):
                if key not in client:
                    continue
                client_size = client[key].size()
                slices = [slice(None, dim) for dim in client_size]
                if "weight" in key or "bias" in key or "mean" in key or "var" in key:
                    if client_weights is None:
                        aggregate_para[slices] += client[key]
                        counter[slices] += 1
                    else:
                        aggregate_para[slices] += (client[key] * client_weights[idx])
                        counter[slices] = 1
                    if self.is_weight[key] and self.fed_dyn:
                        _h_variate[slices] += (client[key] - para[slices])
                elif "var" in key:
                    local_mean_key = key.replace("var", "mean")
                    global_mean_key = key.replace("var", "mean")
                    local_mean = client[local_mean_key]
                    global_mean = aggregate_parameters[global_mean_key][slices]
                    if client_weights is None:
                        aggregate_para[slices] += (client[key] + (local_mean - global_mean) ** 2)
                        counter[slices] += 1
                    else:
                        aggregate_para[slices] += (client[key] + (local_mean - global_mean) ** 2) * client_weights[idx]
                        counter[slices] = 1
                else:
                    aggregate_para += client[key]
                    counter += 1

            aggregate_para[counter > 0] = (aggregate_para[counter > 0] / counter[counter > 0]).type(para.dtype)
            if self.is_weight[key] and self.fed_dyn:
                self.h_variate[key][counter > 0] = self.h_variate[key][counter > 0] - (
                            self.alpha * _h_variate[counter > 0] / counter[counter > 0])
                aggregate_para[counter > 0] = aggregate_para[counter > 0] - self.h_variate[key][counter > 0] / self.alpha

            para[counter > 0] = aggregate_para[counter > 0].type(para.dtype)

        self.supernet.load_state_dict(aggregate_parameters)
def get_kv_parameter(model, classifier_name):
    kv = {}
    for k, p in model.named_parameters():
        if classifier_name in k:
            continue
        kv[k] = p
    return OrderedDict(kv)



def convert_model_to_dict(net: nn.Module, trs=True):
    if trs:
        return net.state_dict()
    else:
        return OrderedDict(net.named_parameters())


def prepare_client_weights(model, model_name, recv_weight):
    new_model_weights = {}
    for k, p in model.state_dict().items():
        if model_name + '.' + k in recv_weight:
            global_k = model_name + '.' + k
            if p.size() == recv_weight[global_k].size():
                new_model_weights[k] = recv_weight[global_k].cpu()
        elif k in recv_weight and p.size() == recv_weight[k].size():
            new_model_weights[k] = recv_weight[k].cpu()

    return new_model_weights


def adjust_learning_rate(optimizer, init_lr, epoch, n_epochs=None, batch=0, lr_schedule_type="exp", nBatch=None):
    """
    Adjust the learning rate for the optimizer based on the specified schedule.

    Parameters:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        epoch (int): Current epoch number.
        init_lr (float): Initial learning rate.
        n_epochs (int): Total number of epochs for training.
        batch (int): Current batch index in the epoch (used for finer-grained adjustments).
        lr_schedule_type (str): Type of learning rate schedule ("constant", "step", "cosine", "exp").
        nBatch (int): Total number of batches in each epoch (used for per-batch learning rate adjustments).

    Returns:
        float: The adjusted learning rate.
    """
    # Calculate the new learning rate based on the specified schedule
    if lr_schedule_type == "constant":
        # Constant learning rate
        new_lr = init_lr

    elif lr_schedule_type == "step":
        # Step decay: reduce lr by a factor every 1/3 of total epochs
        decay_factor = 0.1
        if epoch >= n_epochs * 2 / 3:
            new_lr = init_lr * (decay_factor ** 2)
        elif epoch >= n_epochs / 3:
            new_lr = init_lr * decay_factor
        else:
            new_lr = init_lr

    elif lr_schedule_type == "cosine":
        # Cosine annealing
        total_steps = n_epochs * (nBatch or 1)
        current_step = epoch * (nBatch or 1) + batch
        new_lr = 0.5 * init_lr * (1 + math.cos(math.pi * current_step / total_steps))

    elif lr_schedule_type == "exp":
        # Exponential decay
        decay_rate = 0.97  # Adjust decay rate as needed
        new_lr = init_lr * (decay_rate ** epoch)

    else:
        raise ValueError("Unsupported lr_schedule_type. Use 'constant', 'step', 'cosine', or 'exp'.")

    # Update the optimizer with the new learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

    return new_lr

if __name__ == "__main__":
    print(ks2list(5))
