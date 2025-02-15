import math
import torch.nn as nn
import torch.nn.functional as F

from hypernet.utils.common_utils import min_divisible_value


def init_models(net, model_init="he_fout"):
    """
    Conv2d,
    BatchNorm2d, BatchNorm1d, GroupNorm
    Linear,
    """
    if isinstance(net, list):
        for sub_net in net:
            init_models(sub_net, model_init)
        return
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if model_init == "he_fout":
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif model_init == "he_fin":
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            else:
                raise NotImplementedError
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) in [nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm]:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.zero_()


def replace_bn_with_gn(model, gn_channel_per_group):
    if gn_channel_per_group is None:
        return

    for m in model.modules():
        to_replace_dict = {}
        for name, sub_m in m.named_children():
            if isinstance(sub_m, nn.BatchNorm2d):
                num_groups = sub_m.num_features // min_divisible_value(
                    sub_m.num_features, gn_channel_per_group
                )
                gn_m = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=sub_m.num_features,
                    eps=sub_m.eps,
                    affine=True,
                )

                # load weight
                gn_m.weight.data.copy_(sub_m.weight.data)
                gn_m.bias.data.copy_(sub_m.bias.data)
                # load requires_grad
                gn_m.weight.requires_grad = sub_m.weight.requires_grad
                gn_m.bias.requires_grad = sub_m.bias.requires_grad

                to_replace_dict[name] = gn_m
        m._modules.update(to_replace_dict)


def gen_data_shares(num_clients, num_servers=1):

    # 设备比例划分
    large_ratio = 0.2  # 大型设备占 20%
    medium_ratio = 0.3  # 中型设备占 30%
    small_ratio = 0.5  # 小型设备占 50%

    server_complexity = 20
    large_complexity = 10
    medium_complexity = 5
    small_complexity = 2


    # 客户端类型划分
    num_large_clients = int(num_clients * large_ratio)
    num_medium_clients = int(num_clients * medium_ratio)
    num_small_clients = num_clients - num_large_clients - num_medium_clients

    # 设备数量分布
    device_distribution = {
        'small': num_small_clients,
        'medium': num_medium_clients,
        'large': num_large_clients,
        'server': num_servers,  # 服务器只有1台
    }

    # 总复杂度
    total_complexity = (
        server_complexity +
        num_large_clients * large_complexity +
        num_medium_clients * medium_complexity +
        num_small_clients * small_complexity
    )

    # 计算各类设备的数据占比
    server_share = server_complexity / total_complexity
    large_total_share = (large_complexity * num_large_clients) / total_complexity
    medium_total_share = (medium_complexity * num_medium_clients) / total_complexity
    small_total_share = (small_complexity * num_small_clients) / total_complexity

    # 各类设备下每个客户端的数据占比
    large_client_share = large_total_share / num_large_clients
    medium_client_share = medium_total_share / num_medium_clients
    small_client_share = small_total_share / num_small_clients

    assert round(server_share + large_client_share * num_large_clients +
                 medium_client_share * num_medium_clients +
                 small_client_share * num_small_clients, 2) == 1.0, \
        "数据占比之和不等于1"

    data_share = []
    data_share.extend([small_client_share] * num_small_clients)
    data_share.extend([medium_client_share] * num_medium_clients)
    data_share.extend([large_client_share] * num_large_clients)
    data_share.append(server_share)

    return device_distribution, data_share


def alingn_model_weight(model):
    for name, param in model.named_parameters():
        if param.dim() == 1: # bias
            nn.init.xavier_uniform_(param)