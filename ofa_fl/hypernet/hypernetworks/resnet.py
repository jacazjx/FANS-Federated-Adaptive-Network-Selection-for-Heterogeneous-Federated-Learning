'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import copy
import random
from collections import OrderedDict
from typing import Dict, Union, List

import numpy as np
from torchvision.models import ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from hypernet.datasets.load_cifar import load_cifar, get_datasets
from hypernet.utils.common_utils import make_divisible, adjust_bn_according_to_idx, get_net_device, DynamicWeightedSampler
from hypernet.base.dynamic_modules import DynamicBatchNorm2d, DynamicConv2d, Scaler, MyIdentity, DynamicGroupNorm, \
    DynamicLinear
from itertools import product

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channel=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_emb=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if return_emb:
            return out, self.classifier(out)
        else:
            return self.classifier(out)

    def forward_shallow(self, x, return_emb=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.adapter(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if return_emb:
            return out, self.layer1_classifier(out)
        else:
            return self.layer1_classifier(out)


'''
"1_1_0.5": {  # Smallest Resnet
        "sublayer_0": {"block_0": 0.5}
},
"4_2_0": {  # Full Resnet
        "sublayer_0": {"block_0": 0, "block_1": 0},
        "sublayer_1": {"block_0": 0, "block_1": 0},
        "sublayer_2": {"block_0": 0, "block_1": 0},
        "sublayer_3": {"block_0": 0, "block_1": 0},
    },
'''

class DynamicShortCut(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, norm_way="bn", track_running_stats=True):
        super(DynamicShortCut, self).__init__()
        self.stride = stride
        self.norm_way = norm_way
        self.track_running_stats = track_running_stats

        self.conv = DynamicConv2d(in_planes, out_planes, kernel_size=1, stride=stride)
        if norm_way == "bn":
            self.norm = DynamicBatchNorm2d(out_planes, track_running_stats=track_running_stats)
        elif norm_way == "in":
            self.norm = DynamicGroupNorm(out_planes, out_planes)
        elif norm_way == "ln":
            self.norm = DynamicGroupNorm(1, out_planes)
        else:
            raise NotImplementedError("norm_way must be in, bn, ln")

    def forward(self, x, planes=None):
        return self.norm(self.conv(x, planes))



class DynamicBasicBlock(nn.Module):
    expansion = 1
    CHANNEL_DIVISIBLE = 8

    def __init__(self,
                 in_planes,
                 out_planes,
                 mid_planes=None,
                 stride=1,
                 norm_way="bn",
                 scale_ratio=1.0,
                 use_scaler=False,
                 track_running_stats=True):
        super(DynamicBasicBlock, self).__init__()
        self.stride = stride
        self.norm_way = norm_way
        self.use_scaler = use_scaler
        self.scaler = Scaler(scale_ratio) if use_scaler else MyIdentity()
        self.track_running_stats = track_running_stats

        self.default_in_channels = in_planes
        self.default_mid_channels = mid_planes if mid_planes is not None else out_planes
        self.default_out_channels = out_planes * self.expansion
        self.active_in_channels = self.default_in_channels
        self.active_mid_channels = self.default_mid_channels
        self.active_out_channels = self.default_out_channels

        self.conv1 = DynamicConv2d(self.active_in_channels, self.active_mid_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = DynamicConv2d(self.active_mid_channels, self.active_out_channels, kernel_size=3, stride=1, padding=1)
        if norm_way == "bn":
            self.bn1 = DynamicBatchNorm2d(self.active_mid_channels, track_running_stats=track_running_stats)
            self.bn2 = DynamicBatchNorm2d(self.active_out_channels, track_running_stats=track_running_stats)
        elif norm_way == "in":
            self.bn1 = DynamicGroupNorm(self.active_mid_channels, self.active_mid_channels)
            self.bn2 = DynamicGroupNorm(self.active_out_channels, self.active_out_channels)
        elif norm_way == "ln":
            self.bn1 = DynamicGroupNorm(1, self.active_mid_channels)
            self.bn2 = DynamicGroupNorm(1, self.active_out_channels)
        else:
            raise NotImplementedError("norm_way must be in, bn, ln")

        if stride != 1 or in_planes != out_planes:
            self.shortcut = DynamicShortCut(self.active_in_channels, self.active_out_channels, norm_way=norm_way, stride=stride)
        else:
            self.shortcut = MyIdentity()

    # in_plane == out_plane
    def forward(self, x, planes=None):
        if planes is None:
            planes = self.active_mid_channels
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.scaler(self.conv1(x, planes))))
        out = self.bn2(self.scaler(self.conv2(out)))

        out = out + residual
        out = F.relu(out)
        return out

    def get_subnet(self, in_channels, out_channels):
        device = get_net_device(self)
        subnet = DynamicBasicBlock(
            in_channels,
            self.active_out_channels,
            mid_planes=out_channels,
            stride=self.stride,
            norm_way=self.norm_way,
            scale_ratio=out_channels / self.default_out_channels,
            use_scaler=self.use_scaler,
            track_running_stats=self.track_running_stats
        ).to(device)

        subnet.default_in_channels = self.default_in_channels
        subnet.default_mid_channels = self.default_mid_channels
        subnet.default_out_channels = self.default_out_channels
        subnet.scaler = self.scaler
        subnet.conv1.copy_conv(self.conv1)
        subnet.bn1._copy(self.bn1)
        subnet.conv2.copy_conv(self.conv2)
        subnet.bn2._copy(self.bn2)

        if not isinstance(self.shortcut, MyIdentity):
            subnet.shortcut.conv.copy_conv(self.shortcut.conv)
            subnet.shortcut.norm._copy(self.shortcut.norm)
        return subnet

    def re_organize_weights(self, expand_ratio_list):
        importance = torch.norm(self.conv2.conv.weight, p=1, dim=(0, 2, 3))

        sorted_expand_list = copy.deepcopy(expand_ratio_list)
        sorted_expand_list.sort(reverse=True)
        target_width_list = [
            make_divisible(
                round(self.active_out_channels * expand),
                8,
            )
            for expand in sorted_expand_list
        ]
        right = len(importance)
        base = -len(target_width_list) * 1e5
        for i in range(len(expand_ratio_list)):
            left = target_width_list[i]
            importance[left:right] += base
            base += 1e5
            right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.conv2.conv.weight.data = torch.index_select(
            self.conv2.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.bn1, idx=sorted_idx)
        self.conv1.conv.weight.data = torch.index_select(
            self.conv1.conv.weight.data, 0, sorted_idx
        )


class DynamicBottleneck(nn.Module):
    expansion = 4
    CHANNEL_DIVISIBLE = 8

    def __init__(self,
                 in_planes,
                 out_planes,
                 mid_planes=None,
                 stride=1,
                 norm_way="bn",
                 scale_ratio=1.0,
                 use_scaler=False,
                 track_running_stats=True):
        super(DynamicBottleneck, self).__init__()
        self.stride = stride
        self.norm_way = norm_way
        self.use_scaler = use_scaler
        self.scaler = Scaler(scale_ratio) if use_scaler else MyIdentity()
        self.track_running_stats = track_running_stats

        self.default_in_channels = in_planes
        self.default_mid_channels = mid_planes if mid_planes is not None else out_planes
        self.default_out_channels = out_planes
        self.active_in_channels = self.default_in_channels
        self.active_mid_channels = self.default_mid_channels
        self.active_out_channels = self.default_out_channels

        self.conv1 = DynamicConv2d(self.active_in_channels, self.active_mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = DynamicConv2d(self.active_mid_channels, self.active_mid_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = DynamicConv2d(self.active_mid_channels, self.active_out_channels, kernel_size=1, stride=1, padding=0)

        if norm_way == "bn":
            self.bn1 = DynamicBatchNorm2d(self.active_mid_channels, track_running_stats=track_running_stats)
            self.bn2 = DynamicBatchNorm2d(self.active_mid_channels, track_running_stats=track_running_stats)
            self.bn3 = DynamicBatchNorm2d(self.active_out_channels, track_running_stats=track_running_stats)
        elif norm_way == "in":
            self.bn1 = DynamicGroupNorm(self.active_mid_channels, self.active_mid_channels)
            self.bn2 = DynamicGroupNorm(self.active_mid_channels, self.active_mid_channels)
            self.bn3 = DynamicGroupNorm(self.active_out_channels, self.active_out_channels)
        elif norm_way == "ln":
            self.bn1 = DynamicGroupNorm(1, self.active_mid_channels)
            self.bn2 = DynamicGroupNorm(1, self.active_mid_channels)
            self.bn3 = DynamicGroupNorm(1, self.active_out_channels)
        else:
            raise NotImplementedError("norm_way must be in, bn, ln")

        if stride != 1 or in_planes != out_planes:
            self.shortcut = DynamicShortCut(self.active_in_channels, self.active_out_channels, norm_way=norm_way, stride=stride)
        else:
            self.shortcut = MyIdentity()

    def forward(self, x, planes=None):
        if planes is None:
            planes = self.active_mid_channels
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.scaler(self.conv1(x, planes))))
        out = F.relu(self.bn2(self.scaler(self.conv2(out, planes))))
        out = self.bn3(self.scaler(self.conv3(out)))

        out = out + residual
        out = F.relu(out)
        return out

    def get_subnet(self, in_channels, out_channels):
        device = get_net_device(self)
        subnet = DynamicBottleneck(
            in_channels,
            self.active_out_channels,
            mid_planes=out_channels,
            stride=self.stride,
            scale_ratio=out_channels / self.default_out_channels,
            use_scaler=self.use_scaler,
            track_running_stats=self.track_running_stats
        ).to(device)

        subnet.default_in_channels = self.default_in_channels
        subnet.default_mid_channels = self.default_mid_channels
        subnet.default_out_channels = self.default_out_channels
        subnet.scaler = self.scaler
        subnet.conv1.copy_conv(self.conv1)
        subnet.bn1._copy(self.bn1)
        subnet.conv2.copy_conv(self.conv2)
        subnet.bn2._copy(self.bn2)
        subnet.conv3.copy_conv(self.conv3)
        subnet.bn3._copy(self.bn3)

        if not isinstance(self.shortcut, MyIdentity):
            subnet.shortcut.conv.copy_conv(self.shortcut.conv)
            subnet.shortcut.norm._copy(self.shortcut.norm)
        return subnet

    def re_organize_weights(self, expand_ratio_list):
        importance = torch.norm(self.conv3.conv.weight, p=1, dim=(0, 2, 3))

        sorted_expand_list = copy.deepcopy(expand_ratio_list)
        sorted_expand_list.sort(reverse=True)
        target_width_list = [
            make_divisible(
                round(self.active_out_channels * expand),
                8,
            )
            for expand in sorted_expand_list
        ]
        right = len(importance)
        base = -len(target_width_list) * 1e5
        for i in range(len(expand_ratio_list)):
            left = target_width_list[i]
            importance[left:right] += base
            base += 1e5
            right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.conv3.conv.weight.data = torch.index_select(
            self.conv3.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.bn2, idx=sorted_idx)
        self.conv2.conv.weight.data = torch.index_select(
            self.conv2.conv.weight.data, 0, sorted_idx
        )

        importance = torch.norm(self.conv2.conv.weight, p=1, dim=(0, 2, 3))

        sorted_expand_list = copy.deepcopy(expand_ratio_list)
        sorted_expand_list.sort(reverse=True)
        target_width_list = [
            make_divisible(
                round(self.active_out_channels * expand),
                8,
            )
            for expand in sorted_expand_list
        ]
        right = len(importance)
        base = -len(target_width_list) * 1e5
        for i in range(len(expand_ratio_list)):
            left = target_width_list[i]
            importance[left:right] += base
            base += 1e5
            right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.conv2.conv.weight.data = torch.index_select(
            self.conv2.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.bn2, idx=sorted_idx)
        self.conv1.conv.weight.data = torch.index_select(
            self.conv1.conv.weight.data, 0, sorted_idx
        )




class DynamicTransition(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale_ratio=1.0, use_scaler=False, track_running_stats=True):
        super(DynamicTransition, self).__init__()
        self.stride = stride
        self.conv = DynamicConv2d(in_planes, out_planes, kernel_size=1, stride=stride)
        self.bn = DynamicBatchNorm2d(out_planes, track_running_stats=track_running_stats)
        # self.bn = DynamicGroupNorm(out_planes, out_planes)
        self.use_scaler = use_scaler
        self.scaler = Scaler(scale_ratio) if use_scaler else MyIdentity()
        self.track_running_stats = track_running_stats
        self.default_out_channels = out_planes
        self.default_in_channels = in_planes
        self.active_out_planes = self.default_out_channels
        self.active_in_planes = self.default_in_channels

    def forward(self, x, out_planes=None):
        in_planes = x.size(1)
        if out_planes is None:
            out_planes = self.default_out_channels
        ratio = out_planes / self.default_out_channels

        out = F.relu(self.bn(self.scaler(self.conv(x))))
        # out = self.conv(x)
        return out

    def get_subnet(self, in_planes, out_planes, expand_ratio_list=None):
        device = next(self.parameters()).device
        subnet = DynamicTransition(
            in_planes,
            out_planes,
            stride=self.stride,
            # scale_ratio=out_planes / self.default_out_channels,
            # scale_ratio=ratio,
            use_scaler=self.use_scaler,
            track_running_stats=self.track_running_stats
        ).to(device)

        subnet.default_in_channels = self.default_in_channels
        subnet.default_out_channels = self.default_out_channels
        subnet.scaler = self.scaler
        subnet.bn._copy(self.bn)
        subnet.conv.copy_conv(self.conv)
        return subnet


class SuperResnet(nn.Module):
    NUM_SUBLAYER = 4
    BASE_DEPTH_LIST = [2, 2, 4, 2]
    STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

    def __init__(self, block, num_sublayer=4,
                 depth_list=[2, 2, 4, 2],
                 width_list=[64, 128, 256, 512],
                 input_size=(1, 3, 32, 32),
                 num_classes=10,
                 stride_list=[1, 2, 2, 2],
                 width_pruning_ratio_list=[1, 0.75, 0.5, 0.25],
                 use_scaler=False,
                 track_running_stats=True,
                 norm_way="bn",
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        # For copy subnet
        self.batch_size, self.input_channel, self.width, self.length = input_size
        self.block = block
        self.num_classes = num_classes
        self.width_pruning_ratio_list = width_pruning_ratio_list
        self.NUM_SUBLAYER = num_sublayer
        assert self.NUM_SUBLAYER == len(depth_list), "num_sublayer must be equal to len(num_blocks)"
        self.BASE_DEPTH_LIST = depth_list
        self.STAGE_WIDTH_LIST = width_list
        self.track_running_stats = track_running_stats
        self.norm_way = norm_way
        self.use_scaler = use_scaler
        self.stride_list = stride_list

        in_planes = self.STAGE_WIDTH_LIST[0]
        norm_module = {
            "bn": nn.BatchNorm2d(in_planes, track_running_stats=self.track_running_stats),
            "in": nn.GroupNorm(in_planes, in_planes),
            "ln": nn.GroupNorm(1, in_planes),
        }
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(self.input_channel, in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_module[norm_way],
            nn.ReLU()
        )

        hidden_layer = []
        for i in range(self.NUM_SUBLAYER):
            sub_layer = self._make_layer(block,
                                         in_planes,
                                         self.STAGE_WIDTH_LIST[i],
                                         self.BASE_DEPTH_LIST[i],
                                         self.stride_list[i])
            in_planes = self.STAGE_WIDTH_LIST[i] * block.expansion
            hidden_layer.append((f"sublayer_{i}", sub_layer))

        self.hidden_layer = nn.Sequential(OrderedDict(hidden_layer))

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            DynamicLinear(self.STAGE_WIDTH_LIST[-1] * block.expansion, num_classes),
        )

        # client_idx: subnet_config
        self.subnet_width_progress_configs = None
        self.subnet_depth_progress_configs = None
        self.dynamic_subnet_sampler = None

        self.default_output_path = self.get_default_config()
        self.active_output_path = self.default_output_path
        self.smallest_output_path = {"sublayer_0": {"block_0": width_pruning_ratio_list[-1]}}

        vec = torch.tensor(self.STAGE_WIDTH_LIST) / torch.sum(
            torch.tensor(self.BASE_DEPTH_LIST) * torch.tensor(self.STAGE_WIDTH_LIST))
        self.layer_cost = torch.repeat_interleave(vec, torch.tensor(self.BASE_DEPTH_LIST))


    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        sub_layer = []
        for idx, stride in enumerate(strides):
            sub_layer.append((f"block_{idx}", block(in_planes,
                                                    out_planes*block.expansion,
                                                    mid_planes=out_planes,
                                                    stride=stride,
                                                    norm_way=self.norm_way,
                                                    scale_ratio=max(self.width_pruning_ratio_list),
                                                    use_scaler=self.use_scaler,
                                                    track_running_stats=self.track_running_stats)))
            in_planes = out_planes * block.expansion
        return nn.Sequential(OrderedDict(sub_layer))

    def get_default_config(self):
        default_config = {f"sublayer_{i}": {} for i in range(self.NUM_SUBLAYER)}
        for sublayer_idx in range(self.NUM_SUBLAYER):
            default_config[f'sublayer_{sublayer_idx}'] = {}
            for block_idx in range(self.BASE_DEPTH_LIST[sublayer_idx]):
                default_config[f'sublayer_{sublayer_idx}'][f'block_{block_idx}'] = max(self.width_pruning_ratio_list)

        return default_config

    def get_progressive_subnet_configs(self, A=5, stage=0):

        subnetwork_configs = []

        IndependentRandomSampling = 0
        RecursiveRandomSampling = 1
        WeightedRandomSampling = 2

        '''Independent Random sample'''
        if stage == IndependentRandomSampling:
            for _ in range(A):
                subnetwork_config = self.random_sample_subnet_config()
                if subnetwork_config not in subnetwork_configs:
                    subnetwork_configs.append(subnetwork_config)
            subnetwork_configs = subnetwork_configs[:A]
        elif stage == RecursiveRandomSampling:
            '''Recursive Random sample'''
            last_subnet_config = self.active_output_path
            while True:
                subnetwork_config = self.random_sample_subnet_config(last_subnet_config)
                if last_subnet_config == self.smallest_output_path or last_subnet_config==subnetwork_config:
                    break
                subnetwork_configs.append(subnetwork_config)
                last_subnet_config = subnetwork_config
        elif stage == WeightedRandomSampling:
            '''Dynamic Weighted Sample'''
            if self.dynamic_subnet_sampler is None:
                self.all_subnet_configs = self.generate_all_subnet_configs()[1:]
                self.dynamic_subnet_sampler = DynamicWeightedSampler(len(self.all_subnet_configs), decay_factor=0.9)
            subnetwork_configs = [self.all_subnet_configs[idx] for idx in self.dynamic_subnet_sampler.sample(A)]

        return subnetwork_configs

    # 随机生成前向路径索引
    def random_sample_subnet_config(self, max_net_config=None):
        if max_net_config is None:
            max_net_config = self.active_output_path

        NUM_SUBLAYER = len(max_net_config)
        BASE_DEPTH_LIST = []
        for sublayer_idx in range(NUM_SUBLAYER):
            sublayer_name = f"sublayer_{sublayer_idx}"
            BASE_DEPTH_LIST.append(len(max_net_config[sublayer_name]))
        subnetwork_config = {}

        # # 随机生成每个隐藏层的子层数量（至少为1）
        num_sublayers = random.randint(1, NUM_SUBLAYER)
        for sublayer in range(num_sublayers):
            subnetwork_config[f'sublayer_{sublayer}'] = {}
            # 随机生成每个子层的块数量（至少为1）
            num_blocks = random.randint(1, BASE_DEPTH_LIST[sublayer])
            active_compression_rate = self.width_pruning_ratio_list[self.width_pruning_ratio_list.index(
                max_net_config[f'sublayer_{sublayer}'][f'block_{0}']):]
            compression_rate = random.choice(active_compression_rate)
            for block in range(num_blocks):
                subnetwork_config[f'sublayer_{sublayer}'][f'block_{block}'] = compression_rate
        return subnetwork_config

    # def generate_all_subnet_configs_old(self):
    #     all_configs = []
    #     for num_sublayer in range(self.NUM_SUBLAYER, 0, -1):
    #         self.recursion_all_subnet_configs(num_sublayer, all_configs)
    #     return all_configs

    def generate_all_subnet_configs(self):
        all_configs = []

        # 获取当前 active_output_path 的结构信息
        current_config = self.active_output_path

        num_sublayers = len(current_config)

        # 每一层最多允许的 block 数量
        max_block_num = [
            len(current_config[f"sublayer_{i}"])
            for i in range(num_sublayers)
        ]

        # 每一层允许的最大 width ratio
        max_width_ratio = [
            max(current_config[f"sublayer_{i}"].values())
            for i in range(num_sublayers)
        ]

        def recursion_all_subnet_configs(max_sublayer, all_configs, sublayer=0, current_config=None, budget=None):
            if current_config is None:
                current_config = {}

            if sublayer == max_sublayer:
                all_configs.append(copy.deepcopy(current_config))
                return

            # 只能使用 ≤ 当前层数的 block 数量
            valid_num_blocks = range(self.BASE_DEPTH_LIST[sublayer], 0, -1)
            valid_num_blocks = [n for n in valid_num_blocks if n <= max_block_num[sublayer]]

            # 只能使用 ≤ 当前最大压缩率
            valid_ratios = [r for r in self.width_pruning_ratio_list if r <= max_width_ratio[sublayer]]

            for num_blocks in valid_num_blocks:
                for compression_rate in valid_ratios:
                    current_config[f'sublayer_{sublayer}'] = {f'block_{block}': compression_rate for block in
                                                              range(num_blocks)}
                    recursion_all_subnet_configs(max_sublayer, all_configs, sublayer + 1, current_config)

        # 先从最大层数开始向下搜索
        for num_sublayer in range(self.NUM_SUBLAYER, 0, -1):
            if num_sublayer > len(current_config):
                continue  # 如果当前模型层数不够，则跳过更深层的枚举
            recursion_all_subnet_configs(num_sublayer, all_configs)

        return all_configs

    def recursion_all_subnet_configs(self, max_sublayer, all_configs, sublayer=0, current_config=None, budget=None):
        if current_config is None:
            current_config = {}

        if sublayer == max_sublayer:
            all_configs.append(current_config.copy())
            return

        for num_blocks in range(self.BASE_DEPTH_LIST[sublayer], 0, -1):
            for compression_rate in self.width_pruning_ratio_list:
                current_config[f'sublayer_{sublayer}'] = {f'block_{block}': compression_rate for block in range(num_blocks)}
                self.recursion_all_subnet_configs(max_sublayer, all_configs, sublayer + 1, current_config)

    # e.g., active_output_path = {
    #       'sublayer_0': {'block_0': 0.5},
    #       'sublayer_1': {'block_0': 0.125, 'block_1': 0.125},
    #       'sublayer_2': {'block_0': 0.125}
    #   }
    def forward(self, x, active_output_path=None, return_emb=False):
        if active_output_path is None:
            active_output_path = self.active_output_path

        x = self.embedding_layer(x)

        for sublayer_name, sublayer_configs in active_output_path.items():
            sublayer = getattr(self.hidden_layer, sublayer_name)
            for block_name, width_pruning_ratio in sublayer_configs.items():
                block = getattr(sublayer, block_name)
                if isinstance(block, MyIdentity):
                    continue
                current_out_channel = round(width_pruning_ratio * block.default_mid_channels)
                x = block(x, current_out_channel)

        if return_emb:
            out = x
            for m in self.output_layer[:-1]:
                out = m(out)
            return out, self.output_layer[-1](out)
        else:
            return self.output_layer(x)

    def re_organize_weights(self):
        active_output_path = self.active_output_path
        active_num_sublayer = len(active_output_path)
        for num_sub in range(active_num_sublayer):
            sublayer_name = f'sublayer_{num_sub}'
            active_num_block = len(active_output_path[sublayer_name])
            sublayer = self.hidden_layer[num_sub]
            for num_block in range(active_num_block):
                block_name = f'block_{num_block}'
                width_pruning_ratio = active_output_path[sublayer_name][block_name]
                idx = self.width_pruning_ratio_list.index(width_pruning_ratio)
                block = sublayer[num_block]
                block.re_organize_weights(self.width_pruning_ratio_list[idx:])

    def get_subnet(self, subnet_config=None):
        if subnet_config is None:
            subnet_config = self.active_output_path

        subnet = copy.deepcopy(self)
        subnet.active_output_path = subnet_config

        last_width_pruning_ratio = 1.0
        sub_hidden_layer = []

        last_output_channel = self.STAGE_WIDTH_LIST[0]
        for sublayer_idx in range(self.NUM_SUBLAYER):
            sublayer_name = f'sublayer_{sublayer_idx}'
            sublayer = self.hidden_layer[sublayer_idx]
            subnet_sublayer = []

            if sublayer_name not in subnet_config:
                sub_hidden_layer.append((sublayer_name, MyIdentity()))
                continue

            for block_idx in range(self.BASE_DEPTH_LIST[sublayer_idx]):
                block_name = f'block_{block_idx}'

                if block_name not in subnet_config[sublayer_name]:
                    subnet_sublayer.append((block_name, MyIdentity()))
                    continue

                block = sublayer[block_idx]
                width_pruning_ratio = subnet_config[sublayer_name][block_name]
                current_out_channel = round(width_pruning_ratio * block.default_mid_channels)
                subnet_sublayer.append((block_name, block.get_subnet(last_output_channel, current_out_channel)))
                last_output_channel = block.default_out_channels
            sub_hidden_layer.append((sublayer_name, nn.Sequential(OrderedDict(subnet_sublayer))))

        subnet.hidden_layer = nn.Sequential(OrderedDict(sub_hidden_layer))
        return subnet

def super_resnet18(trs, use_scaler, width_ratio_list: list, num_classes, input_size=(1, 3, 32, 32)):
    return SuperResnet(
        block=DynamicBasicBlock,
        num_sublayer=4,
        depth_list=[2, 2, 2, 2],
        width_list=[64, 128, 256, 512],
        input_size=input_size,
        num_classes=num_classes,
        stride_list=[1, 2, 2, 2],
        use_scaler=use_scaler,
        width_pruning_ratio_list=width_ratio_list,
        track_running_stats=trs
    )


def super_resnet101(trs, use_scaler, width_ratio_list: list):
    return SuperResnet(
        block=DynamicBottleneck,
        num_sublayer=4,
        depth_list=[3, 4, 23, 3],
        width_list=[64, 128, 256, 512],
        input_size=(1, 3, 32, 32),
        num_classes=100,
        stride_list=[1, 2, 2, 2],
        use_scaler=use_scaler,
        width_pruning_ratio_list=width_ratio_list,
        track_running_stats=trs
    )

def super_resnet50(trs, use_scaler, width_ratio_list: list):
    return SuperResnet(
        block=DynamicBottleneck,
        num_sublayer=4,
        depth_list=[3, 4, 6, 3],
        width_list=[64, 128, 256, 512],
        input_size=(1, 3, 32, 32),
        num_classes=100,
        stride_list=[1, 2, 2, 2],
        use_scaler=use_scaler,
        width_pruning_ratio_list=width_ratio_list,
        track_running_stats=trs
    )

def resnet18(input_channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_channel, num_classes)

def resnet34(input_channel, num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], input_channel, num_classes)

def resnet50(input_channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], input_channel, num_classes)




import matplotlib.pyplot as plt
import pandas as pd
from hypernet.utils.evaluation import visualize_model_parameters, count_parameters
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

def eval_all_subnets(model: SuperResnet, data_path, csv_path):

    trainData, valData, testData = get_datasets("cifar10", os.path.join(data_path, "cifar10"))
    train_set, val_set, test_set = trainData, valData, testData

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=5)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=5)

    def fine_tuning(m):
        m.train()
        with torch.no_grad():
            for data in train_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = m(images)

    def test(model, active_path=None):
        # 测试集测试准确率，召回率，F1，以及top-1,top-2,top-3
        correct = 0
        total = 0
        top_1 = 0
        top_2 = 0
        top_3 = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images, active_path)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, top_2_predicted = torch.topk(outputs, 2)
                _, top_3_predicted = torch.topk(outputs, 3)
                for i in range(labels.size(0)):
                    if labels[i] in top_2_predicted[i]:
                        top_2 += 1
                    if labels[i] in top_3_predicted[i]:
                        top_3 += 1
                    if labels[i] == predicted[i]:
                        top_1 += 1
        model_size, accuracy = count_parameters(model, 'MB'), 100 * correct / total
        print(active_path, f"Parameters:{model_size}MB")
        print(f"Accuracy: {accuracy}%")
        print(f"Top-1 Accuracy: {100 * top_1 / total}%")
        print(f"Top-2 Accuracy: {100 * top_2 / total}%")
        print(f"Top-3 Accuracy: {100 * top_3 / total}%")
        return model_size, accuracy

    all_subnet_configs = model.generate_all_subnet_configs()
    # all_subnet_configs.reverse()
    results = []

    def evaluate_subnet(subnet_config, model):

        print(subnet_config)
        subnet = model.get_subnet(subnet_config)
        fine_tuning(subnet)
        model_size, accuracy = test(subnet)
        return model_size, accuracy, subnet_config

    # for subnet_config in all_subnet_configs:
    #     print(subnet_config)
    #     subnet = model.get_subnet(subnet_config)
    #     fine_tuning(subnet)
    #     model_size, accuracy = test(subnet)
    #     results.append((model_size, accuracy))
    max_workers = 5
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_subnet, config, model) for config in all_subnet_configs]
        for future in as_completed(futures):
            try:
                model_size, accuracy, subnet_config = future.result(timeout=10)

                # 单条结果转为 DataFrame
                df = pd.DataFrame([{
                    "size": model_size,
                    "acc": accuracy,
                    "subnet_configs": str(subnet_config)
                }])

                # 追加写入 CSV（header=False 表示不重复写表头）
                df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path))

                results.append((model_size, accuracy, subnet_config))

            except Exception as e:
                print(f"Task failed: {e}")
    # print("saving results...")
    # sizes, accuracies, configs = zip(*results)
    # pd.DataFrame({
    #     "size": sizes,
    #     "acc": accuracies,
    #     "subnet_configs": configs
    # }).to_csv('resnet_result.csv')

    # plt.scatter(sizes, accuracies)
    # plt.xlabel('Model Size (MB)')
    # plt.ylabel('Accuracy (%)')
    # plt.title('Model Size vs Accuracy')
    # plt.show()



import matplotlib.pyplot as plt

def plot_mapp_histograms(mapps, labels, colors=None, title="Frequency of Subnet Configurations",
                         xlabel="Subnet Configurations", ylabel="Frequency",
                         rotation=90, figsize=(20, 8), legend_loc='upper right'):
    """
    绘制多个 mapp 的条形图，用不同颜色区分，并简化 x 轴标签。
    """

    if not mapps or not isinstance(mapps, list):
        raise ValueError("mapps 必须是非空列表")

    if colors is None:
        colors = [f"C{i}" for i in range(len(mapps))]

    if len(colors) < len(mapps):
        raise ValueError("colors 列表长度必须大于等于 mapps 列表长度")

    fig, ax = plt.subplots(nrows=4, figsize=figsize)

    x_array = np.arange(len(mapps[0].keys()))
    bar_width = 0.2
    positions = x_array*bar_width
    all_values = []
    # 自动设置 y 轴范围
    max_value = 0

    for idx, (mapp, color) in enumerate(zip(mapps, colors)):
        values = list(mapp.values())
        _sum_times = sum(values)
        all_values.append(values)
        values = np.log(np.array(values) + 1)
        max_value = max(max_value, max(values))
        _sum_zero = len(values) - np.count_nonzero(values)
        ax[idx].text(0.2, 8.0, F'#Searched: {_sum_times}')
        ax[idx].text(0.2, 9.0, F'#Zero Searched: {_sum_zero}')
        ax[idx].bar(positions, values, width=bar_width, color=color, label=f"{labels[idx]}")
        ax[idx].set_xticks([])
        ax[idx].legend(loc=legend_loc)
        ax[idx].set_ylabel(ylabel)
        ax[idx].set_ylim(0, 10)
    # 隐藏 x 轴标签和刻度

    # plt.xlim(0, len(x_array) - bar_width * (len(mapps) - 1))
    # 设置其他图形属性
    plt.xlabel("Different Subnet Configurations")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'frequency.pdf')
    plt.show()


if __name__ == '__main__':
    from torchinfo import summary
    # from heterofl.src.models.resnet import ResNet, Block
    from itertools import cycle

    batch_size = 128
    sample_sum = [805, 800, 800, 800, 800, 4002, 4003, 7991, 19999]
    # client_arch_lists = torch.load("../../logs/cifar10/seed4321/client_config")
    resnet = super_resnet18(True, False, [1.0, 0.75, 0.5, 0.25], 10)#.cuda()
    mapp = {}
    resnet.get_progressive_subnet_configs("depth")
    for subnet in resnet.generate_all_subnet_configs():
        mapp[str(subnet)] = 0
    # mapp2 = mapp.copy()
    # mapp3 = mapp.copy()
    # mapp4 = mapp.copy()
    # for i in range(50):
    #     for j, arch_list in enumerate(client_arch_lists.values()):
    #         arch_config = random.choice(arch_list)
    #         subnet = resnet.get_subnet(arch_config)
    #         for k in range(sample_sum[j]//batch_size):
    #             mapp[str(arch_config)] += 1
    #             subnet_configs = subnet.get_progressive_subnet_configs(A=5)
    #
    #             for subnet_config in subnet_configs:
    #                 mapp[str(subnet_config)] += 1
    #
    # for i in range(50):
    #     for j, arch_list in enumerate(client_arch_lists.values()):
    #         arch_config = resnet.random_sample_subnet_config(arch_list[-1])
    #         subnet = resnet.get_subnet(arch_config)
    #         for k in range(sample_sum[j] // batch_size):
    #             mapp2[str(arch_config)] += 1
    #             subnet_configs = []
    #             for _ in range(10):
    #                 subnet_config = subnet.random_sample_subnet_config()
    #                 if subnet_config not in subnet_configs:
    #                     subnet_configs.append(subnet_config)
    #             subnet_configs = subnet_configs[:3]
    #             for subnet_config in subnet_configs:
    #                 mapp2[str(subnet_config)] += 1
    #
    #
    # for i in range(50):
    #     for j, arch_list in enumerate(client_arch_lists.values()):
    #         arch_config = random.choice(arch_list)
    #         subnet = resnet.get_subnet(arch_config)
    #         for k in range(sample_sum[j]//batch_size):
    #             mapp3[str(arch_config)] += 1
    #             subnet_configs = []
    #             for _ in range(10):
    #                 subnet_config = subnet.random_sample_subnet_config()
    #                 if subnet_config not in subnet_configs:
    #                     subnet_configs.append(subnet_config)
    #             subnet_configs = subnet_configs[:3]
    #             for subnet_config in subnet_configs:
    #                 mapp3[str(subnet_config)] += 1
    #
    # for i in range(50):
    #     for j, arch_list in enumerate(client_arch_lists.values()):
    #         arch_config = random.choice(arch_list)
    #         subnet = resnet.get_subnet(arch_config)
    #         for k in range(sample_sum[j]//batch_size):
    #             mapp4[str(arch_config)] += 1
    #             subnet_configs = []
    #             last_subnet_config = None
    #             for _ in range(3):
    #                 subnetwork_config = subnet.random_sample_subnet_config(last_subnet_config)
    #                 if subnetwork_config not in subnet_configs:
    #                     subnet_configs.append(subnetwork_config)
    #                     last_subnet_config = subnetwork_config
    #             for subnet_config in subnet_configs:
    #                 mapp4[str(subnet_config)] += 1
    #
    # for i in range(50):
    #     for j, arch_list in enumerate(client_arch_lists.values()):
    #         arch_config = random.choice(arch_list)
    #         subnet = resnet.get_subnet(arch_config)
    #         for k in range(sample_sum[j]//batch_size):
    #             mapp[str(arch_config)] += 1
    #             subnet_configs = subnet.get_progressive_subnet_configs(A=3)
    #             for idx in subnet_configs:
    #                 mapp[str(idx)] += 1

    # print(sum(mapp3.values()))
    # print(sum(mapp4.values()))
    # print(sum(mapp.values()))
    mapps = torch.load("mapps")
    plot_mapp_histograms([mapps[0], mapps[2], mapps[1], mapp], [
        'Random Sampling + Random Sampling',
        "Maximum Sampling + Recursive Sampling",
        'Maximum Sampling + Random Sampling',
        "Maximum Sampling + Dynamic Weighted Sampling"
    ], ['#2D8C5E', '#B8B85E', '#4A8C98', '#D1507D'])
    # plot_mapp_histograms([mapp2], ['Random Sampling + Random Sampling'], ['g'])
    # plot_mapp_histograms([mapps[1]], ['Maximum Sampling + Random Sampling'], 'y')
    # plot_mapp_histograms([mapps[2]], ["Maximum Sampling + Recursive Sampling"], 'b')
    # plot_mapp_histograms([mapps[3]], ["Maximum Sampling + Dynamic Weighted Sampling"], 'r')
    # torch.save([mapp2, mapps[1], mapps[2], mapps[3]], "mapps")
    # plot_mapp_histograms(torch.load("mapps"))
    # import os

    # data_shares = [.1, .1, .1, .1, .1, .5]
    # trainData, valData, testData = load_cifar("cifar10", os.path.join("../../../data", "cifar10"),
    #                                           data_shares, 1, 1)
    #
    # for idx, trainset in enumerate(trainData):
    #     class_distribution = np.zeros(10)
    #     for _, labels in trainset:
    #         class_distribution[labels] += 1
    #     class_distribution = class_distribution / np.sum(class_distribution)
    #     print(f'Client {idx} class distribution:', class_distribution)
    # idx_dataset = -1
    # train_set, val_set, test_set = trainData[idx_dataset], valData[idx_dataset], testData[idx_dataset]
    # print(resnet)
    #
    # summary(resnet, input_size=(128, 3, 32, 32))
    # # 损失函数
    # optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # criterion = nn.CrossEntropyLoss()
    # # dataloader
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)


    def test(model, active_path=None):
        # 测试集测试准确率，召回率，F1，以及top-1,top-2,top-3
        correct = 0
        total = 0
        top_1 = 0
        top_2 = 0
        top_3 = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images, active_path)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, top_2_predicted = torch.topk(outputs, 2)
                _, top_3_predicted = torch.topk(outputs, 3)
                for i in range(labels.size(0)):
                    if labels[i] in top_2_predicted[i]:
                        top_2 += 1
                    if labels[i] in top_3_predicted[i]:
                        top_3 += 1
                    if labels[i] == predicted[i]:
                        top_1 += 1
        model_size, accuracy = count_parameters(model, 'MB'), 100 * correct / total
        print(active_path, f"Parameters:{model_size}MB")
        print(f"Accuracy: {accuracy}%")
        print(f"Top-1 Accuracy: {100 * top_1 / total}%")
        print(f"Top-2 Accuracy: {100 * top_2 / total}%")
        print(f"Top-3 Accuracy: {100 * top_3 / total}%")
        return model_size, accuracy


    def train(model, active_config=None, sub_config=None, epochs=10):
        # 迭代训练resnet到50个epoch，每10个epoch测试一次
        for epoch in range(epochs):
            model.train()
            # model.re_organize_weights()
            for i, (inputs, labels) in enumerate(train_loader):
                subnet_configs = model.get_progressive_subnet_configs()
                # subnet_configs = [get_model_configs("4_2_0.25"), get_model_configs("4_2_0.5"), get_model_configs("4_2_0.25")]
                # model.re_organize_weights()
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                n = 1
                for sub_config in subnet_configs:
                    sub_outputs = model(inputs, sub_config)
                    sub_loss = criterion(sub_outputs, labels)
                    loss += sub_loss
                    n += 1
                # loss /= n
                loss.backward()
                optimizer.step()
                if i % 500 == 0:
                    print(f"Epoch {epoch} Iter {i} Loss {loss.item()}")
            if epoch % 5 == 0:
                test(model)
                # visualize_model_parameters(model.hidden_layer.state_dict(), figsize=(10, 8))


    # subnet_config = resnet.random_sample_subnet_config()
    # width = 0.8
    # subnet1 = resnet.get_subnet(
    #     {
    #         "sublayer_0": {
    #             "block_0": width,
    #             "block_1": width
    #         },
    #         "sublayer_1": {
    #             "block_0": width,
    #             "block_1": width
    #         },
    #         "sublayer_2": {
    #             "block_0": width,
    #             "block_1": width
    #         },
    #         # "sublayer_3": {
    #         #     "block_0": width,
    #         #     "block_1": width
    #         # }
    #     }
    # )
    # print(count_parameters(subnet1))
    # 5.31 7,0.86
    #
    # train(resnet, epochs=10)
    # test(resnet)
    # for i in range(10):
    #     subnet_config = resnet.random_sample_subnet_config()
    #     test(resnet, subnet_config)
    #     subnet1 = resnet.get_subnet(subnet_config)
    #     test(subnet1)



