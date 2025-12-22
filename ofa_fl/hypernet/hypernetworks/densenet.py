import copy
import os
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import math
import time

import torch
from torch import nn
import torch.nn.functional as F
from ..base.dynamic_modules import DynamicConv2d, MyIdentity, DynamicLinear, DynamicBatchNorm2d, \
    DynamicGroupNorm, Scaler
from ..datasets.load_cifar import get_datasets, load_cifar
from ..utils.common_utils import get_net_device, make_divisible, adjust_bn_according_to_idx
from itertools import product

class DynamicBottleneck(nn.Module):
    CHANNEL_DIVISIBLE = 8

    def __init__(self,
                 in_planes,
                 out_planes=None,
                 growth_rate=None,
                 norm_way="bn",
                 scale_ratio=1.0,
                 use_scaler=False,
                 track_running_stats=True):
        super(DynamicBottleneck, self).__init__()
        self.norm_way = norm_way
        self.use_scaler = use_scaler
        self.scaler = Scaler(scale_ratio) if use_scaler else MyIdentity()
        self.track_running_stats = track_running_stats

        self.default_in_channels = in_planes
        self.default_out_channels = 4 * growth_rate
        self.growth_rate = growth_rate
        self.active_in_channels = self.default_in_channels
        self.active_out_channels = 4 * growth_rate if out_planes is None else out_planes

        if norm_way == "bn":
            self.bn1 = DynamicBatchNorm2d(self.active_in_channels, track_running_stats=track_running_stats)
            self.bn2 = DynamicBatchNorm2d(self.active_out_channels, track_running_stats=track_running_stats)
        elif norm_way == "in":
            self.bn1 = DynamicGroupNorm(self.active_in_channels, self.active_in_channels)
            self.bn2 = DynamicGroupNorm(self.active_out_channels, self.active_out_channels)
        elif norm_way == "ln":
            self.bn1 = DynamicGroupNorm(1, self.active_in_channels)
            self.bn2 = DynamicGroupNorm(1, self.active_out_channels)
        else:
            raise NotImplementedError("norm_way must be in, bn, ln")

        self.conv1 = DynamicConv2d(self.active_in_channels, self.active_out_channels, kernel_size=1)
        self.conv2 = DynamicConv2d(self.active_out_channels, self.growth_rate, kernel_size=3, padding=1)

    def forward(self, x, planes=None):
        if planes is None:
            planes = self.active_out_channels

        out = F.relu(self.bn1(self.scaler(x)))
        out = self.conv1(out, planes)
        out = F.relu(self.bn2(self.scaler(out)))
        out = self.conv2(out)

        # Concatenate along the channel dimension
        out = torch.cat([x, out], 1)
        return out

    def get_subnet(self, in_channels, out_channels):
        device = get_net_device(self)
        subnet = DynamicBottleneck(
            in_channels,
            out_channels,
            self.growth_rate,
            norm_way=self.norm_way,
            scale_ratio=out_channels / self.default_out_channels,
            use_scaler=self.use_scaler,
            track_running_stats=self.track_running_stats
        ).to(device)

        subnet.default_in_channels = self.default_in_channels
        subnet.default_out_channels = self.default_out_channels
        subnet.growth_rate = self.growth_rate
        subnet.scaler = self.scaler
        subnet.conv1.copy_conv(self.conv1)
        subnet.bn1._copy(self.bn1)
        subnet.conv2.copy_conv(self.conv2)
        subnet.bn2._copy(self.bn2)

        return subnet

    def re_organize_weights(self, expand_ratio_list):
        importance = torch.norm(self.conv2.conv.weight, p=1, dim=(0, 2, 3))

        sorted_expand_list = copy.deepcopy(expand_ratio_list)
        sorted_expand_list.sort(reverse=True)
        target_width_list = [
            make_divisible(
                round(self.growth_rate * expand),
                self.CHANNEL_DIVISIBLE,
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
    def __init__(self, in_planes, out_planes, stride=1, track_running_stats=True):
        super(DynamicTransition, self).__init__()
        self.stride = stride
        self.conv = DynamicConv2d(in_planes, out_planes, kernel_size=1, stride=stride)
        self.bn = DynamicBatchNorm2d(in_planes, track_running_stats=track_running_stats)
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

        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

    def get_subnet(self, in_planes, out_planes, expand_ratio_list=None):
        device = next(self.parameters()).device
        subnet = DynamicTransition(
            in_planes,
            out_planes,
            stride=self.stride,
            # scale_ratio=out_planes / self.default_out_channels,
            # scale_ratio=ratio,
            track_running_stats=self.track_running_stats
        ).to(device)

        subnet.default_in_channels = self.default_in_channels
        subnet.default_out_channels = self.default_out_channels
        subnet.scaler = self.scaler
        subnet.bn._copy(self.bn)
        subnet.conv.copy_conv(self.conv)
        return subnet


class SuperDenseNet(nn.Module):
    def __init__(self, block, block_config, growth_rate,
                 width_pruning_ratio_list=[1, 0.5],
                 reduction=0.5,
                 num_classes=10,
                 norm_way="bn", track_running_stats=True, *args, **kwargs):
        super(SuperDenseNet, self).__init__(*args, **kwargs)
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = 2*growth_rate
        self.num_classes = num_classes
        self.norm_way = norm_way
        self.track_running_stats = track_running_stats
        self.NUM_SUBLAYER = len(block_config)
        self.BASE_DEPTH_LIST = block_config
        self.STAGE_WIDTH_LIST = [2*growth_rate for i, num_layers in enumerate(block_config)]

        # embedding_layer
        self.embedding_layer = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, self.num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        # hidden_layer
        num_features = self.num_init_features
        hidden_layer = []
        transition_layer = []
        for i, num_layers in enumerate(block_config):

            sublayer = self._make_dense_layer(block, num_features, num_layers)
            hidden_layer.append((f'sublayer_{i}', sublayer))

            num_features += num_layers * growth_rate
            out_features = int(math.floor(num_features*reduction))
            if i != len(block_config) - 1:
                trans = DynamicTransition(num_features, out_features,
                                          track_running_stats=track_running_stats)
                hidden_layer.append((f'transition_{i}', trans))
                num_features = out_features

        self.hidden_layer = nn.Sequential(OrderedDict(hidden_layer))

        # output_layer
        self.output_layer = nn.Sequential(
            DynamicBatchNorm2d(num_features, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            DynamicLinear(num_features, num_classes),
        )

        self.width_pruning_ratio_list = width_pruning_ratio_list
        self.default_output_path = self.get_default_config()
        self.active_output_path = self.default_output_path

        vec = torch.tensor(self.STAGE_WIDTH_LIST) / torch.sum(
            torch.tensor(self.BASE_DEPTH_LIST) * torch.tensor(self.STAGE_WIDTH_LIST))
        self.layer_cost = torch.repeat_interleave(vec, torch.tensor(self.BASE_DEPTH_LIST))

    def _make_dense_layer(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append((f"block_{i}", block(in_planes, growth_rate=self.growth_rate)))
            in_planes += self.growth_rate
        return nn.Sequential(OrderedDict(layers))

    def get_default_config(self):
        default_config = {f"sublayer_{i}": {} for i in range(self.NUM_SUBLAYER)}
        for sublayer_idx in range(self.NUM_SUBLAYER):
            default_config[f'sublayer_{sublayer_idx}'] = {}
            for block_idx in range(self.BASE_DEPTH_LIST[sublayer_idx]):
                default_config[f'sublayer_{sublayer_idx}'][f'block_{block_idx}'] = max(self.width_pruning_ratio_list)
        return default_config

    # def get_progressive_subnet_configs(self, reverse=False):
    #     subnetwork_configs = []
    #     num_sublayers = len(self.active_output_path)
    #     sublayer_depth_list = [len(v) for v in self.active_output_path.values()]
    #     max_depth = max(sublayer_depth_list)
    #     sublayer_ratio_list = [v["block_0"] for v in self.active_output_path.values()]
    #     max_ratio = max(sublayer_ratio_list)
    #
    #     DepthProgressive = 0
    #     WidthProgressive = 1
    #     RandProgressive = 3
    #
    #     choices = [DepthProgressive, WidthProgressive, RandProgressive]
    #     chosen = random.choices(choices, k=1)[0]
    #     if chosen == DepthProgressive:
    #         num_sublayer_list = sorted(range(num_sublayers, 0, -1), reverse=reverse)
    #         for num_sub in num_sublayer_list:
    #             subnetwork_config = {}
    #             for i, (k, y) in enumerate(self.active_output_path.items()):
    #                 if i != num_sub - 1:
    #                     subnetwork_config[k] = self.active_output_path[k]
    #                 else:
    #                     num_depth_list = sorted(range(len(self.active_output_path[k]), 0, -1), reverse=reverse)
    #                     for depth in num_depth_list:
    #                         subnetwork_config[k] = {}
    #                         for j, (kk, yy) in enumerate(self.active_output_path[k].items()):
    #                             subnetwork_config[k][kk] = yy
    #                             if j == depth - 1:
    #                                 break
    #                         if subnetwork_config != self.active_output_path:
    #                             subnetwork_configs.append(subnetwork_config.copy())
    #                 if i == num_sub - 1:
    #                     break
    #
    #     elif chosen == WidthProgressive:
    #         # Step 1: 为每个子层生成独立的可用宽度比例列表
    #         sublayer_ratio_ranges = {}
    #         for sublayer_name in self.active_output_path:
    #             # 获取该子层在active_output_path中的最大比例
    #             max_ratio_in_sublayer = max(self.active_output_path[sublayer_name].values())
    #
    #             # 截取允许的比例列表：从最小到该子层的最大比例
    #             valid_ratios = [
    #                 ratio for ratio in self.width_pruning_ratio_list
    #                 if ratio <= max_ratio_in_sublayer
    #             ]
    #             sublayer_ratio_ranges[sublayer_name] = sorted(valid_ratios, reverse=False)  # 升序排列
    #
    #         # Step 2: 生成所有可能的组合（笛卡尔积）
    #         all_ratio_combinations = product(
    #             *[sublayer_ratio_ranges[f"sublayer_{i}"] for i in range(num_sublayers)]
    #         )
    #
    #         # Step 3: 构建子网络配置
    #         for ratio_combination in all_ratio_combinations:
    #             subnetwork_config = {}
    #             for sublayer_idx, ratio in enumerate(ratio_combination):
    #                 sublayer_name = f"sublayer_{sublayer_idx}"
    #                 subnetwork_config[sublayer_name] = {}
    #                 # 保持每层块数不变，仅修改宽度比例
    #                 for block_name in self.active_output_path[sublayer_name]:
    #                     subnetwork_config[sublayer_name][block_name] = ratio
    #             subnetwork_configs.append(subnetwork_config)
    #
    #     else:
    #         for _ in range(5):
    #             subnetwork_config = self.random_sample_subnet_config()
    #             if subnetwork_config not in subnetwork_configs:
    #                 subnetwork_configs.append(subnetwork_config)
    #     random.shuffle(subnetwork_configs)
    #     subnetwork_configs = subnetwork_configs[:3]
    #
    #     return subnetwork_configs
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
                subnetwork_config = self.random_sample_subnet_config(last_subnet_config, stage)
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

    def random_sample_subnet_config(self, max_net_config=None, budget=None):
        if max_net_config is None:
            max_net_config = self.active_output_path

        NUM_SUBLAYER = len(max_net_config)
        BASE_DEPTH_LIST = []
        for sublayer_idx in range(NUM_SUBLAYER):
            sublayer_name = f"sublayer_{sublayer_idx}"
            BASE_DEPTH_LIST.append(len(max_net_config[sublayer_name]))
        subnetwork_config = {}

        num_sublayers = random.randint(1, NUM_SUBLAYER)

        for sublayer in range(num_sublayers):
            subnetwork_config[f'sublayer_{sublayer}'] = {}

            num_blocks = random.randint(1, BASE_DEPTH_LIST[sublayer])
            active_compression_rate = self.width_pruning_ratio_list[self.width_pruning_ratio_list.index(
                max_net_config[f'sublayer_{sublayer}'][f'block_{0}']):]
            compression_rate = random.choice(active_compression_rate)
            for block in range(num_blocks):
                subnetwork_config[f'sublayer_{sublayer}'][f'block_{block}'] = compression_rate
        return subnetwork_config

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

    def generate_all_subnet_configs_old(self):
        all_configs = []
        for num_sublayer in range(self.NUM_SUBLAYER, 0, -1):
            self.recursion_all_subnet_configs(num_sublayer, all_configs)
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
                current_out_channel = round(width_pruning_ratio * block.default_out_channels)
                x = block(x, current_out_channel)
            transition_name = f"transition_{sublayer_name[-1]}"
            if transition_name in self.hidden_layer._modules:
                transition = getattr(self.hidden_layer, transition_name)
                x = transition(x)

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
            sublayer = getattr(self.hidden_layer, sublayer_name)
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

        del subnet.hidden_layer
        last_width_pruning_ratio = 1.0
        sub_hidden_layer = []

        last_output_channel = self.num_init_features
        for sublayer_idx in range(self.NUM_SUBLAYER):
            sublayer_name = f'sublayer_{sublayer_idx}'
            sublayer = getattr(self.hidden_layer, sublayer_name)
            subnet_sublayer = []

            if sublayer_name not in subnet_config:
                sub_hidden_layer.append((sublayer_name, MyIdentity()))
                subnet_sublayer_config = []
            else:
                subnet_sublayer_config = subnet_config[sublayer_name]


            for block_idx in range(self.BASE_DEPTH_LIST[sublayer_idx]):
                block_name = f'block_{block_idx}'

                if block_name not in subnet_sublayer_config:
                    subnet_sublayer.append((block_name, MyIdentity()))
                    continue

                block = sublayer[block_idx]

                width_pruning_ratio = subnet_config[sublayer_name][block_name]
                current_out_channel = round(width_pruning_ratio * block.default_out_channels)
                subnet_sublayer.append((block_name, block.get_subnet(last_output_channel, current_out_channel)))
                last_output_channel = block.default_in_channels+block.growth_rate
            sub_hidden_layer.append((sublayer_name, nn.Sequential(OrderedDict(subnet_sublayer))))

            transition_name = f"transition_{sublayer_idx}"
            if transition_name in self.hidden_layer._modules:
                if sublayer_idx < len(subnet_config):
                    transition = getattr(self.hidden_layer, transition_name)
                    sub_hidden_layer.append((transition_name, copy.deepcopy(transition)))
                    last_output_channel = transition.default_out_channels
                else:
                    sub_hidden_layer.append((transition_name, MyIdentity()))


        subnet.hidden_layer = nn.Sequential(OrderedDict(sub_hidden_layer))
        return subnet

def super_densenet121(growth_rate=12,
                      num_classes=10,
                      width_pruning_ratio_list=(1, 0.5),
                      norm_way="bn",
                      track_running_stats=True):
    return SuperDenseNet(
        block=DynamicBottleneck,
        growth_rate=growth_rate,
        width_pruning_ratio_list=width_pruning_ratio_list,
        block_config=(6, 12, 24, 16),
        num_classes=num_classes,
        norm_way=norm_way,
        track_running_stats=track_running_stats
    )



import matplotlib.pyplot as plt
import pandas as pd
from hypernet.utils.evaluation import visualize_model_parameters, count_parameters

def eval_all_subnets(model: SuperDenseNet, data_path, csv_path):

    trainData, valData, testData = get_datasets("cifar100", os.path.join(data_path, "cifar100"))
    train_set, val_set, test_set = trainData, valData, testData

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=5)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=5)
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
    total = len(all_subnet_configs)
    step = total // 5000 if total >= 5000 else 1  # 防止总数小于5000时出现除零错误
    all_subnet_configs = all_subnet_configs[::step]

    def evaluate_subnet(subnet_config, model):

        print(subnet_config)
        subnet = model.get_subnet(subnet_config)
        fine_tuning(subnet)
        model_size, accuracy = test(subnet)
        return model_size, accuracy, subnet_config

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

if __name__ == '__main__':
    from torchinfo import summary

    densenet = super_densenet121(num_classes=100).cuda()
    print(len(densenet.generate_all_subnet_configs()))
    summary(densenet, input_size=(128, 3, 32, 32))
    data_shares = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.5]

    trainData, valData, testData = load_cifar("cifar100", os.path.join("../../../data", "cifar100"),
                                              data_shares, 0.5, 1)
    idx_dataset = -1
    train_set, val_set, test_set = trainData[idx_dataset], valData[idx_dataset], testData[idx_dataset]

    _, _, testData = get_datasets("cifar100", os.path.join("../../../data", "cifar100"))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)


    subnet_config = densenet.default_output_path
    subnet = densenet.get_subnet(subnet_config)

    optimizer = torch.optim.SGD(densenet.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

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


    # train(densenet)
    # test(densenet)
    # para = densenet.state_dict()
    # densenet.load_state_dict(torch.load("../../logs/cifar100/seed4321/hypernet.pth").state_dict())
    # test(densenet)


