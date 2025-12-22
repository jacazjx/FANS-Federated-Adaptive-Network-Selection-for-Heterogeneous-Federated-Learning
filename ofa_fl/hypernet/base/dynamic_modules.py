import copy
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hypernet.utils.common_utils import get_same_padding, sub_filter_start_end, val2list, ks2list


class DynamicLayerNorm(nn.LayerNorm):
    def __init__(self, max_features, eps=1e-12):
        super(DynamicLayerNorm, self).__init__(max_features, eps)
        self.max_features = max_features
        self.eps = eps

        self.active_features = max_features

    def get_active_params(self, num_features):
        weight = self.weight[:num_features]
        bias = self.bias[:num_features]
        return weight, bias

    def forward(self, x, num_features=None):
        if num_features is None:
            num_features = self.active_features
        normalized_shape = x.shape[-1]
        weight, bias = self.get_active_params(num_features)

        out = F.layer_norm(x, (normalized_shape,), weight, bias, self.eps)
        return out

    def copy_ln(self, ln):
        feature_dim = self.max_features
        active_param = ln.get_active_params(feature_dim)
        self.weight.data.copy_(active_param[0])
        self.bias.data.copy_(active_param[1])
        self.eps = ln.eps
        self.to(ln.weight.device)


class DynamicLinear(nn.Linear):
    def __init__(self, max_in_features, max_out_features, bias=True, multi_head=1):
        super(DynamicLinear, self).__init__(max_in_features, max_out_features, bias)
        self.default_in_features = max_in_features
        self.active_in_features = max_in_features
        self.default_out_features = max_out_features
        self.active_out_features = max_out_features
        self.multi_head = multi_head

    def get_active_weight(self, out_features, in_features):

        weight_chunks = torch.chunk(self.weight, int(self.multi_head))
        out_features_per_head = out_features // self.multi_head

        weight = [weight_chunks[i][:out_features_per_head, :in_features] for i in range(int(self.multi_head))]
        return torch.cat(weight, dim=0)

    def get_active_bias(self, out_features):
        if self.bias is not None:
            bias_chunks = torch.chunk(self.bias, int(self.multi_head))
            out_features_per_head = out_features // self.multi_head
            bias = [bias_chunks[i][:out_features_per_head] for i in range(int(self.multi_head))]
            return torch.cat(bias, dim=0)
        else:
            return None

    # copy from larger linear
    def copy_linear(self, linear):
        out_features, in_features = self.active_out_features, self.active_in_features
        weight = linear.get_active_weight(out_features, in_features)
        bias = linear.get_active_bias(out_features)
        self.default_out_features, self.default_in_features = linear.default_out_features, linear.default_in_features
        self.weight.data.copy_(weight)
        if self.bias is not None:
            self.bias.data.copy_(bias)
        self.to(linear.weight.device)

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(-1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y


class MyIdentity(nn.Module):
    def __init__(self):
        super(MyIdentity, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class MyGlobalAvgPool2d(nn.Module):
    def __init__(self, keep_dim=True):
        super(MyGlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return "MyGlobalAvgPool2d(keep_dim=%s)" % self.keep_dim


class DynamicConv2d(nn.Module):
    """
        Conv2d with Weight Standardization
        https://github.com/joe-siyuan-qiao/WeightStandardization
    """
    WS_EPS = 1e-5

    def __init__(
            self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0
    ):
        super(DynamicConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            bias=False,
        )

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = (
                weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .mean(dim=3, keepdim=True)
            )
            weight = weight - weight_mean
            std = (
                    weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
                    + self.WS_EPS
            )
            weight = weight / std.expand_as(weight)
        return weight

    @property
    def weight(self):
        return self.conv.weight

    def copy_conv(self, conv):
        out_channel, in_channel = self.out_channels, self.in_channels
        filter = conv.get_active_filter(out_channel, in_channel)
        self.conv.weight.data.copy_(filter)
        self.conv.to(conv.conv.weight.device)

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.out_channels
        in_channel = x.size(1)

        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        # padding = get_same_padding(self.kernel_size)
        # if out_channel != self.out_channels:
        # filters = self.weight_standardization(filters)

        y = F.conv2d(x, filters, None, self.stride, self.padding, self.dilation, 1)
        return y


class DynamicGroupConv2d(nn.Module):
    WS_EPS = 1e-5

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size_list,
            groups_list,
            stride=1,
            dilation=1,
    ):
        super(DynamicGroupConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_list = kernel_size_list
        self.groups_list = groups_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=min(self.groups_list),
            bias=False,
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_groups = min(self.groups_list)

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = (
                weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .mean(dim=3, keepdim=True)
            )
            weight = weight - weight_mean
            std = (
                    weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
                    + self.WS_EPS
            )
            weight = weight / std.expand_as(weight)
        return weight

    def get_active_filter(self, kernel_size, groups):
        start, end = sub_filter_start_end(max(self.kernel_size_list), kernel_size)
        filters = self.conv.weight[:, :, start:end, start:end]

        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_in_channels = self.in_channels // groups
        sub_ratio = filters.size(1) // sub_in_channels

        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_in_channels
            filter_crops.append(sub_filter[:, start: start + sub_in_channels, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def forward(self, x, kernel_size=None, groups=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if groups is None:
            groups = self.active_groups

        filters = self.get_active_filter(kernel_size, groups).contiguous()
        padding = get_same_padding(kernel_size)
        filters = self.weight_standardization(filters)
        y = F.conv2d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            groups,
        )
        return y


class Scaler(nn.Module):
    def __init__(self, rate=1.0):
        super().__init__()
        self.rate = rate

    def forward(self, input, ratio=None):
        if ratio is None:
            ratio = self.rate
        if isinstance(ratio, float):
            output = input / ratio #if self.training else input
        else:
            plane = input.size(1)
            ratio = ratio[:plane].to(input.device)
            output = input / ratio.view(1, plane, 1, 1)
        return output


class DynamicBatchNorm2d(nn.BatchNorm2d):
    SET_RUNNING_STATISTICS = False

    def __init__(self, num_features,
                 eps: float = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device: Any = None,
                 dtype: Any = None
                 ):
        super(DynamicBatchNorm2d, self).__init__(num_features,
                                                 eps=eps,
                                                 momentum=momentum if track_running_stats else None,
                                                 affine=affine,
                                                 track_running_stats=track_running_stats,
                                                 device=device,
                                                 dtype=dtype)
        # self.device = device
        # self.dtype = dtype

    def get_active_param(self, num_features):
        return {
            "weight": self.weight[:num_features],
            "bias": self.bias[:num_features],
            "running_mean": self.running_mean[:num_features] if self.track_running_stats else None,
            "running_var": self.running_var[:num_features] if self.track_running_stats else None,
            "eps": self.eps,
            "momentum": self.momentum,
            "track_running_stats": self.track_running_stats,
            "device": self.weight.device,
            "dtype": self.weight.dtype
        }

    def _copy(self, bn):
        feature_dim = self.num_features

        active_param = bn.get_active_param(feature_dim)
        self.track_running_stats = active_param["track_running_stats"]
        if self.track_running_stats:
            self.running_mean.data.copy_(active_param["running_mean"])
            self.running_var.data.copy_(active_param["running_var"])
        self.weight.data.copy_(active_param["weight"])
        self.bias.data.copy_(active_param["bias"])
        self.eps = active_param["eps"]
        self.momentum = active_param["momentum"]
        device = active_param["device"]
        self.to(device)

    def forward(self, x):
        feature_dim = x.size(1)
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None
        running_mean = running_mean[:feature_dim] if running_mean is not None else None
        running_var = running_var[:feature_dim] if running_var is not None else None

        return F.batch_norm(
            x,
            running_mean,
            running_var,
            self.weight[:feature_dim],
            self.bias[:feature_dim],
            bn_training,
            exponential_average_factor,
            self.eps
        )

class DynamicGroupNorm(nn.GroupNorm):
    def __init__(
            self, num_groups, num_channels, eps=1e-5, affine=True, channel_per_group=1
    ):
        super(DynamicGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.channel_per_group = channel_per_group

    def forward(self, x):
        n_channels = x.size(1)
        n_groups = n_channels // self.channel_per_group
        return F.group_norm(
            x, n_groups, self.weight[:n_channels], self.bias[:n_channels], self.eps
        )

    def _copy(self, gn: nn.GroupNorm):
        num_features = self.num_channels
        self.channel_per_group = min(num_features, gn.channel_per_group)
        self.weight.data.copy_(gn.weight[:num_features])
        self.bias.data.copy_(gn.bias[:num_features])
        self.eps = gn.eps
        self.affine = gn.affine
        self.to(gn.weight.device)

    @property
    def bn(self):
        return self


class DynamicSeparableConv2d(nn.Module):

    def __init__(self, max_in_channel, max_out_channel, max_kernel_size, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channel = max_in_channel
        self.max_out_channel = max_out_channel
        self.kernel_size_list = ks2list(max_kernel_size)
        self.stride = stride
        self.dilation = dilation

        self.depth_wise = DynamicDepthWise(
            self.max_in_channel, self.kernel_size_list, self.stride, self.dilation, self.max_in_channel
        )
        self.point_wise = DynamicConv2d(
            self.max_in_channel, self.max_out_channel, 1, 1, 1
        )

    def forward(self, x, kernel_size=None, out_channel=None):
        x = self.depth_wise(x, kernel_size)
        x = self.point_wise(x, out_channel)
        return x

    def get_width_resolution_module(self, in_channel, kernel_size=None, out_channel=None):
        sub_module = DynamicSeparableConv2d(
            in_channel,
            out_channel,
            kernel_size,
            self.stride,
            self.dilation
        )

        sub_kernel = self.depth_wise.get_active_filter(
            in_channel,
            kernel_size=kernel_size
        )
        sub_pointwise = self.point_wise.get_active_filter(
            out_channel,
            in_channel
        )
        sub_module.depth_wise.conv.weight.data.copy_(sub_kernel)
        sub_module.point_wise.conv.weight.data.copy_(sub_pointwise)

        scaling_matrix = {}
        sub_ks = list(set(sub_module.depth_wise.kernel_size_list))
        for i in range(len(sub_ks) - 1, 0, -1):
            src_ks = sub_ks[i]
            target_ks = sub_ks[i - 1]
            param_name = "%dto%d_matrix" % (src_ks, target_ks)
            scaling_matrix[param_name] = self.depth_wise.get_parameter(param_name)

        for name, para in scaling_matrix.items():
            sub_module.depth_wise.register_parameter(name, para)

        return sub_module


class DynamicDepthWise(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1, groups=1):
        super(DynamicDepthWise, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list  # e.g., [7, 5, 3, 1]
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=groups,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)  # 5to3
                # noinspection PyArgumentList
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small ** 2)  # 5to3: [9, 9] 、 7to5: [25, 25]
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)
        # 从中间开始，取出kernel size大小的filter
        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        # 设置了线性变换的，对所取的filter做线性变换
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                           :out_channel, :in_channel, :, :
                           ]  # start with max kernel
            # 倒过来，从大的kernel size开始，逐渐缩小kernel size，并做线性变换
            for i in range(len(self._ks_set) - 1, 0, -1):  # [3, 5, 7]
                src_ks = self._ks_set[i]  # 7
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]  # 5
                start, end = sub_filter_start_end(src_ks, target_ks)
                # 先抠出来，此时维度为 [in_channel, in_channel, target_ks, target_ks]
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )  # [in_channel, in_channel, target_ks * target_ks]
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                # [in_channel * in_channel, target_ks * target_ks]
                _input_filter = F.linear(
                    _input_filter,
                    self.get_parameter("%dto%d_matrix" % (src_ks, target_ks)),
                )
                # [in_channel * in_channel, target_ks * target_ks] * [target_ks**2, target_ks**2]
                # -> [in_channel * in_channel, target_ks * target_ks] 只是经过线性变换一下 (变化矩阵可学习)
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks ** 2
                )  # [in_channel, in_channel, target_ks * target_ks]
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )  # [in_channel, in_channel, target_ks, target_ks]
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, in_channel)
        return y


if __name__ == "__main__":
    from torchinfo import summary

    batch_size, in_channels, height, width = 100, 3, 224, 224
    test_input = torch.randn(batch_size, in_channels, height, width)
    kernel = 3
    stride = 2
    padding = 1
    dy_conv = DynamicConv2d(in_channels, 100, kernel, stride)
    conv = nn.Conv2d(in_channels, 100, kernel, stride, padding=padding)
    dy_se_conv = DynamicSeparableConv2d(in_channels, 100, kernel, stride)

    print(f"dy_conv:")
    summary(dy_conv, (batch_size, in_channels, height, width), device='cpu')

    print(f"conv:")
    summary(conv, (batch_size, in_channels, height, width), device='cpu')

    print(f"dy_se_conv:")
    summary(dy_se_conv, (batch_size, in_channels, height, width), device='cpu')

    print(dy_conv(test_input, 50).shape)
    print(conv(test_input).shape)
    print("output diversity is", (int((height + 2 * padding - kernel) / stride) + 1),
          (int((width + 2 * padding - kernel) / stride) + 1))

    dy_se_conv_3_100_7 = DynamicSeparableConv2d(in_channels, 100, 7, stride)
    dy_se_conv_3_50_5 = dy_se_conv_3_100_7.get_width_resolution_module(in_channels, 5, 50)
    dy_se_conv_3_10_3 = dy_se_conv_3_50_5.get_width_resolution_module(in_channels, 3, 10)

    summary(dy_se_conv_3_100_7, (batch_size, in_channels, height, width), device='cpu')
    summary(dy_se_conv_3_50_5, (batch_size, in_channels, height, width), device='cpu')
    summary(dy_se_conv_3_10_3, (batch_size, in_channels, height, width), device='cpu')
