import numpy as np
from sklearn.metrics import accuracy_score


def calculate_SLC_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true_bool = y_true.argmax(-1)
    else:
        y_true_bool = y_true

    if len(y_pred.shape) > 1:
        y_pred_bool = y_pred.argmax(-1)
    else:
        y_pred_bool = y_pred


    res = {'ACC': accuracy_score(y_true_bool, y_pred_bool)}

    return res

def display_results(results, metrics=['ACC'], logger=None):
    if logger is not None:
        logger.critical('{0:>10}'.format("Label") + ' '.join(['%10s'] * len(metrics)) % tuple([m for m in metrics]))
        logger.critical('{0:>10}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([results[m] for m in metrics]))
    else:
        print('{0:>20}'.format("Label") + ' '.join(['%10s'] * len(metrics)) % tuple([m for m in metrics]))
        print('{0:>20}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([results[m] for m in metrics]))

    return [results[m] for m in metrics]


def count_parameters(net, unit="MB"):
    total_params = sum(p.numel() for p in net.parameters())
    if unit == "KB":
        return total_params / 1024
    elif unit == "MB":
        return total_params / 1024 / 1024
    elif unit == "GB":
        return total_params / 1024 / 1024 / 1024
    else:
        raise ValueError("do not support: %s" % unit)

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def visualize_model_parameters(state_dict, figsize=(10, 8), max_plots_per_row=1):
    """
    可视化模型参数。

    参数:
    - model: PyTorch 模型
    - figsize: 图像大小，默认为 (10, 8)
    - max_plots_per_row: 每行的最大子图数量，默认为 3
    """

    # 计算需要的子图数量
    num_params = len(state_dict)
    num_rows = (num_params + max_plots_per_row - 1) // max_plots_per_row

    # 创建一个新的图形
    fig, axes = plt.subplots(nrows=num_rows, ncols=max_plots_per_row, figsize=figsize)

    # 如果只有一个参数，调整 axes 为单个轴
    if num_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    extra_axes = []

    # 遍历每个参数
    for ax, (name, param) in zip(axes, state_dict.items()):
        # 将参数转换为 numpy 数组
        param_np = param.detach().cpu().numpy()

        # 处理不同维度的参数
        if param_np.ndim == 1:
            # 一维参数
            sns.heatmap(param_np.reshape(1, -1), cmap='coolwarm', ax=ax, annot=False, cbar=False)
            ax.axis('off')
        elif param_np.ndim == 2:
            # 二维参数
            sns.heatmap(param_np, cmap='coolwarm', ax=ax, annot=False, cbar=False)
            ax.axis('off')
        elif param_np.ndim == 4:
            # 四维参数，选择第一个切片
            param_np = np.linalg.norm(param_np, ord=1, axis=(2, 3))
            sns.heatmap(param_np, cmap='coolwarm', ax=ax, annot=False, cbar=False)
            ax.axis('off')
        else:
            extra_axes.append(ax)

    # 隐藏多余的子图
    for ax in extra_axes:
        fig.delaxes(ax)

    # 调整子图布局
    plt.tight_layout(pad=0.1)
    plt.show()