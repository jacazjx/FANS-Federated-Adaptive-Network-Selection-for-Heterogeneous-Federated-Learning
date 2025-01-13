
__all__ = [
    "accuracy",
    "FedDynLoss"
]

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class FedDynLoss(nn.Module):
    def __init__(self, lambda_reg):
        super(FedDynLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, global_params: OrderedDict, model: nn.Module):
        # 计算交叉熵损失
        loss = 0
        # 添加正则化项
        for param in model.parameters():
            if param.grad is not None:
                loss += torch.norm(param - global_params[param.name]) ** 2

        return self.lambda_reg * loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, mask):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mask = mask

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): 模型的输出，形状为 (batch_size, num_classes, ...)。
            targets (torch.Tensor): 真实标签，形状为 (batch_size, ...)。
            mask (torch.Tensor): 掩码，0 表示不计算损失，1 表示计算损失，形状与 targets 相同。

        Returns:
            torch.Tensor: 计算后的损失值。
        """
        # 使用 CrossEntropyLoss 进行损失计算，返回每个位置的损失
        loss = F.cross_entropy(logits, targets, reduction='none')  # shape: (batch_size, ...)

        # 应用掩码，将掩码为 0 的位置的损失置为 0
        loss = loss * self.mask

        # 计算有效位置的平均损失
        return loss.sum() / self.mask.sum()


class KLLoss(nn.Module):
    """KL divergence loss for self distillation."""

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred, label):
        """KL loss forward."""
        predict = F.log_softmax(pred / self.temperature, dim=1)
        target_data = F.softmax(label / self.temperature, dim=1)
        target_data = target_data + 10 ** (-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss = (
            self.temperature
            * self.temperature
            * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        )
        return loss