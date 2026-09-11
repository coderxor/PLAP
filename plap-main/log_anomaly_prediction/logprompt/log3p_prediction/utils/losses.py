import torch
from torch import nn
from log_anomaly_prediction.logprompt.log3p_prediction.config import *


class FocalLoss(nn.Module):
    """Focal Loss实现，用于处理不平衡数据集"""

    def __init__(self, alpha=None, gamma=2, weight=None, reduction='mean'):
        """
        初始化Focal Loss

        Args:
            alpha: 平衡因子，增大会更关注少数类
            gamma: 聚焦参数，增大会更关注难分类样本
            weight: 类别权重
            reduction: 损失归约方式
        """
        super(FocalLoss, self).__init__()
        # 如果没有指定alpha，根据数据集选择默认值
        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha
        
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算损失

        Args:
            inputs: 模型输出
            targets: 目标标签

        Returns:
            计算的损失值
        """
        BCE_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss