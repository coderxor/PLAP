import torch
from torch import nn


class ProjectionLayer(nn.Module):
    """降维层，将高维嵌入映射到低维空间"""

    def __init__(self, input_dim, output_dim):
        """
        初始化投影层

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
        """
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """前向传播"""
        return self.projection(x)