import torch
from torch import nn
import torch.nn.functional as F
from log_anomaly_prediction.logprompt.log3p_prediction.config import *


class FeatureFusionLayer(nn.Module):
    """
    特征融合层：使用交叉注意力机制让不同特征部分（SEM/SEQ/TIME/PARAM）之间能够互相增强
    这样可以充分利用多个特征组合时的协同效应
    """
    def __init__(self, part_dim=2 * EMBEDDING_DIM, num_parts=EMBEDDING_PARTS):
        super(FeatureFusionLayer, self).__init__()
        self.part_dim = part_dim
        self.num_parts = num_parts
        
        # 对每个特征部分做独立编码（增强表示能力）
        self.part_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(part_dim, part_dim),
                nn.LayerNorm(part_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_parts)
        ])
        
        # 交叉注意力：让每个part都能关注到其他parts的信息
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=part_dim,
            num_heads=min(4, part_dim // 16),  # 确保num_heads不会太大
            batch_first=True,
            dropout=0.1
        )
        
        # 融合后的维度投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(part_dim, part_dim),
            nn.LayerNorm(part_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_parts, part_dim)
        Returns:
            fused: (batch_size, seq_len, num_parts, part_dim) 融合后的特征
        """
        batch_size, seq_len, num_parts, part_dim = x.size()
        
        # 对每个part做独立编码，增强表示能力
        encoded_parts = []
        for i in range(num_parts):
            part_i = x[:, :, i, :]  # (B, T, D)
            encoded_i = self.part_encoders[i](part_i)  # (B, T, D)
            encoded_parts.append(encoded_i)
        
        # 堆叠所有parts: (B, T, P, D)
        all_parts = torch.stack(encoded_parts, dim=2)  # (B, T, P, D)
        
        # 重塑为 (B, T*P, D) 以便进行交叉注意力（使用reshape避免非连续张量问题）
        all_parts_flat = all_parts.reshape(batch_size, seq_len * num_parts, part_dim)
        
        # 交叉注意力：让每个part都能关注到所有parts的信息，产生协同效应
        fused_flat, _ = self.cross_attention(all_parts_flat, all_parts_flat, all_parts_flat)
        
        # 恢复形状 (B, T, P, D)（使用reshape避免非连续张量问题）
        fused = fused_flat.reshape(batch_size, seq_len, num_parts, part_dim)
        
        # 残差连接：保留原始信息
        fused = fused + all_parts
        
        # 逐时间步、逐part投影（使用reshape避免非连续张量问题）
        fused = fused.reshape(batch_size * seq_len * num_parts, part_dim)
        fused = self.fusion_proj(fused)
        fused = fused.reshape(batch_size, seq_len, num_parts, part_dim)
        
        return fused


class LSTMWithAttention(nn.Module):
    """LSTM模型加注意力机制，带特征融合层"""

    def __init__(self, input_size=EMBEDDING_PARTS * EMBEDDING_DIM * 2, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_labels=NUM_LABELS):
        """
        初始化模型

        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            num_labels: 分类标签数量
        """
        super(LSTMWithAttention, self).__init__()
        
        self.input_size = input_size
        self.part_dim = 2 * EMBEDDING_DIM
        self.num_parts = EMBEDDING_PARTS

        # 特征融合层：让多个特征部分之间能够互相增强
        # 当启用多个特征时，这个层可以让它们产生协同效应，从而比单独使用效果更好
        if EMBEDDING_PARTS > 1:
            self.feature_fusion = FeatureFusionLayer(self.part_dim, self.num_parts)
            self.use_fusion = True
        else:
            self.use_fusion = False

        # 为每个特征部分引入自适应门控（按样本、按时间步）
        # 输入每个 part 的向量，输出该 part 的 gate logit
        self.part_gate_layer = nn.Sequential(
            nn.Linear(self.part_dim, self.part_dim),
            nn.GELU(),
            nn.Linear(self.part_dim, 1)
        )

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.attention_weight_layer = nn.Linear(hidden_size, 1, bias=False)

    def attention(self, lstm_output):
        """
        注意力机制

        Args:
            lstm_output: LSTM输出，形状为(batch_size, seq_len, hidden_size)

        Returns:
            加权后的输出和注意力权重
        """
        # 通过一个线性层计算注意力权重
        attn_weights = self.attention_weight_layer(lstm_output)  # (batch_size, seq_len, 1)

        # 将权重转换为概率分布
        attn_weights = nn.functional.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)

        # 通过广播机制对 lstm_output 进行加权求和
        attn_output = torch.sum(attn_weights * lstm_output, dim=1)  # (batch_size, hidden_size)

        # 返回加权后的输出和权重
        return attn_output, attn_weights

    def forward(self, embeddings):
        """
        前向传播

        Args:
            embeddings: 输入嵌入，形状为(batch_size, seq_len, input_size)

        Returns:
            模型输出和注意力权重
        """
        batch_size, seq_len, feat_dim = embeddings.size()
        part_dim = self.part_dim

        if feat_dim == self.num_parts * part_dim:
            # 重新整理为 (B, T, P, 2D)
            x = embeddings.view(batch_size, seq_len, self.num_parts, part_dim)
            
            # 如果启用多个特征部分，使用特征融合层让它们互相增强
            if self.use_fusion and self.num_parts > 1:
                x = self.feature_fusion(x)  # (B, T, P, D)
            
            # 应用自适应门控权重（按样本、按时间步、按 part）
            gate_logits = self.part_gate_layer(x)              # (B, T, P, 1)
            gate_weights = torch.softmax(gate_logits, dim=2)   # 在 part 维度做 softmax
            x = x * gate_weights                               # (B, T, P, D)
            
            # 展平回原始形状，供 LSTM 使用
            embeddings = x.view(batch_size, seq_len, feat_dim)

        # LSTM + 注意力机制
        lstm_out, _ = self.lstm(embeddings)
        attn_output, attn_weights = self.attention(lstm_out)
        x = self.fc(self.dropout(attn_output))
        
        return x, attn_weights