"""多模态骨质疏松预警模型：静态全连接 + 动态 LSTM + 交叉注意力融合 (PyTorch)。

对应申报书 3.1.5：静态通道处理基线特征，动态通道(LSTM)提取时序模式，
融合层采用交叉注意力机制生成联合表征
    H_fusion = Softmax(Q_static · K_dynamicᵀ / √d) · V_dynamic
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


class StaticEncoder(nn.Module):
    """静态特征编码器：多层全连接网络。"""

    def __init__(self, in_dim: int, hidden: tuple = (64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden[-1]

    def forward(self, x):
        return self.net(x)


class DynamicEncoder(nn.Module):
    """动态特征编码器：LSTM 提取时序模式。"""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x):  # x: (B, T, F)
        out, (h, c) = self.lstm(x)
        return out[:, -1, :]  # 取最后时间步隐状态


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合：静态作 Query，动态作 Key/Value。"""

    def __init__(self, static_dim: int, dynamic_dim: int, fusion_dim: int = 32):
        super().__init__()
        self.Wq = nn.Linear(static_dim, fusion_dim)
        self.Wk = nn.Linear(dynamic_dim, fusion_dim)
        self.Wv = nn.Linear(dynamic_dim, fusion_dim)
        self.scale = fusion_dim ** 0.5

    def forward(self, static_rep, dynamic_rep):
        Q = self.Wq(static_rep)              # (B, d)
        K = self.Wk(dynamic_rep)             # (B, d)
        V = self.Wv(dynamic_rep)             # (B, d)
        attn = torch.softmax(torch.matmul(Q, K.T) / self.scale, dim=-1)  # (B, B)
        fused = torch.matmul(attn, V)        # (B, d)
        return fused


class MultiModalOPModel(nn.Module):
    """多模态骨质疏松预警模型。"""

    def __init__(self, n_static: int, n_dynamic: int, seq_len: int,
                 static_hidden=(64, 32), dynamic_hidden=64, fusion_dim=32,
                 dropout=0.3):
        super().__init__()
        self.static_enc = StaticEncoder(n_static, static_hidden, dropout)
        self.dynamic_enc = DynamicEncoder(n_dynamic, dynamic_hidden, dropout=dropout)
        self.fusion = CrossAttentionFusion(self.static_enc.out_dim,
                                           self.dynamic_enc.out_dim, fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, static, dynamic):
        s = self.static_enc(static)
        d = self.dynamic_enc(dynamic)
        fused = self.fusion(s, d)
        logit = self.head(fused)
        return logit.squeeze(-1), fused
