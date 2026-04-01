"""
阶段 3：X-Encoder 模块

使用 LSTM 将时序输入 X (seq_len, feature_dim) 编码到高维特征空间 (hidden_dim,)。
包含:
  - LSTM 时序特征提取器
  - 线性投影���（映射到度量学习空间）
  - 分类头（用于 CrossEntropy 监督训练）
"""

import torch
import torch.nn as nn


class XEncoder(nn.Module):
    """
    X 数据编码器（LSTM + 线性投影）。

    结构: X → LSTM → 取最后隐状态 → Linear → 特征向量 (hidden_dim)

    Args:
        feature_dim: 输入特征维度
        hidden_dim: 编码输出特征维度
        lstm_hidden_size: LSTM 隐状态维��
        lstm_layers: LSTM 层数
        num_classes: 分类头输出的类别数（等于 K-Center 的 K）
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        num_classes: int = 5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes

        # LSTM 时序特征提取器
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,    # 输入 shape: (batch, seq_len, feature_dim)
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )

        # 线性投影层：LSTM 隐状态 → 度量学习特征空间
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 分类头：用于 CrossEntropy 监督训练
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        将 X 编码到高维特征空间。

        Args:
            X: shape (batch, seq_len, feature_dim)

        Returns:
            features: shape (batch, hidden_dim)
        """
        # LSTM 提取时序特征，取最后一个时间步的隐状态
        lstm_out, (h_n, _) = self.lstm(X)
        # lstm_out: (batch, seq_len, lstm_hidden_size)
        # h_n: (num_layers, batch, lstm_hidden_size)

        # 取最后一层 LSTM 的隐状态
        last_hidden = h_n[-1]  # (batch, lstm_hidden_size)

        # 投影到度量学习特征空间
        features = self.projection(last_hidden)  # (batch, hidden_dim)
        return features

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，同时返回特征向量和分类 logits。

        Args:
            X: shape (batch, seq_len, feature_dim)

        Returns:
            features: 编码后的特征, shape (batch, hidden_dim)
            logits: 分类 logits, shape (batch, num_classes)
        """
        features = self.encode(X)       # (batch, hidden_dim)
        logits = self.classifier(features)  # (batch, num_classes)
        return features, logits
