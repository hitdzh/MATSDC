"""
阶段 3：X-Encoder 模块

使用 PatchTST 特征提取器（与 Stage 2 Y-Encoder 同架构）将时序输入 X 编码到高维特征空间。
采用组合模式：复用 PatchTSTFeatureExtractor 作为 backbone，上层添加投影层和分类头。

结构:
  X → PatchTST(RevIN → Overlapping Patches → Transformer → Aggregation) → projection → 特征向量
                                                                                   → classifier → logits
"""

import torch
import torch.nn as nn

from src.layers.PatchTSTEncoder import PatchTSTFeatureExtractor


class XEncoder(nn.Module):
    """
    X 数据编码器（PatchTST backbone + 线性投影 + 分类头）。

    Args:
        feature_dim: 输入特征维度
        hidden_dim: 编码输出特征维度
        num_classes: 分类头输出的类别数（等于 K-Center 的 K）
        seq_len: 输入序列长度
        patch_len: patch 长度
        stride: patch 步长（需 < patch_len 以保证重叠）
        d_model: Transformer 模型维度
        n_heads: 注意力头数
        e_layers: Transformer 编码器层数
        d_ff: 前馈网络维度
        dropout: Dropout 率
        activation: 激活函数
        aggregation: 聚合策略 ('flatten' 保留全部信息, 'max' 取最大激活)
        concat_rev_params: 是否拼接 RevIN 的均值和标准差
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int = 5,
        seq_len: int = 336,
        patch_len: int = 24,
        stride: int = 12,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = 'gelu',
        aggregation: str = 'flatten',
        concat_rev_params: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # PatchTST backbone（组合复用）
        self.backbone = PatchTSTFeatureExtractor(
            c_in=feature_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            aggregation=aggregation,
            target_dim=hidden_dim,
            concat_rev_params=concat_rev_params,
        )

        backbone_output_dim = self.backbone.get_output_dim()

        # 投影层: backbone 输出 → hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 分类头: 用于 CrossEntropy 监督训练
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        将 X 编码到高维特征空间。

        Args:
            X: shape (batch, seq_len, feature_dim)

        Returns:
            features: shape (batch, hidden_dim)
        """
        backbone_features = self.backbone(X)   # (batch, backbone_output_dim)
        features = self.projection(backbone_features)  # (batch, hidden_dim)
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
