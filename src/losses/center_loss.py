"""
Center Loss

Center Loss 是一种度量学习损失函数，最早由 Wen et al. (ECCV 2016) 提出，
用于深度特征学习中的类内紧致性约束。

数学定义:
  L_center = (1 / 2B) * Σ_{i=1}^{B} || f(x_i) - c_{y_i} ||^2

  其中:
    f(x_i) ∈ R^d  : 样本 x_i 经过编码器后的特征向量
    c_{y_i} ∈ R^d : 样本 x_i 所属类别 y_i 的中心向量
    B              : 批大小

中心向量的更新规则:
  c_k ← c_k - alpha * Δc_k
  其中 Δc_k = (Σ_{y_i=k} f(x_i) - n_k * c_k) / (1 + n_k)

  等价于: c_k 被拉向该 batch 中类别 k 的所有样本特征的均值方向。
  alpha 通常取较大值 (如 0.5)，通过独立的 SGD 优化器实现。
"""

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center Loss 度量学习损失函数。

    为 K 个类别各维护一个可学习的中心向量 c_k ∈ R^{feat_dim}。
    训练时将同类样本特征拉向其类别中心，实现类内紧致。

    Args:
        num_classes: 类别数量 K
        feat_dim: 特征向量维度 d
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # K 个类别中心向量，作为可学习参数
        # 初始化方式: 随机小值，与特征空间同尺度
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim) * 0.1
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算 Center Loss。

        数学过程:
          1. 从 self.centers 中取出每个样本对应类别的中心: c_{y_i}
          2. 计算每个样本特征与其类别中心的欧氏距离平方: ||f(x_i) - c_{y_i}||^2
          3. 对所有样本求均值: (1 / 2B) * Σ ||f(x_i) - c_{y_i}||^2

        梯度传播机制:
          - 对 features 的梯度: ∂L/∂f(x_i) = (f(x_i) - c_{y_i}) / B
            → 编码器参数收到梯度，使特征向中心靠近
          - 对 centers 的梯度: ∂L/∂c_k = -Σ_{y_i=k}(f(x_i) - c_k) / B
            → 通过独立的 SGD 优化器更新 centers（反向传播自动处理）

        注意: centers 的更新通过独立的优化器控制学习率 α，
        而不是直接使用主优化器，这样可以对中心漂移进行更精细的控制。

        Args:
            features: 编码器输出的特征向量, shape (B, feat_dim)
            labels: 样本的伪标签, shape (B,), 值域 [0, K)

        Returns:
            loss: 标量 Center Loss 值
        """
        batch_size = features.size(0)

        # 1. 取出每个样本对应类别的中心向量
        # labels shape: (B,) → centers_of_labels shape: (B, feat_dim)
        centers_of_labels = self.centers[labels]

        # 2. 计算每个样本特征与其类别中心的欧氏距离平方
        # ||f(x_i) - c_{y_i}||^2 = Σ_d (f_id - c_id)^2
        diff = features - centers_of_labels  # (B, feat_dim)
        loss = 0.5 * torch.sum(diff ** 2) / batch_size

        return loss
