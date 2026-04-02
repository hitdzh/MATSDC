"""
Supervised Contrastive Loss (SupCon)

基于 Khosla et al. (NeurIPS 2020) "Supervised Contrastive Learning"。

对每个 anchor 样本 i:
  L_i = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(z_i·z_p / τ) / Σ_{a≠i} exp(z_i·z_a / τ) ]

其中:
  P(i) = {p ∈ batch : y_p == y_i 且 p ≠ i}  — 同标签的正样本集
  τ 为温度参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss。

    在余弦空间中优化：同类样本相似度最大化，异类样本相似度最小化。
    与 Center Loss（欧氏空间紧致）互补。

    Args:
        temperature: 温度参数 τ，控制对比强度。较小的 τ 使模型更关注困难负样本。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 编码器输出的特征向量, shape (B, D)
            labels: 样本的伪标签, shape (B,), 值域 [0, K)

        Returns:
            loss: 标量 SupCon Loss 值
        """
        device = features.device
        batch_size = features.size(0)

        # L2 归一化，将特征映射到单位超球面（余弦相似度空间）
        features = F.normalize(features, p=2, dim=1)

        # 相似度矩阵: (B, B)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 正样本掩码: 同标签且非自身
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        self_mask = torch.eye(batch_size, device=device).bool()
        positive_mask = labels_eq & ~self_mask  # (B, B)

        # 数值稳定: 减去每行最大值
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # log-softmax over all non-self samples
        exp_logits = torch.exp(logits) * ~self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 对每个 anchor，取其所有正样本对的平均 log 概率
        n_positives = positive_mask.sum(dim=1)  # (B,)
        mean_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / (n_positives + 1e-12)

        # 仅对有正样本的 anchor 计算损失
        has_positives = n_positives > 0
        if has_positives.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -mean_log_prob[has_positives].mean()
        return loss
