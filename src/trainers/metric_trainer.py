"""
度量学习联合训练模块

实现 X-Encoder 的联合训练逻辑:
  - 分类损失 (CrossEntropy): 利用伪标签进行监督训练
  - 度量学习损失 (Center Loss): 缩小同类特征在高维空间的欧氏距离
  - 监督对比损失 (SupCon Loss): 在余弦空间优化同类相似度

联合 Loss:
  L_total = L_CE + λ₁ * L_center + λ₂ * L_supcon
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class MetricTrainer:
    """
    X-Encoder 度量学习训练器。

    两个优化器:
      1. Adam: 更新 X-Encoder 参数(encoder + classifier)
      2. SGD:  更新 Center Loss 的 centers 参数（较大学习率控制中心漂移速度）
    """

    def __init__(
        self,
        encoder: nn.Module,
        center_loss: nn.Module,
        supcon_loss: nn.Module = None,
        lr_encoder: float = 1e-3,
        lr_centers: float = 0.5,
        center_loss_weight: float = 1.0,
        supcon_loss_weight: float = 0.1,
        device: str = 'cpu',
    ):
        self.encoder = encoder.to(device)
        self.center_loss = center_loss.to(device)
        self.supcon_loss = supcon_loss.to(device) if supcon_loss else None
        self.center_loss_weight = center_loss_weight
        self.supcon_loss_weight = supcon_loss_weight
        self.device = device

        # 主优化器: Adam 更新编码器所有参数
        self.optimizer_encoder = optim.Adam(
            self.encoder.parameters(), lr=lr_encoder
        )

        # Center Loss 专属优化器: SGD 仅更新 centers 参数
        # 使用较大学习率（如 0.5）使中心能够快速追踪特征分布的变化
        self.optimizer_centers = optim.SGD(
            self.center_loss.parameters(), lr=lr_centers
        )

        # 分类损失
        self.ce_criterion = nn.CrossEntropyLoss()

    def train_epoch(
        self,
        dataloader: DataLoader,
        pseudo_labels: np.ndarray,
    ) -> float:
        """
        训练一个 epoch

        X-Encoder forward -> L_CE -> L_center -> Loss -> backpropagation

        Returns:
            avg_loss: 该 epoch 的平均总损失
        """
        self.encoder.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            # batch_data 可能是 (X_batch,) 或 (X_batch, Y_batch)
            X_batch = batch_data[0].to(self.device)

            # 计算 batch 中样本对应的全局索引
            start = batch_idx * dataloader.batch_size
            end = start + X_batch.size(0)
            batch_labels = torch.from_numpy(pseudo_labels[start:end]).long().to(self.device)

            # 过滤掉异常值标签（-1），仅使用有效标签
            valid_mask = batch_labels >= 0
            if valid_mask.sum() == 0:
                continue

            X_valid = X_batch[valid_mask]
            labels_valid = batch_labels[valid_mask]

            # 前向传播
            features, logits = self.encoder(X_valid)

            # 计算分类损失
            loss_ce = self.ce_criterion(logits, labels_valid)

            # 计算度量学��损失（Center Loss）
            loss_center = self.center_loss(features, labels_valid)

            # 联合损失
            loss_total = loss_ce + self.center_loss_weight * loss_center

            # 监督对比损失（SupCon Loss）
            if self.supcon_loss is not None:
                loss_supcon = self.supcon_loss(features, labels_valid)
                loss_total = loss_total + self.supcon_loss_weight * loss_supcon

            # 反向传播
            self.optimizer_encoder.zero_grad()
            self.optimizer_centers.zero_grad()

            loss_total.backward()

            self.optimizer_encoder.step()
            self.optimizer_centers.step()

            total_loss += loss_total.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def fit(
        self,
        dataloader: DataLoader,
        pseudo_labels: np.ndarray,
        epochs: int,
    ) -> list[float]:
        """
        完整训练
        """
        loss_history = []

        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, pseudo_labels)
            loss_history.append(avg_loss)
            print(f"  Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.4f}")

        return loss_history
