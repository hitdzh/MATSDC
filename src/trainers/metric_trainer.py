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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import numpy as np
from contextlib import nullcontext


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
        class_weights: torch.Tensor | None = None,
        min_lr: float = 1e-5,
        max_grad_norm: float | None = 1.0,
        device: str = 'cpu',
    ):
        self.encoder = encoder.to(device)
        self.center_loss = center_loss.to(device)
        self.supcon_loss = supcon_loss.to(device) if supcon_loss else None
        self.center_loss_weight = center_loss_weight
        self.supcon_loss_weight = supcon_loss_weight
        self.device = device
        self.transfer_non_blocking = device == 'cuda'
        self.min_lr = min_lr
        self.max_grad_norm = max_grad_norm

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
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.scheduler = None

        self.autocast_context, self.amp_enabled, self.amp_dtype = self._get_amp_context()
        if device == 'cuda':
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        else:
            self.scaler = None

    def _get_amp_context(self):
        if self.device != 'cuda':
            return nullcontext, False, None

        supports_bf16 = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16

        def _autocast():
            if hasattr(torch, 'autocast'):
                return torch.autocast(device_type='cuda', dtype=amp_dtype)
            return torch.cuda.amp.autocast(dtype=amp_dtype)

        return _autocast, True, amp_dtype

    def train_epoch(
        self,
        dataloader: DataLoader,
        pseudo_labels: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        训练一个 epoch

        Returns:
            包含 total / ce / center / supcon 的平均损失统计
        """
        self.encoder.train()
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_center = 0.0
        total_loss_supcon = 0.0
        n_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            # 训练阶段期望 batch 至少包含输入 X，推荐同时包含对应标签
            X_batch = batch_data[0].to(self.device, non_blocking=self.transfer_non_blocking)

            if len(batch_data) > 1:
                batch_labels = batch_data[1].long().to(self.device, non_blocking=self.transfer_non_blocking)
            elif pseudo_labels is not None:
                # 兼容旧调用方式：当 DataLoader 未携带标签时，按顺序切片。
                start = batch_idx * dataloader.batch_size
                end = start + X_batch.size(0)
                batch_labels = torch.from_numpy(pseudo_labels[start:end]).long().to(
                    self.device,
                    non_blocking=self.transfer_non_blocking,
                )
            else:
                raise ValueError("MetricTrainer requires labels in the batch or via pseudo_labels.")

            # 过滤掉异常值标签（-1），仅使用有效标签
            valid_mask = batch_labels >= 0
            if valid_mask.sum() == 0:
                continue

            X_valid = X_batch[valid_mask]
            labels_valid = batch_labels[valid_mask]

            # 反向传播
            self.optimizer_encoder.zero_grad(set_to_none=True)
            self.optimizer_centers.zero_grad(set_to_none=True)

            loss_supcon_value = 0.0
            with self.autocast_context():
                features, logits = self.encoder(X_valid)
                loss_ce = self.ce_criterion(logits, labels_valid)
                loss_center = self.center_loss(features, labels_valid)
                loss_total = loss_ce + self.center_loss_weight * loss_center
                if self.supcon_loss is not None:
                    loss_supcon = self.supcon_loss(features, labels_valid)
                    loss_total = loss_total + self.supcon_loss_weight * loss_supcon
                    loss_supcon_value = loss_supcon.item()

            if self.amp_enabled:
                self.scaler.scale(loss_total).backward()
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer_encoder)
                    self.scaler.unscale_(self.optimizer_centers)
                    clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                    clip_grad_norm_(self.center_loss.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer_encoder)
                self.scaler.step(self.optimizer_centers)
                self.scaler.update()
            else:
                loss_total.backward()
                if self.max_grad_norm is not None:
                    clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                    clip_grad_norm_(self.center_loss.parameters(), self.max_grad_norm)
                self.optimizer_encoder.step()
                self.optimizer_centers.step()

            total_loss += loss_total.item()
            total_loss_ce += loss_ce.item()
            total_loss_center += loss_center.item()
            total_loss_supcon += loss_supcon_value
            n_batches += 1

        denom = max(n_batches, 1)
        return {
            "total": total_loss / denom,
            "ce": total_loss_ce / denom,
            "center": total_loss_center / denom,
            "supcon": total_loss_supcon / denom,
        }

    def fit(
        self,
        dataloader: DataLoader,
        pseudo_labels: np.ndarray | None,
        epochs: int,
    ) -> list[dict[str, float]]:
        """
        完整训练
        """
        loss_history = []
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_encoder,
            T_max=max(1, epochs),
            eta_min=self.min_lr,
        )

        for epoch in range(epochs):
            if epoch == 0 and self.amp_enabled:
                print(f"  Stage 3 AMP enabled on CUDA ({self.amp_dtype})")
            epoch_metrics = self.train_epoch(dataloader, pseudo_labels)
            loss_history.append(epoch_metrics)
            current_lr = self.optimizer_encoder.param_groups[0]['lr']
            print(
                f"  Epoch [{epoch + 1}/{epochs}]  "
                f"Total: {epoch_metrics['total']:.4f}  "
                f"CE: {epoch_metrics['ce']:.4f}  "
                f"Center: {epoch_metrics['center']:.4f}  "
                f"SupCon: {epoch_metrics['supcon']:.4f}  "
                f"LR: {current_lr:.6g}"
            )
            self.scheduler.step()

        return loss_history
