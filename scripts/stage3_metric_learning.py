"""
阶段 3：X 空间度量学习训练

加载阶段 2 输出的 (X, Y, pseudo_labels)，构建 X-Encoder 和 Center Loss，
执行联合训练（CE + Center Loss），保存训练好的模型和提取的特征。

用法:
  python scripts/stage3_metric_learning.py --input outputs/stage2/pseudo_labels.pt
  python scripts/stage3_metric_learning.py --input outputs/stage2/pseudo_labels.pt --epochs 10
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.default import PipelineConfig
from src.models.x_encoder import XEncoder
from src.losses.center_loss import CenterLoss
from src.trainers.metric_trainer import MetricTrainer


def main():
    parser = argparse.ArgumentParser(description='Stage 3: X-Encoder Metric Learning Training')
    parser.add_argument('--input', type=str, required=True, help='阶段2 输出的 .pt 文件路径')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--lr_encoder', type=float, default=None, help='X-Encoder 学习率')
    parser.add_argument('--lr_centers', type=float, default=None, help='Center Loss 学习率')
    parser.add_argument('--center_loss_weight', type=float, default=None, help='Center Loss 权重 λ')
    parser.add_argument('--output_dir', type=str, default='outputs/stage3',
                        help='输出目录')
    args = parser.parse_args()

    # 加载阶段 2 数据
    data = torch.load(args.input, weights_only=False)
    all_X = data['X']           # (N, seq_len, feature_dim)
    all_Y = data['Y']           # (N, pre_len, feature_dim)
    pseudo_labels = data['pseudo_labels']  # (N,)
    cfg: PipelineConfig = data['config']

    # 覆盖训练参数
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr_encoder is not None:
        cfg.lr_encoder = args.lr_encoder
    if args.lr_centers is not None:
        cfg.lr_centers = args.lr_centers
    if args.center_loss_weight is not None:
        cfg.center_loss_weight = args.center_loss_weight

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Stage 3] X-Encoder Metric Learning Training")
    print(f"  epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
          f"lr={cfg.lr_encoder}, lr_centers={cfg.lr_centers}, λ={cfg.center_loss_weight}")

    # 构建 X-Encoder
    x_encoder = XEncoder(
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.x_hidden_dim,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_layers=cfg.lstm_layers,
        num_classes=cfg.K,
    )

    # 构建 Center Loss
    center_loss = CenterLoss(
        num_classes=cfg.K,
        feat_dim=cfg.x_hidden_dim,
    )

    # 构建训练器
    trainer = MetricTrainer(
        encoder=x_encoder,
        center_loss=center_loss,
        lr_encoder=cfg.lr_encoder,
        lr_centers=cfg.lr_centers,
        center_loss_weight=cfg.center_loss_weight,
        device=device,
    )

    # 过滤异常值标签
    valid_mask = pseudo_labels >= 0
    valid_X = all_X[valid_mask]
    valid_labels = pseudo_labels[valid_mask]

    print(f"  Valid samples: {valid_X.shape[0]} / {all_X.shape[0]}")

    # 构建训练 DataLoader
    train_dataset = TensorDataset(valid_X)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # 训练
    loss_history = trainer.fit(
        dataloader=train_loader,
        pseudo_labels=valid_labels,
        epochs=cfg.epochs,
    )

    # 提取全量 X 的最终特征
    x_encoder.eval()
    with torch.no_grad():
        all_features = x_encoder.encode(all_X.to(device))

    print(f"  Final features shape: {all_features.shape}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'trained_encoder.pt')
    torch.save({
        'encoder_state_dict': x_encoder.state_dict(),
        'center_loss_state_dict': center_loss.state_dict(),
        'loss_history': loss_history,
        'features': all_features.cpu(),
        'config': cfg,
        'X': all_X,
        'Y': all_Y,
    }, save_path)
    print(f"  Saved to {save_path}")


if __name__ == '__main__':
    main()
