"""
时序数据无监督样本挖掘 Pipeline — 验证脚本

使用随机生成的 dummy 时序数据，按顺序调用四个阶段，
验证整个 Pipeline 的可运行性和数据 shape 一致性。

使用 src/ 目录结构，与 CondensedTSF 项目风格保持一致。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

# 导入各阶段模块（src/ 目录结构）
from configs.default import PipelineConfig
from src.data.sliding_window import SlidingWindowDataset
from src.layers.PatchTSTEncoder import PatchTSTFeatureExtractor, create_patchtst_encoder
from src.models.x_encoder import XEncoder
from src.clustering.kcenter import generate_pseudo_labels
from src.losses.center_loss import CenterLoss
from src.losses.supcon_loss import SupervisedContrastiveLoss
from src.trainers.metric_trainer import MetricTrainer
from src.clustering.spectral_selector import SpectralSelector


def main():
    # ==================== 配置 ====================
    args = PipelineConfig(
        seq_len=50,
        pre_len=10,
        feature_dim=3,
        K=5,
        n_clusters=4,
        n_prototypes=4,
        epochs=2,
        batch_size=32,
        lr_encoder=1e-3,
        lr_centers=0.5,
        center_loss_weight=1.0,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: seq_len={args.seq_len}, pre_len={args.pre_len}, "
          f"feature_dim={args.feature_dim}, K={args.K}, "
          f"n_clusters={args.n_clusters}, n_prototypes={args.n_prototypes}")
    print("=" * 60)

    # ==================== 阶段 1：数据构建 ====================
    print("\n[Stage 1] Sliding Window Data Construction")
    np.random.seed(42)
    T = 1000
    time_series = np.random.randn(T, args.feature_dim).astype(np.float32)

    dataset = SlidingWindowDataset(time_series, args.seq_len, args.pre_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    n_samples = len(dataset)
    print(f"  Time series length T={T}, samples N={n_samples}")

    # Pre-extract all X, Y for subsequent stages
    all_X, all_Y = [], []
    for X_batch, Y_batch in dataloader:
        all_X.append(X_batch)
        all_Y.append(Y_batch)
    all_X = torch.cat(all_X, dim=0)  # (N, seq_len, feature_dim)
    all_Y = torch.cat(all_Y, dim=0)  # (N, pre_len, feature_dim)
    print(f"  X shape: {all_X.shape}")
    print(f"  Y shape: {all_Y.shape}")

    # ==================== 阶段 2：Y 空间伪打标 ====================
    print("\n[Stage 2] Y-Space K-Center Pseudo Labeling")

    # Build Y-PreEncoder using PatchTSTFeatureExtractor
    # Note: pre_len is used as seq_len since Y is the target window
    y_encoder = PatchTSTFeatureExtractor(
        c_in=args.feature_dim,
        seq_len=args.pre_len,
        patch_len=4,           # Small patch for short Y sequences
        stride=2,              # 50% overlap
        d_model=args.y_hidden_dim,
        n_heads=2,
        e_layers=1,
        d_ff=128,
        dropout=0.1,
        activation='gelu',
        aggregation='max',
        concat_rev_params=True,
    ).to(device)

    # Generate pseudo labels via K-Center clustering on Y features
    pseudo_labels, centers_idx = generate_pseudo_labels(
        Y_data=all_Y.to(device),
        encoder=y_encoder,
        K=args.K,
        method=args.kcenter_method,
    )
    print(f"  K-Center method: {args.kcenter_method}")
    print(f"  Pseudo label distribution: {[int((pseudo_labels == k).sum()) for k in range(args.K)]}")
    print(f"  Center indices: {centers_idx}")
    print(f"  pseudo_labels shape: {pseudo_labels.shape}")

    # ==================== 阶段 3：X 空间度量学习 ====================
    print("\n[Stage 3] X-Encoder Metric Learning Training")

    # Build X-Encoder (PatchTST backbone)
    x_encoder = XEncoder(
        feature_dim=args.feature_dim,
        hidden_dim=args.x_hidden_dim,
        num_classes=args.K,
        seq_len=args.seq_len,
        patch_len=4,           # Small patch for short test sequences (seq_len=50)
        stride=2,              # 50% overlap
        d_model=args.x_hidden_dim,
        n_heads=2,
        e_layers=1,
        d_ff=128,
        dropout=0.1,
        activation='gelu',
        aggregation='max',     # Use max for small test sequences
        concat_rev_params=True,
    )

    # Build Center Loss
    center_loss = CenterLoss(
        num_classes=args.K,
        feat_dim=args.x_hidden_dim,
    )

    # Build Supervised Contrastive Loss
    supcon_loss = SupervisedContrastiveLoss(
        temperature=args.supcon_temperature,
    )

    # Build trainer and train
    trainer = MetricTrainer(
        encoder=x_encoder,
        center_loss=center_loss,
        supcon_loss=supcon_loss,
        lr_encoder=args.lr_encoder,
        lr_centers=args.lr_centers,
        center_loss_weight=args.center_loss_weight,
        supcon_loss_weight=args.supcon_loss_weight,
        device=device,
    )

    # Filter out outlier labels (-1), create DataLoader for valid data
    valid_mask = pseudo_labels >= 0
    valid_X = all_X[valid_mask]
    valid_labels = pseudo_labels[valid_mask]

    # Create new DataLoader (shuffle=True for better training)
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(valid_X)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training
    loss_history = trainer.fit(
        dataloader=train_loader,
        pseudo_labels=valid_labels,
        epochs=args.epochs,
    )

    # Extract final features for all X (for Stage 4)
    x_encoder.eval()
    with torch.no_grad():
        all_features = x_encoder.encode(all_X.to(device))  # (N, hidden_dim)
    print(f"  X-Encoder output features shape: {all_features.shape}")

    # ==================== 阶段 4：谱聚类 + 典型样本提取 ====================
    print("\n[Stage 4] Spectral Clustering & Prototype Extraction")

    selector = SpectralSelector(
        n_clusters=args.n_clusters,
        n_prototypes=args.n_prototypes,
        sigma=args.spectral_sigma,
    )

    prototypes = selector.cluster_and_select(
        features=all_features,
        X_raw=all_X,
        Y_raw=all_Y,
    )

    # ==================== 结果打印 ====================
    print("\n" + "=" * 60)
    print("Prototype Extraction Results:")
    print(f"  N = {args.n_prototypes} prototypes")
    print(f"  X shape:          {prototypes['X'].shape}")
    print(f"  Y shape:          {prototypes['Y'].shape}")
    print(f"  features shape:   {prototypes['features'].shape}")
    print(f"  cluster_ids:      {prototypes['cluster_ids']}")
    print(f"  indices:          {prototypes['indices']}")
    print("=" * 60)
    print("\nPipeline verification complete!")


if __name__ == '__main__':
    main()
