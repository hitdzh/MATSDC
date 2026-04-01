"""
完整 Pipeline 运行脚本

按顺序执行所有 4 个阶段：
  阶段 1: 数据准备 → 切分滑动窗口 (X, Y)
  阶段 2: Y-PreEnocder + K-Center 聚类
  阶段 3: X 空间度量学习 → CE + Center Loss 联合训练
  阶段 4: 谱聚类 + 典型样本提取

用法:
  python scripts/run_full_pipeline.py
  python scripts/run_full_pipeline.py --dataset ETTh1 --seq_len 336 --pre_len 96
  python scripts/run_full_pipeline.py --resume_from 3  # 从阶段3开始续跑
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from configs.default import PipelineConfig
from src.data.sliding_window import SlidingWindowDataset
from src.data.dataset_factory import TimeSeriesForecastDataset
from src.layers.PatchTSTEncoder import PatchTSTFeatureExtractor
from src.models.x_encoder import XEncoder
from src.clustering.kcenter import generate_pseudo_labels
from src.losses.center_loss import CenterLoss
from src.trainers.metric_trainer import MetricTrainer
from src.clustering.spectral_selector import SpectralSelector


def run_stage1(cfg: PipelineConfig, args) -> tuple[torch.Tensor, torch.Tensor]:
    """阶段 1"""
    print(f"[Stage 1] Data Preparation — dataset={cfg.dataset}")

    if cfg.dataset == 'dummy':
        np.random.seed(42)
        time_series = np.random.randn(args.T, cfg.feature_dim).astype(np.float32)
        dataset = SlidingWindowDataset(time_series, cfg.seq_len, cfg.pre_len)
    else:
        dataset = TimeSeriesForecastDataset(
            dataset_name=cfg.dataset,
            split='train',
            seq_len=cfg.seq_len,
            pre_len=cfg.pre_len,
        )
        cfg.feature_dim = dataset.feature_dim

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    all_X, all_Y = [], []
    for X_batch, Y_batch in dataloader:
        all_X.append(X_batch)
        all_Y.append(Y_batch)
    all_X = torch.cat(all_X, dim=0)
    all_Y = torch.cat(all_Y, dim=0)

    print(f"  X shape: {all_X.shape}")
    print(f"  Y shape: {all_Y.shape}")
    print(f"  Total samples: {all_X.shape[0]}")

    # 保存中间结果
    os.makedirs('outputs/stage1', exist_ok=True)
    torch.save({'X': all_X, 'Y': all_Y, 'config': cfg},
               f'outputs/stage1/{cfg.dataset}_data.pt')

    return all_X, all_Y


class _ReconDecoder(nn.Module):
    """线性解码器：特征向量 → 重建原始时序窗口"""
    def __init__(self, feat_dim, seq_len, c_in):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, seq_len * c_in),
        )

    def forward(self, features):
        B = features.size(0)
        return self.net(features).reshape(B, self.seq_len, self.c_in)


def _pretrain_on_y(encoder, Y_data, epochs, batch_size, lr, train_ratio, device):
    """自监督重建训练编码器。"""
    feat_dim = encoder.get_output_dim()
    seq_len = Y_data.size(1)
    c_in = Y_data.size(2)
    decoder = _ReconDecoder(feat_dim, seq_len, c_in).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()

    # 自监督处理
    N = Y_data.size(0)
    n_train = int(N * train_ratio)
    # 打乱重排
    idx = torch.randperm(N)
    Y_shuf = Y_data[idx]

    train_loader = DataLoader(TensorDataset(Y_shuf[:n_train]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Y_shuf[n_train:]), batch_size=batch_size)

    print(f"\nTraining encoder ({epochs} epochs, train={n_train})")

    best_val = float('inf')
    best_state = None
    for epoch in range(epochs):
        encoder.train(); decoder.train()
        t_loss, n = 0.0, 0
        for b in train_loader:
            x = b[0].to(device)
            recon = decoder(encoder(x))
            loss = criterion(recon, x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item(); n += 1
        t_loss /= max(n, 1)

        encoder.eval(); decoder.eval()
        v_loss, n = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                x = b[0].to(device)
                recon = decoder(encoder(x))
                v_loss += criterion(recon, x).item(); n += 1
        v_loss /= max(n, 1)

        mark = ''
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            mark = ' *'
        print(f"    [{epoch+1}/{epochs}] Train: {t_loss:.6f}  Val: {v_loss:.6f}{mark}")

    if best_state:
        encoder.load_state_dict(best_state)
    encoder.eval()
    print(f"  Pretraining done. Best val loss: {best_val:.6f}")
    return encoder


def _build_y_encoder(cfg, device, pretrained_path=None, Y_data=None):
    """获取y_encoder"""
    # 读取预训练的 y_encoder (如果存在)
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, weights_only=False)
        ec = ckpt['encoder_config']
        encoder = PatchTSTFeatureExtractor(
            c_in=ec['c_in'], seq_len=ec['seq_len'], patch_len=ec['patch_len'],
            stride=ec['stride'], d_model=ec['d_model'], n_heads=ec['n_heads'],
            e_layers=ec['e_layers'], d_ff=ec['d_ff'], activation='gelu',
            aggregation=ec['aggregation'], concat_rev_params=ec['concat_rev_params'],
        ).to(device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        encoder.eval()
        print(f"  Loaded pretrained encoder: {pretrained_path}")
        cfg.y_hidden_dim = ec['output_dim']
        return encoder

    # 就地训练
    assert Y_data is not None
    patch_len = min(16, cfg.pre_len)
    stride = max(1, patch_len // 2)
    n_heads = max(1, cfg.y_hidden_dim // 32)
    encoder = PatchTSTFeatureExtractor(
        c_in=cfg.feature_dim, seq_len=cfg.pre_len,
        patch_len=patch_len, stride=stride,
        d_model=cfg.y_hidden_dim, n_heads=n_heads,
        e_layers=1, d_ff=cfg.y_hidden_dim * 2,
        dropout=0.1, activation='gelu',
        aggregation='max', concat_rev_params=True,
    ).to(device)
    return _pretrain_on_y(encoder, Y_data, cfg.pretrain_epochs, cfg.pretrain_batch_size,
                          cfg.pretrain_lr, 0.8, device)


def run_stage2(all_X, all_Y, cfg, device, pretrained_encoder=None) -> np.ndarray:
    """阶段 2"""
    print(f"\n[Stage 2] Pseudo Labeling")

    y_encoder = _build_y_encoder(cfg, device, pretrained_encoder, Y_data=all_Y)

    pseudo_labels, centers_idx = generate_pseudo_labels(
        Y_data=all_Y.to(device),
        encoder=y_encoder,
        K=cfg.K,
        method=cfg.kcenter_method,
    )

    label_dist = [int((pseudo_labels == k).sum()) for k in range(cfg.K)]
    n_outliers = int((pseudo_labels == -1).sum())
    # 后续用于加权参考
    print(f"  Label distribution: {label_dist}")
    if n_outliers > 0:
        print(f"  Outliers: {n_outliers}")

    # 保存
    os.makedirs('outputs/stage2', exist_ok=True)
    torch.save({'pseudo_labels': pseudo_labels, 'centers_idx': centers_idx, 'config': cfg},
               'outputs/stage2/pseudo_labels.pt')

    return pseudo_labels


def run_stage3(all_X, all_Y, pseudo_labels, cfg: PipelineConfig, device: str) -> torch.Tensor:
    """阶段 3"""
    print(f"\n[Stage 3] X-Encoder Metric Learning Training")

    # 过滤异常值
    valid_mask = pseudo_labels >= 0
    valid_X = all_X[valid_mask]
    valid_labels = pseudo_labels[valid_mask]

    # 构建模型
    x_encoder = XEncoder(
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.x_hidden_dim,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_layers=cfg.lstm_layers,
        num_classes=cfg.K,
    )

    center_loss = CenterLoss(
        num_classes=cfg.K,
        feat_dim=cfg.x_hidden_dim,
    )

    trainer = MetricTrainer(
        encoder=x_encoder,
        center_loss=center_loss,
        lr_encoder=cfg.lr_encoder,
        lr_centers=cfg.lr_centers,
        center_loss_weight=cfg.center_loss_weight,
        device=device,
    )

    # 训练
    train_dataset = TensorDataset(valid_X)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    loss_history = trainer.fit(
        dataloader=train_loader,
        pseudo_labels=valid_labels,
        epochs=cfg.epochs,
    )

    # 提取全量特征
    x_encoder.eval()
    with torch.no_grad():
        all_features = x_encoder.encode(all_X.to(device))

    # 保存
    os.makedirs('outputs/stage3', exist_ok=True)
    torch.save({
        'encoder_state_dict': x_encoder.state_dict(),
        'center_loss_state_dict': center_loss.state_dict(),
        'loss_history': loss_history,
        'features': all_features.cpu(),
        'config': cfg,
        'X': all_X,
        'Y': all_Y,
    }, 'outputs/stage3/trained_encoder.pt')

    return all_features


def run_stage4(all_X, all_Y, all_features, cfg: PipelineConfig) -> dict:
    """阶段 4"""
    print(f"\n[Stage 4] Spectral Clustering & Prototype Extraction")

    selector = SpectralSelector(
        n_clusters=cfg.n_clusters,
        n_prototypes=cfg.n_prototypes,
        sigma=cfg.spectral_sigma,
    )

    prototypes = selector.cluster_and_select(
        features=all_features,
        X_raw=all_X,
        Y_raw=all_Y,
    )

    # 保存
    os.makedirs('outputs/stage4', exist_ok=True)
    torch.save({'prototypes': prototypes, 'config': cfg},
               'outputs/stage4/prototypes.pt')

    # 保存数据
    label_len = cfg.seq_len // 2

    x_enc = torch.from_numpy(prototypes['X'])                  # (N, seq_len, feature_dim)
    x_dec = torch.cat([x_enc[:, -label_len:, :],
                       torch.from_numpy(prototypes['Y'])], dim=1)  # (N, label_len+pred_len, feature_dim)

    train_data = {
        'x_enc': x_enc,         # encoder 输入: (N, seq_len, feature_dim)
        'x_dec': x_dec,         # decoder 输入: (N, label_len+pred_len, feature_dim)
        'y_enc': prototypes['Y'],  # encoder 对应的目标: (N, pred_len, feature_dim)
        'features': prototypes['features'],
        'cluster_ids': prototypes['cluster_ids'],
        'indices': prototypes['indices'],
        'config': cfg,
    }

    os.makedirs('outputs/CondensedDatasets', exist_ok=True)
    torch.save(train_data, f'outputs/CondensedDatasets/Condensed_{cfg.dataset}.pt')
    print(f"\n  Saved Condensed_{cfg.dataset}.pt — x_enc: {x_enc.shape}, x_dec: {x_dec.shape}")

    return prototypes


def main():
    parser = argparse.ArgumentParser(description='Full Pipeline: Time Series Prototype Mining')
    parser.add_argument('--dataset', type=str, default='dummy',
                        choices=['dummy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                                 'electricity', 'traffic', 'weather'])
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pre_len', type=int, default=96)
    parser.add_argument('--feature_dim', type=int, default=3, help='仅 dummy')
    parser.add_argument('--T', type=int, default=1000, help='仅 dummy')
    parser.add_argument('--K', type=int, default=5, help='K-Center 聚类数')
    parser.add_argument('--n_clusters', type=int, default=4, help='谱聚类簇数')
    parser.add_argument('--n_prototypes', type=int, default=4, help='提取典型样本数')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='Y-Encoder 预训练轮数')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='Y-Encoder 预训练学习率')
    parser.add_argument('--pretrain_batch_size', type=int, default=64, help='Y-Encoder 预训练批大小')
    parser.add_argument('--resume_from', type=int, default=1, choices=[1, 2, 3, 4],
                        help='从指定阶段开始（需有前序输出）')
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                        help='预训练 Y-Encoder 路径 (pretrain_y_encoder.py 输出) ')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 构建配置
    cfg = PipelineConfig(
        dataset=args.dataset,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        feature_dim=args.feature_dim,
        K=args.K,
        n_clusters=args.n_clusters,
        n_prototypes=args.n_prototypes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_batch_size=args.pretrain_batch_size,
    )

    # ===== 阶段 1 =====
    if args.resume_from <= 1:
        all_X, all_Y = run_stage1(cfg, args)
    else:
        print(f"\n[Stage 1] Skipped (resume_from={args.resume_from})")
        data = torch.load(f'outputs/stage1/{cfg.dataset}_data.pt', weights_only=False)
        all_X, all_Y = data['X'], data['Y']
        cfg = data['config']

    # ===== 阶段 2 =====
    if args.resume_from <= 2:
        pseudo_labels = run_stage2(all_X, all_Y, cfg, device,
                                   pretrained_encoder=args.pretrained_encoder)
    else:
        print(f"\n[Stage 2] Skipped (resume_from={args.resume_from})")
        data = torch.load('outputs/stage2/pseudo_labels.pt', weights_only=False)
        pseudo_labels = data['pseudo_labels']
        cfg = data['config']

    # ===== 阶段 3 =====
    if args.resume_from <= 3:
        all_features = run_stage3(all_X, all_Y, pseudo_labels, cfg, device)
    else:
        print(f"\n[Stage 3] Skipped (resume_from={args.resume_from})")
        data = torch.load('outputs/stage3/trained_encoder.pt', weights_only=False)
        all_features = data['features']
        all_X, all_Y = data['X'], data['Y']
        cfg = data['config']

    # ===== 阶段 4 =====
    prototypes = run_stage4(all_X, all_Y, all_features, cfg)

    # 最终结果
    print("\n")
    print(f"  N = {cfg.n_prototypes} prototypes extracted")
    print(f"  X shape:          {prototypes['X'].shape}")
    print(f"  Y shape:          {prototypes['Y'].shape}")
    print(f"  features shape:   {prototypes['features'].shape}")
    print(f"  cluster_ids:      {prototypes['cluster_ids']}")
    print(f"  indices:          {prototypes['indices']}")


if __name__ == '__main__':
    main()
