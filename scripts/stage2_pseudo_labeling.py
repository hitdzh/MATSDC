"""
阶段 2：Y 空间伪打标

加载阶段 1 输出的 (X, Y) 数据，使用 PatchTST Y-PreEncoder 编码 Y，
执行 K-Center 聚类生成伪标签，保存伪标签和中心索引供阶段 3 使用。

编码器来源:
  1. --pretrained_encoder: 加载 pretrain_y_encoder.py 输出的预训练权重
  2. 默认 (无路径): 就地用 Y 数据做自监督重建训练，训练完毕后用于聚类

用法:
  # 默认模式（就地训练编码器）
  python scripts/stage2_pseudo_labeling.py --input outputs/stage1/dummy_data.pt

  # 加载预训练编码器（跳过训练）
  python scripts/stage2_pseudo_labeling.py --input outputs/stage1/dummy_data.pt \
      --pretrained_encoder outputs/pretrain/y_encoder.pt

  # 就地训练时自定义训练参数
  python scripts/stage2_pseudo_labeling.py --input outputs/stage1/dummy_data.pt \
      --pretrain_epochs 5 --pretrain_lr 1e-3

  # Robust K-Center
  python scripts/stage2_pseudo_labeling.py --input outputs/stage1/ETTh1_data.pt \
      --pretrained_encoder outputs/pretrain/y_encoder.pt --method robust
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from configs.default import PipelineConfig
from src.layers.PatchTSTEncoder import PatchTSTFeatureExtractor
from src.clustering.kcenter import generate_pseudo_labels


# ---------------------------------------------------------------------------
# 自监督重建训练相关组件
# ---------------------------------------------------------------------------

class _ReconstructionDecoder(nn.Module):
    """线性解码器：特征向量 → 重建原始时序窗口"""

    def __init__(self, feat_dim: int, seq_len: int, c_in: int):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, seq_len * c_in),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.size(0)
        return self.decoder(features).reshape(B, self.seq_len, self.c_in)


def _pretrain_encoder(
    encoder: PatchTSTFeatureExtractor,
    Y_data: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    train_ratio: float,
    device: str,
) -> PatchTSTFeatureExtractor:
    """
    就地自监督训练编码器（重建任务）。

    训练流程:
      1. 将 Y 数据按 train_ratio 划分训练/验证集
      2. Encoder 编码 Y → 高维特征
      3. Decoder 重建回原始 Y
      4. MSE 重建损失
      5. 训练结束后只保留 encoder，丢弃 decoder

    Args:
        encoder: 待训练的 PatchTST 编码器
        Y_data: shape (N, pre_len, feature_dim)
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        train_ratio: 训练集比例
        device: 计算设备

    Returns:
        训练好的 encoder（eval 模式）
    """
    feat_dim = encoder.get_output_dim()
    seq_len = Y_data.size(1)
    c_in = Y_data.size(2)

    decoder = _ReconstructionDecoder(feat_dim, seq_len, c_in).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    criterion = nn.MSELoss()

    # 划分训练/验证
    N = Y_data.size(0)
    n_train = int(N * train_ratio)
    indices = torch.randperm(N)
    Y_shuffled = Y_data[indices]

    train_ds = TensorDataset(Y_shuffled[:n_train])
    val_ds = TensorDataset(Y_shuffled[n_train:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"  Pretraining encoder (reconstruction)...")
    print(f"    Train: {n_train}, Val: {N - n_train}, Epochs: {epochs}")

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # --- Train ---
        encoder.train()
        decoder.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch[0].to(device)
            features = encoder(x)
            reconstructed = decoder(features)
            loss = criterion(reconstructed, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # --- Validate ---
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                features = encoder(x)
                reconstructed = decoder(features)
                loss = criterion(reconstructed, x)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            marker = ' *'

        print(f"    Epoch [{epoch + 1:3d}/{epochs}]  "
              f"Train: {train_loss:.6f}  Val: {val_loss:.6f}{marker}")

    # 恢复最优权重
    if best_state is not None:
        encoder.load_state_dict(best_state)
    encoder.eval()

    print(f"  Pretraining done. Best val loss: {best_val_loss:.6f}")
    return encoder


# ---------------------------------------------------------------------------
# 编码器构建
# ---------------------------------------------------------------------------

def _build_default_encoder(cfg: PipelineConfig, device: str) -> PatchTSTFeatureExtractor:
    """构建默认 PatchTST 编码器（尚未训练）。"""
    patch_len = min(4, cfg.pre_len)
    stride = max(1, patch_len // 2)
    n_heads = max(1, cfg.y_hidden_dim // 32)

    return PatchTSTFeatureExtractor(
        c_in=cfg.feature_dim,
        seq_len=cfg.pre_len,
        patch_len=patch_len,
        stride=stride,
        d_model=cfg.y_hidden_dim,
        n_heads=n_heads,
        e_layers=1,
        d_ff=cfg.y_hidden_dim * 2,
        dropout=0.1,
        activation='gelu',
        aggregation='max',
        concat_rev_params=True,
    ).to(device)


def build_encoder(
    cfg: PipelineConfig,
    device: str,
    pretrained_path: str | None = None,
    Y_data: torch.Tensor | None = None,
    pretrain_epochs: int = 5,
    pretrain_lr: float = 1e-3,
    pretrain_batch_size: int = 64,
    pretrain_train_ratio: float = 0.8,
) -> PatchTSTFeatureExtractor:
    """
    构建 Y-PreEncoder，保证返回的一定是训练过的编码器。

    路径 A — 加载预训练权重:
        pretrained_path 指定 → 直接加载，不做训练。

    路径 B — 就地自监督训练:
        pretrained_path 为 None → 构建编码器 → 用 Y_data 做重建训练 → 返回。

    Args:
        cfg: PipelineConfig
        device: 计算设备
        pretrained_path: 预训练编码器 .pt 文件路径（可选）
        Y_data: 原始 Y 数据，路径 B 时必须提供
        pretrain_epochs: 就地训练轮数
        pretrain_lr: 就地训练学习率
        pretrain_batch_size: 就地训练批大小
        pretrain_train_ratio: 就地训练训练集比例

    Returns:
        训练好的 PatchTSTFeatureExtractor（eval 模式）
    """
    if pretrained_path and os.path.exists(pretrained_path):
        # ---- 路径 A: 加载预训练编码器 ----
        ckpt = torch.load(pretrained_path, weights_only=False)
        enc_cfg = ckpt['encoder_config']

        encoder = PatchTSTFeatureExtractor(
            c_in=enc_cfg['c_in'],
            seq_len=enc_cfg['seq_len'],
            patch_len=enc_cfg['patch_len'],
            stride=enc_cfg['stride'],
            d_model=enc_cfg['d_model'],
            n_heads=enc_cfg['n_heads'],
            e_layers=enc_cfg['e_layers'],
            d_ff=enc_cfg['d_ff'],
            activation='gelu',
            aggregation=enc_cfg['aggregation'],
            concat_rev_params=enc_cfg['concat_rev_params'],
        ).to(device)

        encoder.load_state_dict(ckpt['encoder_state_dict'])
        encoder.eval()

        print(f"  Loaded pretrained encoder from {pretrained_path}")
        print(f"    d_model={enc_cfg['d_model']}, patch_len={enc_cfg['patch_len']}, "
              f"stride={enc_cfg['stride']}, output_dim={enc_cfg['output_dim']}")

        # 同步 cfg
        cfg.y_hidden_dim = enc_cfg['output_dim']
        return encoder

    # ---- 路径 B: 就地自监督训练 ----
    assert Y_data is not None, "Y_data required when pretrained_encoder is not specified"

    encoder = _build_default_encoder(cfg, device)
    print(f"  Built new encoder, output_dim={encoder.get_output_dim()}")
    print(f"  No pretrained encoder specified → training from scratch on Y data")

    encoder = _pretrain_encoder(
        encoder=encoder,
        Y_data=Y_data,
        epochs=pretrain_epochs,
        batch_size=pretrain_batch_size,
        lr=pretrain_lr,
        train_ratio=pretrain_train_ratio,
        device=device,
    )

    return encoder


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Y-Space K-Center Pseudo Labeling')
    parser.add_argument('--input', type=str, required=True, help='阶段1 输出的 .pt 文件路径')

    # 编码器来源
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                        help='预训练编码器路径；不指定则就地训练')

    # 就地训练参数（仅 --pretrained_encoder 未指定时生效）
    parser.add_argument('--pretrain_epochs', type=int, default=5,
                        help='就地训练轮数 [5]')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3,
                        help='就地训练学习率 [1e-3]')
    parser.add_argument('--pretrain_batch_size', type=int, default=64,
                        help='就地训练批大小 [64]')
    parser.add_argument('--pretrain_train_ratio', type=float, default=0.8,
                        help='就地训练训练集比例 [0.8]')

    # 聚类参数
    parser.add_argument('--K', type=int, default=None, help='K-Center 聚类数')
    parser.add_argument('--method', type=str, default='greedy',
                        choices=['greedy', 'robust'], help='K-Center 方法')
    parser.add_argument('--outliers_fraction', type=float, default=0.05,
                        help='Robust K-Center 异常值比例')
    parser.add_argument('--output_dir', type=str, default='outputs/stage2',
                        help='输出目录')
    args = parser.parse_args()

    # 加载阶段 1 数据
    data = torch.load(args.input, weights_only=False)
    all_X = data['X']
    all_Y = data['Y']
    cfg: PipelineConfig = data['config']

    if args.K is not None:
        cfg.K = args.K
    cfg.kcenter_method = args.method

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Stage 2] Y-Space K-Center Pseudo Labeling")
    print(f"  K={cfg.K}, method={cfg.kcenter_method}")
    print(f"  Y shape: {all_Y.shape}")

    # 构建 Y-PreEncoder（已训练 / 加载预训练）
    y_encoder = build_encoder(
        cfg=cfg,
        device=device,
        pretrained_path=args.pretrained_encoder,
        Y_data=all_Y,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_train_ratio=args.pretrain_train_ratio,
    )

    print(f"  Y-Encoder output dim: {y_encoder.get_output_dim()}")

    # 生成伪标签
    kcenter_kwargs = {}
    if cfg.kcenter_method == 'robust':
        kcenter_kwargs['outliers_fraction'] = cfg.outliers_fraction

    pseudo_labels, centers_idx = generate_pseudo_labels(
        Y_data=all_Y.to(device),
        encoder=y_encoder,
        K=cfg.K,
        method=cfg.kcenter_method,
        **kcenter_kwargs,
    )

    n_outliers = int((pseudo_labels == -1).sum())
    label_dist = [int((pseudo_labels == k).sum()) for k in range(cfg.K)]
    print(f"  Pseudo label distribution: {label_dist}")
    if n_outliers > 0:
        print(f"  Outliers (label=-1): {n_outliers}")
    print(f"  Center indices: {centers_idx}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'pseudo_labels.pt')
    torch.save({
        'pseudo_labels': pseudo_labels,
        'centers_idx': centers_idx,
        'config': cfg,
        'X': all_X,
        'Y': all_Y,
    }, save_path)
    print(f"  Saved to {save_path}")


if __name__ == '__main__':
    main()
