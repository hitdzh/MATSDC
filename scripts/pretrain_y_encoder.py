"""
PatchTST Y-PreEncoder 自监督预训练脚本

用法:
  # 使用真实数据集
  python scripts/pretrain_y_encoder.py --dataset ETTh1 --seq_len 96 --pre_len 24 --epochs 20

  # 加载阶段1已有数据
  python scripts/pretrain_y_encoder.py --input outputs/stage1/ETTh1_data.pt --epochs 20

  # stage2 中加载预训练编码器
  python scripts/stage2_pseudo_labeling.py --input ... --pretrained_encoder outputs/pretrain/y_encoder.pt
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


class ReconstructionDecoder(nn.Module):
    """
    线性解码器：将编码器输出的特征向量重建为原始时序窗口。

    结构: feature_vector → Linear → ReLU → Linear → reshape → (seq_len, c_in)
    """

    def __init__(self, feat_dim: int, seq_len: int, c_in: int, hidden_dim: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in

        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, seq_len * c_in),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 编码器输出, shape (B, feat_dim)
        Returns:
            reconstructed: 重建的时序窗口, shape (B, seq_len, c_in)
        """
        B = features.size(0)
        out = self.decoder(features)
        return out.reshape(B, self.seq_len, self.c_in)


class AutoencoderTrainer:
    """自编码器训练器"""

    def __init__(self, encoder, decoder, lr=1e-3, device='cpu'):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr,
        )
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.encoder.train()
        self.decoder.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x = batch[0].to(self.device)  # (B, seq_len, c_in)

            # 前向: encode → reconstruct
            features = self.encoder(x)             # (B, feat_dim)
            reconstructed = self.decoder(features)  # (B, seq_len, c_in)

            # 重建损失
            loss = self.criterion(reconstructed, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x = batch[0].to(self.device)
            features = self.encoder(x)
            reconstructed = self.decoder(features)
            loss = self.criterion(reconstructed, x)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Pretrain PatchTST Y-Encoder (Self-Supervised Reconstruction)')
    parser.add_argument('--input', type=str, default=None,
                        help='阶段1输出的 .pt 文件（优先使用）')
    parser.add_argument('--dataset', type=str, default='dummy',
                        choices=['dummy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                                 'electricity', 'traffic', 'weather'],
                        help='数据集名称（仅当 --input 未指定时使用）')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='训练窗口长度（默认用 pre_len，因为编码器编码的是 Y 窗口）')
    parser.add_argument('--pre_len', type=int, default=10,
                        help='Y 窗口长度（编码器的实际输入 seq_len）')
    parser.add_argument('--feature_dim', type=int, default=3, help='仅 dummy')
    parser.add_argument('--T', type=int, default=1000, help='仅 dummy')
    # PatchTST 结构参数
    parser.add_argument('--patch_len', type=int, default=None,
                        help='Patch 长度（默认自动适配 pre_len）')
    parser.add_argument('--stride', type=int, default=None,
                        help='步长（默认自动适配 patch_len）')
    parser.add_argument('--d_model', type=int, default=64, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=None, help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=2, help='Transformer 层数')
    parser.add_argument('--d_ff', type=int, default=128, help='FFN 维度')
    parser.add_argument('--aggregation', type=str, default='max',
                        choices=['max', 'flatten'], help='聚合策略')
    parser.add_argument('--concat_rev_params', action='store_true', default=True,
                        help='拼接 RevIN 参数')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--output_dir', type=str, default='outputs/pretrain',
                        help='输出目录')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ==================== 加载数据 ====================
    # 编码器输入长度 = Y 窗口长度 (pre_len)
    enc_seq_len = args.pre_len

    if args.input and os.path.exists(args.input):
        # 从阶段1输出加载
        print(f"[Pretrain] Loading data from {args.input}")
        data = torch.load(args.input, weights_only=False)
        all_X = data['X']  # (N, seq_len, feature_dim) — 这里是完整的 X 窗口
        all_Y = data['Y']  # (N, pre_len, feature_dim) — Y 窗口
        cfg: PipelineConfig = data['config']
        feature_dim = cfg.feature_dim

        # 训练目标是 Y 窗口（编码器要学习 Y 的特征表示）
        train_data = all_Y
        print(f"  Using Y windows from stage1 output: {train_data.shape}")
    else:
        # 从头加载数据
        print(f"[Pretrain] Loading dataset: {args.dataset}")
        if args.dataset == 'dummy':
            np.random.seed(42)
            time_series = np.random.randn(args.T, args.feature_dim).astype(np.float32)
            feature_dim = args.feature_dim

            # 用滑动窗口切出所有长度为 pre_len 的子序列作为训练数据
            N = len(time_series) - enc_seq_len + 1
            windows = []
            for i in range(N):
                windows.append(time_series[i:i + enc_seq_len])
            train_data = torch.from_numpy(np.stack(windows))
        else:
            ds = TimeSeriesForecastDataset(
                dataset_name=args.dataset,
                split='train',
                seq_len=enc_seq_len,
                pre_len=1,  # 只需要 X 部分，pre_len=1 满足接口要求
            )
            feature_dim = ds.feature_dim
            # 提取所有 X 窗口
            loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
            batch = next(iter(loader))
            train_data = batch[0]  # (N, enc_seq_len, feature_dim)

        print(f"  Train data shape: {train_data.shape}")

    # ==================== 自动适配 patch 参数 ====================
    patch_len = args.patch_len if args.patch_len else min(4, enc_seq_len)
    stride = args.stride if args.stride else max(1, patch_len // 2)
    n_heads = args.n_heads if args.n_heads else max(1, args.d_model // 32)

    # 确保能产生至少 1 个 patch
    num_patches = (enc_seq_len - patch_len) // stride + 1
    if num_patches < 1:
        patch_len = enc_seq_len
        stride = 1
        num_patches = 1
        print(f"  [Auto-adjust] patch_len={patch_len}, stride={stride} (too short for default)")

    print(f"  PatchTST config: patch_len={patch_len}, stride={stride}, "
          f"num_patches={num_patches}, d_model={args.d_model}, n_heads={n_heads}")

    # ==================== 划分训练/验证 ====================
    N = train_data.size(0)
    n_train = int(N * args.train_ratio)
    indices = torch.randperm(N)
    train_data_shuffled = train_data[indices]

    train_set = TensorDataset(train_data_shuffled[:n_train])
    val_set = TensorDataset(train_data_shuffled[n_train:])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {n_train}, Val: {N - n_train}")

    # ==================== 构建模型 ====================
    encoder = PatchTSTFeatureExtractor(
        c_in=feature_dim,
        seq_len=enc_seq_len,
        patch_len=patch_len,
        stride=stride,
        d_model=args.d_model,
        n_heads=n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=0.1,
        activation='gelu',
        aggregation=args.aggregation,
        concat_rev_params=args.concat_rev_params,
    ).to(device)

    feat_dim = encoder.get_output_dim()

    decoder = ReconstructionDecoder(
        feat_dim=feat_dim,
        seq_len=enc_seq_len,
        c_in=feature_dim,
        hidden_dim=feat_dim * 2,
    ).to(device)

    print(f"  Encoder output dim: {feat_dim}")
    print(f"  Total params: {sum(p.numel() for p in encoder.parameters()):,}")

    # ==================== 训练 ====================
    trainer = AutoencoderTrainer(encoder, decoder, lr=args.lr, device=device)

    print(f"\n  Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = ' *'
            # 保存最优模型
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}

        print(f"  Epoch [{epoch + 1:3d}/{args.epochs}]  "
              f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}{marker}")

    # ==================== 保存 ====================
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'y_encoder.pt')

    torch.save({
        'encoder_state_dict': best_state,
        'encoder_config': {
            'c_in': feature_dim,
            'seq_len': enc_seq_len,
            'patch_len': patch_len,
            'stride': stride,
            'd_model': args.d_model,
            'n_heads': n_heads,
            'e_layers': args.e_layers,
            'd_ff': args.d_ff,
            'aggregation': args.aggregation,
            'concat_rev_params': args.concat_rev_params,
            'output_dim': feat_dim,
        },
        'training_info': {
            'epochs': args.epochs,
            'best_val_loss': best_val_loss,
            'dataset': args.dataset,
        },
    }, save_path)

    print(f"\n  Best val loss: {best_val_loss:.6f}")
    print(f"  Encoder saved to {save_path}")
    print(f"  Usage in stage2: --pretrained_encoder {save_path}")


if __name__ == '__main__':
    main()
