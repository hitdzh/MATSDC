"""
阶段 1：数据准备脚本

加载原始时序数据（CSV 或随机生成），构建滑动窗��数据集，
并将切分后的 (X, Y) 对保存为 .pt 文件，供后续阶段使用。

用法:
  python scripts/stage1_data_preparation.py --dataset dummy
  python scripts/stage1_data_preparation.py --dataset ETTh1
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.default import PipelineConfig
from src.data.sliding_window import SlidingWindowDataset
from src.data.dataset_factory import TimeSeriesForecastDataset


def prepare_from_raw(time_series: np.ndarray, cfg: PipelineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从原始时序数组构建滑动窗口 (X, Y) 对。

    Args:
        time_series: shape (T, feature_dim)
        cfg: PipelineConfig

    Returns:
        all_X: (N, seq_len, feature_dim)
        all_Y: (N, pre_len, feature_dim)
    """
    dataset = SlidingWindowDataset(time_series, cfg.seq_len, cfg.pre_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    all_X, all_Y = [], []
    for X_batch, Y_batch in dataloader:
        all_X.append(X_batch)
        all_Y.append(Y_batch)

    all_X = torch.cat(all_X, dim=0)
    all_Y = torch.cat(all_Y, dim=0)
    return all_X, all_Y


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Data Preparation')
    parser.add_argument('--dataset', type=str, default='dummy',
                        choices=['dummy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                                 'electricity', 'traffic', 'weather'],
                        help='数据集名称 (dummy=随机生成)')
    parser.add_argument('--seq_len', type=int, default=None, help='输入序列长度')
    parser.add_argument('--pre_len', type=int, default=None, help='预测目标长度')
    parser.add_argument('--feature_dim', type=int, default=3, help='特征维度 (仅 dummy)')
    parser.add_argument('--T', type=int, default=1000, help='时序长度 (仅 dummy)')
    parser.add_argument('--output_dir', type=str, default='outputs/stage1',
                        help='输出目录')
    args = parser.parse_args()

    # 加载配置
    cfg = PipelineConfig()
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.pre_len is not None:
        cfg.pre_len = args.pre_len

    print(f"[Stage 1] Data Preparation — dataset={args.dataset}")
    print(f"  seq_len={cfg.seq_len}, pre_len={cfg.pre_len}")

    if args.dataset == 'dummy':
        np.random.seed(42)
        time_series = np.random.randn(args.T, args.feature_dim).astype(np.float32)
        cfg.feature_dim = args.feature_dim

        all_X, all_Y = prepare_from_raw(time_series, cfg)
    else:
        # 使用真实数据集（训练集 split）
        ds = TimeSeriesForecastDataset(
            dataset_name=args.dataset,
            split='train',
            seq_len=cfg.seq_len,
            pre_len=cfg.pre_len,
        )
        cfg.feature_dim = ds.feature_dim
        all_X, all_Y = prepare_from_raw(ds.get_full_series(), cfg)

    print(f"  X shape: {all_X.shape}")
    print(f"  Y shape: {all_Y.shape}")
    print(f"  Total samples: {all_X.shape[0]}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.dataset}_data.pt')
    torch.save({
        'X': all_X,
        'Y': all_Y,
        'config': cfg,
    }, save_path)
    print(f"  Saved to {save_path}")


if __name__ == '__main__':
    main()
