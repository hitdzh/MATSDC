"""
阶段 1：滑动窗口数据构建模块

将原始时间序列通过滑动窗口机制切分为:
  - X: 输入特征窗口, shape (seq_len, feature_dim)
  - Y: 预测目标窗口, shape (pre_len, feature_dim)

时间对齐关系: X[i] = time_series[i : i+seq_len]
              Y[i] = time_series[i+seq_len : i+seq_len+pre_len]
即 Y 紧跟在 X 之后，保证严格的时间顺序对应。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    滑动窗口数据集，将 1D/多变量时序数据切分为 (X, Y) 对。

    Args:
        time_series: 原始时序数据, shape (T, feature_dim) 或 (T,)
        seq_len: 输入序列 X 的窗口长度
        pre_len: 预测目标 Y 的窗口长度
    """

    def __init__(self, time_series: np.ndarray, seq_len: int, pre_len: int):
        super().__init__()

        # 确保 time_series 是 2D: (T, feature_dim)
        if time_series.ndim == 1:
            time_series = time_series[:, np.newaxis]
        assert time_series.ndim == 2, f"Expected 1D or 2D array, got {time_series.ndim}D"

        self.time_series = time_series.astype(np.float32)
        self.seq_len = seq_len
        self.pre_len = pre_len

        T = time_series.shape[0]
        assert T >= seq_len + pre_len, (
            f"时间序列长度 T={T} 必须大于等于 seq_len + pre_len = {seq_len + pre_len}"
        )

        # 样本总数: T - seq_len - pre_len + 1
        self.n_samples = T - seq_len - pre_len + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回第 idx 个 (X, Y) 对。

        X: time_series[idx : idx + seq_len]           → shape (seq_len, feature_dim)
        Y: time_series[idx + seq_len : idx + seq_len + pre_len] → shape (pre_len, feature_dim)
        """
        x_start = idx
        x_end = idx + self.seq_len
        y_end = x_end + self.pre_len

        X = self.time_series[x_start:x_end]    # (seq_len, feature_dim)
        Y = self.time_series[x_end:y_end]      # (pre_len, feature_dim)

        return torch.from_numpy(X), torch.from_numpy(Y)
