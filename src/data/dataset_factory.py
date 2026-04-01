"""
数据集加载与预处理模块

支持加载 datasets/ 目录下的时序预测基准数据集:
  - ETTh1, ETTh2 (每小时采样, 7 特征)
  - ETTm1, ETTm2 (每15分钟采样, 7 特征)
  - electricity  (每小时采样, 321 特征)
  - traffic      (每小时采样, 862 特征)
  - weather      (每10分钟采样, 21 特征)

遵循时序预测的标准数据划分方式:
  - 训练集 (train): 前 70%
  - 验证集 (val):   中间 10%
  - 测试集 (test):  最后 20%

提供 Z-Score 标准化（仅用训练集的统计量）。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

_DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'datasets')


class TimeSeriesForecastDataset(Dataset):
    """
    时序预测数据集。

    Args:
        dataset_name: 数据集名称 (如 'ETTh1', 'electricity', 'weather' 等)
        split: 数据划分 ('train', 'val', 'test')
        seq_len: 输入序列长度
        pre_len: 预测目标长度
        scale: 是否进行 Z-Score 标准化
        data_dir: 数据集目录路径，默认为项目根目录下的 datasets/
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = 'train',
        seq_len: int = 336,
        pre_len: int = 96,
        scale: bool = True,
        data_dir: str | None = None,
    ):
        super().__init__()
        assert split in ('train', 'val', 'test'), f"split must be 'train', 'val' or 'test', got '{split}'"

        self.dataset_name = dataset_name
        self.split = split
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.scale = scale

        # 加载原始数据
        data_path = os.path.join(data_dir or _DATASETS_DIR, f'{dataset_name}.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")
        raw_data = self._load_csv(data_path)

        # 标准化
        self.mean: float | None = None
        self.std: float | None = None
        if scale:
            self.mean, self.std = self._compute_train_stats(raw_data)
            data = (raw_data - self.mean) / self.std
        else:
            data = raw_data

        # 按比例划分
        self.data = self._split_data(data)

    def _load_csv(self, path: str) -> np.ndarray:
        """
        加载 CSV 文件，跳过日期列，返回浮点数矩阵。

        Returns:
            data: shape (T, feature_dim)，其中 T 为时间步数
        """
        import pandas as pd
        df = pd.read_csv(path)
        # 去除第一列（日期时间戳）
        if df.columns[0].lower() in ('date', 'datetime', 'timestamp', 'time'):
            df = df.drop(columns=[df.columns[0]])
        return df.values.astype(np.float32)

    def _compute_train_stats(self, raw_data: np.ndarray) -> tuple[float, float]:
        """
        用训练集部分计算全局均值和标准差
        """
        n_total = raw_data.shape[0]
        n_train = int(n_total * 0.7)
        train_data = raw_data[:n_train]
        mean = train_data.mean()
        std = train_data.std()
        std = max(std, 1e-8)  # 避免除零
        return mean, std

    def _split_data(self, data: np.ndarray) -> np.ndarray:
        """
        按标准比例划分数据集:
          train: 前 70%
          val:   中间 10% (70% ~ 80%)
          test:  最后 20% (80% ~ 100%)
        """
        n_total = data.shape[0]
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.1)

        if self.split == 'train':
            return data[:n_train]
        elif self.split == 'val':
            # 向前多取 seq_len ，使第一个 Y 从 n_train 位置开始
            start = max(0, n_train - self.seq_len)
            return data[start:n_train + n_val]
        else:  # test
            start = max(0, n_train + n_val - self.seq_len)
            return data[start:]

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len - self.pre_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (X, Y) 对。

        Returns:
            X: shape (seq_len, feature_dim)
            Y: shape (pre_len, feature_dim)
        """
        s = idx
        e = s + self.seq_len
        X = self.data[s:e]
        Y = self.data[e:e + self.pre_len]
        return torch.from_numpy(X), torch.from_numpy(Y)

    @property
    def feature_dim(self) -> int:
        return self.data.shape[1]

    def get_full_series(self) -> np.ndarray:
        """返回当前 split 的完整时序数据（已标准化），shape (T, feature_dim)"""
        return self.data.copy()
