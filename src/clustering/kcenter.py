"""
K-Center 聚类：
  1. greedy_kcenter  — 贪心 K-Center (默认)，经典的远点采样贪心策略
  2. robust_kcenter  — Robust K-Center, 在贪心基础上支持异常值排除
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist


def greedy_kcenter(
    features: np.ndarray,
    K: int,
    distance_metric: str = 'euclidean',
) -> tuple[np.ndarray, np.ndarray]:
    """
    贪心 K-Center 聚类

    Args:
        features: 特征矩阵, shape (N, D)
        K: 中心点数量
        distance_metric: 距离度量类型，传给 scipy.spatial.distance.cdist

    Returns:
        centers_idx: K 个中心点在 features 中的索引, shape (K,)
        labels: 每个样本的最近中心 ID, shape (N,)
    """
    N = features.shape[0]
    K = min(K, N)

    # 随机选第一个中心
    centers_idx = [np.random.randint(N)]

    # 初始化距离数组：每个点到最近中心的距离
    dist_to_center = cdist(features, features[centers_idx[-1:]], metric=distance_metric).squeeze(axis=1)

    # 贪心迭代选中心
    for _ in range(1, K):
        # 选距离现有中心最远的点作为新中心
        new_center_idx = np.argmax(dist_to_center)
        centers_idx.append(new_center_idx)

        # 更新距离数组：取与最近中心的距离
        new_dists = cdist(features, features[new_center_idx:new_center_idx + 1], metric=distance_metric).squeeze(axis=1)
        dist_to_center = np.minimum(dist_to_center, new_dists)

    centers_idx = np.array(centers_idx)

    # 根据最终中心点为每个样本分配标签
    dist_matrix = cdist(features, features[centers_idx], metric=distance_metric)  # (N, K)
    labels = np.argmin(dist_matrix, axis=1)  # (N,)

    return centers_idx, labels


def robust_kcenter(
    features: np.ndarray,
    K: int,
    outliers_fraction: float = 0.05,
    distance_metric: str = 'euclidean',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust K-Center 聚类

    Args:
        features: 特征矩阵, shape (N, D)
        K: 中心点数量
        outliers_fraction: 异常值比例 (0.0 ~ 1.0)
        distance_metric: 距离度量类型

    Returns:
        centers_idx: K 个中心点索引, shape (K,)
        labels: 每个样本的簇标签（异常值标记为 -1), shape (N,)
        outlier_mask: 布尔数组, True 表示异常值, shape (N,)
    """
    N = features.shape[0]

    # 执行贪心 K-Center
    centers_idx, labels = greedy_kcenter(features, K, distance_metric)

    # 计算每个样本到其簇中心的距离
    center_features = features[centers_idx]  # (K, D)

    dist_to_center = np.array([
        np.linalg.norm(features[i] - center_features[labels[i]])
        for i in range(N)
    ])

    # 标记异常值
    # 选取距离最大的 n_outliers 个样本作为异常值
    n_outliers = int(np.ceil(N * outliers_fraction))
    n_outliers = min(n_outliers, N - K)  # 保证至少 K 个正常样本

    # 按距离降序排列，选取最远的 n_outliers 个
    outlier_indices = np.argsort(dist_to_center)[::-1][:n_outliers]
    outlier_mask = np.zeros(N, dtype=bool)
    outlier_mask[outlier_indices] = True

    # 对非异常样本重新分配标签
    labels = labels.copy()
    labels[outlier_mask] = -1  # 异常值标记为 -1

    # 重新计算非异常样本到各中心的最近距离
    normal_mask = ~outlier_mask
    if normal_mask.sum() > 0:
        dist_matrix = cdist(features[normal_mask], features[centers_idx], metric=distance_metric)
        labels[normal_mask] = np.argmin(dist_matrix, axis=1)

    return centers_idx, labels, outlier_mask


def generate_pseudo_labels(
    Y_data: torch.Tensor,
    encoder: torch.nn.Module,
    K: int,
    method: str = 'greedy',
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Y-PreEncoder 编码 Y, 然后通过 K-Center 聚类生成伪标签
    """
    # 编码所有 Y 到高维特征空间
    encoder.eval()
    with torch.no_grad():
        features = encoder(Y_data)  # (N, hidden_dim)
    features_np = features.cpu().numpy()

    # 选择 K-Center 方法
    if method == 'robust':
        centers_idx, labels, outlier_mask = robust_kcenter(features_np, K, **kwargs)
    elif method == 'greedy':
        centers_idx, labels = greedy_kcenter(features_np, K, **kwargs)
    else:
        raise ValueError(f"Unknown kcenter method: {method}, expected 'greedy' or 'robust'")

    return labels, centers_idx
