"""
K-Center 聚类：
  1. kcenter  — 贪心 K-Center，经典的远点采样贪心策略
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset


def _normalize_method_name(method: str) -> str:
    if method == 'greedy':
        return 'kcenter'
    return method


def kcenter(
    features: np.ndarray,
    K: int,
    distance_metric: str = 'euclidean',
) -> tuple[np.ndarray, np.ndarray]:
    """
    贪心 K-Center 聚类。

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

    centers_idx = [np.random.randint(N)]
    dist_to_center = cdist(
        features,
        features[centers_idx[-1:]],
        metric=distance_metric,
    ).squeeze(axis=1)

    for _ in range(1, K):
        new_center_idx = np.argmax(dist_to_center)
        centers_idx.append(new_center_idx)

        new_dists = cdist(
            features,
            features[new_center_idx:new_center_idx + 1],
            metric=distance_metric,
        ).squeeze(axis=1)
        dist_to_center = np.minimum(dist_to_center, new_dists)

    centers_idx = np.array(centers_idx)
    dist_matrix = cdist(features, features[centers_idx], metric=distance_metric)
    labels = np.argmin(dist_matrix, axis=1)

    return centers_idx, labels


def generate_pseudo_labels(
    Y_data: torch.Tensor,
    encoder: torch.nn.Module,
    K: int,
    method: str = 'kcenter',
    batch_size: int | None = None,
    num_workers: int = 0,
    distance_metric: str = 'euclidean',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Y-PreEncoder 编码 Y，然后通过 K-Center 聚类生成伪标签。
    """
    method = _normalize_method_name(method)
    if method != 'kcenter':
        raise ValueError(
            f"Unknown kcenter method: {method}, expected 'kcenter'"
        )

    encoder.eval()
    device = next(encoder.parameters()).device
    dataset = TensorDataset(Y_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or len(dataset),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )

    feature_chunks = []
    with torch.no_grad():
        for batch in loader:
            batch_y = batch[0].to(device, non_blocking=device.type == 'cuda')
            features = encoder(batch_y)
            feature_chunks.append(features.cpu())
    features_np = torch.cat(feature_chunks, dim=0).numpy()

    centers_idx, labels = kcenter(
        features_np,
        K,
        distance_metric=distance_metric,
    )
    return labels, centers_idx
