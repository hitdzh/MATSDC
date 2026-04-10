from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "dbscan requires CUDA. CPU execution is not supported."
        )
    return torch.device('cuda')


def _validate_dbscan_args(
    eps: float,
    min_samples: int,
    chunk_size: int,
) -> None:
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    if min_samples <= 0:
        raise ValueError(f"min_samples must be > 0, got {min_samples}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")


def _chunk_ranges(total: int, chunk_size: int):
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def _coerce_features(features: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = np.asarray(features)

    if features_np.ndim != 2:
        raise ValueError(
            f"DBSCAN expects a 2D feature matrix, got shape={features_np.shape}."
        )

    return np.ascontiguousarray(features_np, dtype=np.float32)


def _count_neighbors(
    features_t: torch.Tensor,
    eps: float,
    chunk_size: int,
) -> np.ndarray:
    n_samples = features_t.size(0)
    neighbor_counts = np.empty(n_samples, dtype=np.int64)

    with torch.inference_mode():
        for start, end in _chunk_ranges(n_samples, chunk_size):
            query = features_t[start:end]
            dists = torch.cdist(query, features_t, p=2)
            neighbor_counts[start:end] = (dists <= eps).sum(dim=1).cpu().numpy()

    return neighbor_counts


def _expand_clusters(
    features_t: torch.Tensor,
    core_mask: np.ndarray,
    eps: float,
    chunk_size: int,
) -> np.ndarray:
    n_samples = features_t.size(0)
    labels = np.full(n_samples, -1, dtype=np.int64)
    expanded_core = np.zeros(n_samples, dtype=bool)
    cluster_id = 0

    core_indices = np.flatnonzero(core_mask)
    if core_indices.size == 0:
        return labels

    with torch.inference_mode():
        for seed in core_indices:
            if expanded_core[seed]:
                continue

            queue = [int(seed)]
            expanded_core[seed] = True
            labels[seed] = cluster_id
            head = 0

            while head < len(queue):
                frontier = np.asarray(queue[head:head + chunk_size], dtype=np.int64)
                head += frontier.size

                frontier_t = torch.from_numpy(frontier).to(
                    features_t.device,
                    non_blocking=True,
                )
                frontier_features = features_t.index_select(0, frontier_t)
                dists = torch.cdist(frontier_features, features_t, p=2)
                neighbor_union = (dists <= eps).any(dim=0).cpu().numpy()

                unlabeled_mask = neighbor_union & (labels == -1)
                labels[unlabeled_mask] = cluster_id

                new_core = np.flatnonzero(
                    neighbor_union & core_mask & (~expanded_core)
                )
                if new_core.size > 0:
                    expanded_core[new_core] = True
                    queue.extend(new_core.tolist())

            cluster_id += 1

    return labels


def _select_representatives(
    features_np: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    cluster_ids = np.unique(labels[labels >= 0])
    representatives = []

    for cluster_id in cluster_ids:
        member_indices = np.flatnonzero(labels == cluster_id)
        cluster_features = features_np[member_indices]
        centroid = cluster_features.mean(axis=0)
        rep_pos = np.argmin(np.linalg.norm(cluster_features - centroid, axis=1))
        representatives.append(member_indices[rep_pos])

    return np.asarray(representatives, dtype=np.int64)


def dbscan(
    features: np.ndarray | torch.Tensor,
    eps: float,
    min_samples: int,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在 GPU 上对特征执行欧氏距离 DBSCAN。

    Args:
        features: 特征矩阵，shape (N, D)
        eps: DBSCAN 半径阈值
        min_samples: 成为 core point 的最小邻域样本数（包含自身）
        chunk_size: 分块距离计算大小，用于控制显存占用

    Returns:
        labels: shape (N,)，噪声点为 -1
        representative_indices: shape (C,)，每个非噪声簇的代表样本索引
    """
    _validate_dbscan_args(eps, min_samples, chunk_size)
    device = _require_cuda()

    features_np = _coerce_features(features)
    n_samples = features_np.shape[0]
    if n_samples == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    features_t = torch.from_numpy(features_np).to(device, non_blocking=True)
    neighbor_counts = _count_neighbors(features_t, eps=eps, chunk_size=chunk_size)
    core_mask = neighbor_counts >= min_samples
    labels = _expand_clusters(
        features_t,
        core_mask=core_mask,
        eps=eps,
        chunk_size=chunk_size,
    )
    representative_indices = _select_representatives(features_np, labels)
    return labels, representative_indices


def generate_pseudo_labels(
    Y_data: torch.Tensor,
    encoder: torch.nn.Module,
    eps: float,
    min_samples: int,
    batch_size: int | None = None,
    num_workers: int = 0,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Y-PreEncoder 编码 Y，然后使用 GPU DBSCAN 生成伪标签。

    Args:
        Y_data: Y 窗口数据，shape (N, pre_len, feature_dim)
        encoder: 已放置在 CUDA 上的编码器
        eps: DBSCAN 半径阈值
        min_samples: 成为 core point 的最小邻域样本数
        batch_size: 编码批大小
        num_workers: DataLoader 工作进程数
        chunk_size: GPU 分块邻域搜索大小

    Returns:
        labels: shape (N,)，噪声点为 -1
        representative_indices: 每个非噪声簇的代表样本索引
    """
    _validate_dbscan_args(eps, min_samples, chunk_size)
    device = next(encoder.parameters()).device
    if device.type != 'cuda':
        raise RuntimeError(
            "generate_pseudo_labels in dbscan.py requires the encoder to be on CUDA."
        )

    encoder.eval()
    dataset = TensorDataset(Y_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or len(dataset),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feature_chunks = []
    with torch.inference_mode():
        for batch in loader:
            batch_y = batch[0].to(device, non_blocking=True)
            features = encoder(batch_y)
            feature_chunks.append(features.detach().cpu())

    features = torch.cat(feature_chunks, dim=0)
    return dbscan(
        features=features,
        eps=eps,
        min_samples=min_samples,
        chunk_size=chunk_size,
    )


__all__ = ['dbscan', 'generate_pseudo_labels']
