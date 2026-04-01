"""
谱聚类与样本提取

流程:
  1. 特征 L2 归一化
  2. 计算 RBF 亲和度矩阵
  3. 执行谱聚类得到簇分配
  4. 从每个簇中选取 prototype (典型样本)
"""

import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist


class SpectralSelector:
    """
    谱聚类 + 样本提取

    Args:
        n_clusters: 谱聚类簇数
        n_prototypes: 最终提取的典型样本数量 N
        sigma: RBF 核带宽参数；若为 None 则自动估计（使用中位数距离）
    """

    def __init__(self, n_clusters: int = 4, n_prototypes: int = 4, sigma: float | None = None):
        self.n_clusters = n_clusters
        self.n_prototypes = n_prototypes
        self.sigma = sigma

    def _compute_affinity(self, features: np.ndarray) -> np.ndarray:
        """
        计算 RBF (高斯) 亲和度矩阵
        """
        # 计算欧氏距离矩阵
        dist_matrix = cdist(features, features, metric='euclidean')  # (N, N)
        dist_sq = dist_matrix ** 2

        # 确定 sigma
        if self.sigma is None:
            upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            sigma = np.median(upper_tri[upper_tri > 0])
            sigma = max(sigma, 1e-8)
        else:
            sigma = self.sigma

        # RBF 核
        affinity = np.exp(-dist_sq / (2 * sigma ** 2))
        return affinity

    def _select_prototypes(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> list[int]:
        """
        从谱聚类结果中选取 N 个典型样本 (prototype)。
        """
        unique_labels = np.unique(labels)
        n_clusters_actual = len(unique_labels)

        # 计算每个簇内的最近质心样本
        cluster_prototypes = {}  # cluster_id -> index of closest-to-centroid sample
        cluster_sizes = {}

        for c in unique_labels:
            mask = labels == c
            cluster_features = features[mask]
            cluster_indices = np.where(mask)[0]
            cluster_sizes[c] = mask.sum()

            # 计算簇内均值（虚拟质心）
            centroid = cluster_features.mean(axis=0)  # (D,)

            # 计算每个簇内样本到质心的欧氏距离
            dists_to_centroid = np.linalg.norm(cluster_features - centroid, axis=1)

            # 选取距离最小的样本索引
            closest_idx_in_cluster = np.argmin(dists_to_centroid)
            cluster_prototypes[c] = cluster_indices[closest_idx_in_cluster]

        # 分配 n_prototypes 个名额
        if self.n_prototypes >= n_clusters_actual:
            selected = [cluster_prototypes[c] for c in unique_labels]

            remaining = self.n_prototypes - n_clusters_actual
            if remaining > 0:
                total_size = sum(cluster_sizes.values())
                extra_per_cluster = {}
                allocated = 0
                for c in unique_labels:
                    proportion = cluster_sizes[c] / total_size
                    extra = int(np.floor(proportion * remaining))
                    extra_per_cluster[c] = extra
                    allocated += extra

                # 处理取整误差，将剩余名额分给最大的簇
                for c in sorted(unique_labels, key=lambda x: cluster_sizes[x], reverse=True):
                    if allocated >= remaining:
                        break
                    extra_per_cluster[c] += 1
                    allocated += 1

                # 为每个簇从次近的样本中补充 prototype
                for c in unique_labels:
                    extra = extra_per_cluster.get(c, 0)
                    if extra <= 0:
                        continue
                    mask = labels == c
                    cluster_indices = np.where(mask)[0]
                    cluster_features = features[mask]
                    centroid = cluster_features.mean(axis=0)
                    dists = np.linalg.norm(cluster_features - centroid, axis=1)

                    # 按距离排序，跳过已选的第一个
                    sorted_order = np.argsort(dists)
                    count = 0
                    for rank in sorted_order:
                        candidate = cluster_indices[rank]
                        if candidate not in selected:
                            selected.append(candidate)
                            count += 1
                            if count >= extra:
                                break
        else:
            total_size = sum(cluster_sizes.values())
            weights = np.array([cluster_sizes[c] / total_size for c in unique_labels])
            selected_clusters = np.random.choice(
                unique_labels, size=self.n_prototypes, replace=False, p=weights
            )
            selected = [cluster_prototypes[c] for c in selected_clusters]

        return selected

    def cluster_and_select(
        self,
        features: torch.Tensor | np.ndarray,
        X_raw: torch.Tensor | np.ndarray,
        Y_raw: torch.Tensor | np.ndarray,
    ) -> dict:
        """
        执行谱聚类 + 样本提取
        """
        # 转为 numpy
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = np.asarray(features)

        if isinstance(X_raw, torch.Tensor):
            X_np = X_raw.detach().cpu().numpy()
        else:
            X_np = np.asarray(X_raw)

        if isinstance(Y_raw, torch.Tensor):
            Y_np = Y_raw.detach().cpu().numpy()
        else:
            Y_np = np.asarray(Y_raw)

        # L2 归一化特征
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features_normed = features_np / norms  # (N, D)

        # 计算亲和度矩阵
        affinity = self._compute_affinity(features_normed)

        # 谱聚类
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='kmeans',
        )
        cluster_labels = sc.fit_predict(affinity)  # (N,)

        print(f"  谱聚类完成: {self.n_clusters} 个簇, "
              f"簇大小分布: {[int((cluster_labels == i).sum()) for i in range(self.n_clusters)]}")

        # 选取典型样本
        prototype_indices = self._select_prototypes(features_normed, cluster_labels)

        # 收集结果
        result = {
            'X': X_np[prototype_indices],                            # (N_proto, seq_len, feature_dim)
            'Y': Y_np[prototype_indices],                            # (N_proto, pre_len, feature_dim)
            'features': features_np[prototype_indices],              # (N_proto, D)
            'cluster_ids': cluster_labels[prototype_indices],        # (N_proto,)
            'indices': np.array(prototype_indices),                  # (N_proto,)
        }

        return result
