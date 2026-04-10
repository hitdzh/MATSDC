"""
谱聚类与样本提取

  1. 特征 L2 归一化
  2. 计算 RBF 亲和度矩阵
  3. 计算归一化拉普拉斯矩阵
  4. 特征分解得到特征向量
  5. K-Means++ 聚类特征向量
  6. 从每个簇中选取 prototype 
"""

import numpy as np
import torch


class SpectralSelector:
    """
    谱聚类 + 样本提取

    Args:
        n_clusters: 谱聚类簇数
        n_prototypes: 最终提取的典型样本数量 N
        sigma: RBF 核带宽参数(default = 0.5)
    """

    def __init__(
        self,
        n_clusters: int = 4,
        n_prototypes: int = 4,
        sigma: float | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_prototypes = n_prototypes
        self.sigma = sigma


    def _estimate_sigma(self, features_t: torch.Tensor) -> float:
        N = features_t.shape[0]
        dist = torch.cdist(features_t, features_t, p=2)  # (N, N)
        upper_idx = torch.triu_indices(N, N, offset=1, device=features_t.device)
        upper = dist[upper_idx[0], upper_idx[1]]  # 1D
        sigma = torch.median(upper[upper > 0]).item()
        del dist, upper
        return max(sigma, 1e-5)
    
    # 亲和度矩阵计算
    def _compute_affinity(
        self,
        features_t: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        # L2距离
        dist = torch.cdist(features_t, features_t, p=2)  # (N, N)

        dist_sq = dist.pow(2)
        del dist

        # RBF核: exp(-||x-y||^2 / (2σ²))
        inv_2sigma_sq = -0.5 / (sigma ** 2)
        affinity = torch.exp(dist_sq * inv_2sigma_sq)
        del dist_sq

        # RBF核对角线应为1.0
        affinity.fill_diagonal_(1.0)
        return affinity

    # 归一化拉普拉斯矩阵
    def _compute_laplacian(self, affinity: torch.Tensor) -> torch.Tensor:
        """
        L = I - D^(-1/2) * A * D^(-1/2)
        """
        N = affinity.shape[0]
        d = affinity.sum(dim=1).clamp_(min=1e-10)  # (N,)
        d_inv_sqrt = d.pow(-0.5)  # (N,)

        # D^(-1/2) * A * D^(-1/2)
        L = affinity * d_inv_sqrt.unsqueeze(1)
        L = L * d_inv_sqrt.unsqueeze(0)

        # L_sym = I - (D^(-1/2) * A * D^(-1/2))
        L.neg_()  # L = -L
        L.fill_diagonal_(0.0)
        L += torch.eye(N, device=affinity.device, dtype=affinity.dtype)  # 加单位阵

        return L

    # 特征分解
    def _eigendecompose_laplacian(
        self,
        L: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Returns:
            (N, k) 行归一化的top-k特征向量
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # 取 除特征值0对应特征向量外 最小的k个
        U = eigenvectors[:, 1:k + 1]  # (N, k)

        # 将绝对值最大的位置置为正
        max_idx = U.abs().argmax(dim=0)  # (k,)
        signs = torch.sign(U[max_idx, torch.arange(k, device=U.device)])
        signs[signs == 0] = 1
        U = U * signs

        # 归一化
        row_norms = U.norm(dim=1, keepdim=True).clamp_(min=1e-12)
        U = U / row_norms
        return U
    

    def _kmeans(
        self,
        X: torch.Tensor,
        k: int,
        n_iter: int = 50,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> np.ndarray:
        """
        K-Means++聚类。

        Args:
            X: (N, D) 特征向量
            k: 簇数
            n_iter: 最大迭代轮数
            tol: 收敛阈值
            seed: 随机种子

        Returns:
            (N,) 簇标签
        """
        N, D = X.shape
        device = X.device
        k = min(k, N)

        torch.manual_seed(seed)

        # K-Means++ 初始化
        centroids = torch.empty(k, D, device=device, dtype=X.dtype)

        # 随机选一个点
        centroids[0] = X[torch.randperm(N, device=device)[0]]

        # 按到最近中心距离的平方比例抽样
        for i in range(1, k):
            dists = torch.cdist(X, centroids[:i])  # (N, i)
            min_sq = dists.min(dim=1)[0].pow(2)  # (N,)

            # 构造累积分布
            cumsum = min_sq.cumsum(dim=0)
            threshold = torch.rand(1, device=device) * cumsum[-1]
            idx = (cumsum > threshold).nonzero(as_tuple=True)[0][0].item()
            centroids[i] = X[idx]

        labels = torch.empty(N, dtype=torch.long, device=device)

        for _ in range(n_iter):
            dists = torch.cdist(X, centroids)  # (N, k)
            new_labels = dists.argmin(dim=1)  # (N,)

            # 处理空簇: 用randperm选一个随机点填补
            counts = new_labels.bincount(minlength=k)
            empty = (counts == 0).nonzero(as_tuple=True)[0]
            if len(empty) > 0:
                fill_idx = torch.randperm(N, device=device)[:len(empty)]
                for c, fi in zip(empty.tolist(), fill_idx.tolist()):
                    new_labels[fi] = c

            # 批量更新centroids
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = new_labels == c
                if mask.sum() > 0:
                    new_centroids[c] = X[mask].mean(dim=0)

            change = (new_centroids - centroids).norm() / (centroids.norm() + 1e-8)
            centroids = new_centroids

            if change < tol:
                return new_labels.cpu().numpy()

            labels = new_labels

        return labels.cpu().numpy()


    def _spectral_cluster(
        self,
        features_normed: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        features_t = torch.from_numpy(features_normed).to(
            device, non_blocking=True
        )

        # 估算sigma
        if self.sigma is None:
            sigma = self._estimate_sigma(features_t)
        else:
            sigma = self.sigma

        # RBF亲和度矩阵
        affinity = self._compute_affinity(features_t, sigma)
        del features_t
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

        # 归一化拉普拉斯
        L = self._compute_laplacian(affinity)
        del affinity
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

        # 特征分解
        U = self._eigendecompose_laplacian(L, self.n_clusters)
        del L
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

        labels = self._kmeans(U, self.n_clusters, seed=42)
        del U
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

        return labels


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

        cluster_prototypes = {}
        cluster_sizes = {}

        for c in unique_labels:
            mask = labels == c
            cluster_features = features[mask]
            cluster_indices = np.where(mask)[0]
            cluster_sizes[c] = mask.sum()

            centroid = cluster_features.mean(axis=0)
            dists_to_centroid = np.linalg.norm(
                cluster_features - centroid, axis=1
            )
            closest_idx_in_cluster = np.argmin(dists_to_centroid)
            cluster_prototypes[c] = cluster_indices[closest_idx_in_cluster]

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

                for c in sorted(
                    unique_labels, key=lambda x: cluster_sizes[x], reverse=True
                ):
                    if allocated >= remaining:
                        break
                    extra_per_cluster[c] += 1
                    allocated += 1

                for c in unique_labels:
                    extra = extra_per_cluster.get(c, 0)
                    if extra <= 0:
                        continue
                    mask = labels == c
                    cluster_indices = np.where(mask)[0]
                    cluster_features = features[mask]
                    centroid = cluster_features.mean(axis=0)
                    dists = np.linalg.norm(cluster_features - centroid, axis=1)

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
            weights = np.array(
                [cluster_sizes[c] / total_size for c in unique_labels]
            )
            selected_clusters = np.random.choice(
                unique_labels,
                size=self.n_prototypes,
                replace=False,
                p=weights,
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

        N = features_np.shape[0]

        # L2归一化
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features_normed = features_np / norms

        device = torch.device("cuda")
        cluster_labels = self._spectral_cluster(features_normed, device)

        print(
            f"  谱聚类完成 [GPU]: {self.n_clusters} 簇, "
            f"大小分布: {[int((cluster_labels == i).sum()) for i in range(self.n_clusters)]}"
        )

        # Prototype选取
        prototype_indices = self._select_prototypes(features_normed, cluster_labels)

        result = {
            "X": X_np[prototype_indices],
            "Y": Y_np[prototype_indices],
            "features": features_np[prototype_indices],
            "cluster_ids": cluster_labels[prototype_indices],
            "indices": np.array(prototype_indices),
        }

        return result
