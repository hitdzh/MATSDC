"""
谱聚类与样本提取

采用 GPU-only 的 landmark 近似谱聚类：
  1. 特征 L2 归一化
  2. 采样 landmark 构造二部图相似度
  3. 在 landmark 空间求近似谱嵌入
  4. K-Means++ 聚类嵌入特征
  5. 从每个簇中选取 prototype
"""

import numpy as np
import torch


class SpectralSelector:
    """
    谱聚类 + 样本提取。

    Args:
        n_clusters: 谱聚类簇数
        n_prototypes: 最终提取的典型样本数量 N
        sigma: RBF 核带宽参数
        n_landmarks: 近似谱聚类的 landmark 数量
        chunk_size: 构建二部图亲和度时的分块大小
        seed: 随机种子
    """

    def __init__(
        self,
        n_clusters: int = 4,
        n_prototypes: int = 4,
        sigma: float | None = None,
        n_landmarks: int = 1024,
        chunk_size: int = 4096,
        seed: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_prototypes = n_prototypes
        self.sigma = sigma
        self.n_landmarks = n_landmarks
        self.chunk_size = chunk_size
        self.seed = seed

    def _synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def _chunk_slices(self, total: int):
        for start in range(0, total, self.chunk_size):
            yield start, min(start + self.chunk_size, total)

    def _estimate_sigma(self, features_t: torch.Tensor) -> float:
        n_points = features_t.shape[0]
        if n_points <= 1:
            return 1.0

        dist = torch.cdist(features_t, features_t, p=2)
        upper_idx = torch.triu_indices(
            n_points,
            n_points,
            offset=1,
            device=features_t.device,
        )
        upper = dist[upper_idx[0], upper_idx[1]]
        positive = upper[upper > 0]
        if positive.numel() == 0:
            return 1.0
        sigma = torch.median(positive).item()
        return max(sigma, 1e-5)

    def _compute_cross_affinity(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        dist_sq = torch.cdist(lhs, rhs, p=2).pow(2)
        inv_2sigma_sq = -0.5 / (sigma ** 2)
        return torch.exp(dist_sq * inv_2sigma_sq)

    def _sample_landmarks(self, features_t: torch.Tensor) -> tuple[torch.Tensor, int]:
        n_points = features_t.shape[0]
        n_landmarks = min(n_points, max(self.n_clusters, self.n_landmarks))

        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        landmark_indices = torch.randperm(n_points, generator=generator)[:n_landmarks]
        landmark_indices = landmark_indices.to(features_t.device, non_blocking=True)
        return features_t[landmark_indices], n_landmarks

    def _build_normalized_bipartite_graph(
        self,
        features_t: torch.Tensor,
        landmarks_t: torch.Tensor,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_points = features_t.shape[0]
        n_landmarks = landmarks_t.shape[0]

        row_sums = torch.empty(n_points, dtype=features_t.dtype)
        col_sums = torch.zeros(n_landmarks, device=features_t.device, dtype=features_t.dtype)

        for start, end in self._chunk_slices(n_points):
            affinity = self._compute_cross_affinity(features_t[start:end], landmarks_t, sigma)
            row_sums[start:end] = affinity.sum(dim=1).cpu()
            col_sums += affinity.sum(dim=0)
            del affinity

        row_sums = row_sums.clamp_min(1e-10)
        col_sums = col_sums.clamp_min(1e-10)
        return row_sums, col_sums

    def _approximate_spectral_embedding(
        self,
        features_t: torch.Tensor,
        landmarks_t: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        n_points = features_t.shape[0]
        k = min(self.n_clusters, landmarks_t.shape[0])

        row_sums, col_sums = self._build_normalized_bipartite_graph(
            features_t,
            landmarks_t,
            sigma,
        )
        col_scale = torch.sqrt(col_sums).unsqueeze(0)

        gram = torch.zeros(
            (landmarks_t.shape[0], landmarks_t.shape[0]),
            device=features_t.device,
            dtype=features_t.dtype,
        )
        for start, end in self._chunk_slices(n_points):
            affinity = self._compute_cross_affinity(features_t[start:end], landmarks_t, sigma)
            row_scale = torch.sqrt(
                row_sums[start:end].to(features_t.device, non_blocking=True)
            ).unsqueeze(1)
            normalized = affinity / row_scale / col_scale
            gram += normalized.transpose(0, 1) @ normalized
            del affinity, row_scale, normalized

        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(1e-10)
        basis = eigenvectors[:, -k:]
        singular_values = torch.sqrt(eigenvalues[-k:]).unsqueeze(0)

        embeddings = torch.empty(
            (n_points, k),
            device=features_t.device,
            dtype=features_t.dtype,
        )
        for start, end in self._chunk_slices(n_points):
            affinity = self._compute_cross_affinity(features_t[start:end], landmarks_t, sigma)
            row_scale = torch.sqrt(
                row_sums[start:end].to(features_t.device, non_blocking=True)
            ).unsqueeze(1)
            normalized = affinity / row_scale / col_scale
            chunk_embedding = (normalized @ basis) / singular_values
            chunk_embedding = chunk_embedding / chunk_embedding.norm(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
            embeddings[start:end] = chunk_embedding
            del affinity, row_scale, normalized, chunk_embedding

        return embeddings

    def _kmeans(
        self,
        X: torch.Tensor,
        k: int,
        n_iter: int = 50,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """
        K-Means++聚类。

        Args:
            X: (N, D) 特征向量
            k: 簇数
            n_iter: 最大迭代轮数
            tol: 收敛阈值

        Returns:
            (N,) 簇标签
        """
        n_points, dim = X.shape
        device = X.device
        k = min(k, n_points)

        torch.manual_seed(self.seed)

        centroids = torch.empty(k, dim, device=device, dtype=X.dtype)
        centroids[0] = X[torch.randperm(n_points, device=device)[0]]

        for i in range(1, k):
            dists = torch.cdist(X, centroids[:i])
            min_sq = dists.min(dim=1)[0].pow(2)
            total_mass = min_sq.sum().item()
            if total_mass <= 0:
                centroids[i] = X[torch.randperm(n_points, device=device)[0]]
                continue

            cumsum = min_sq.cumsum(dim=0)
            threshold = torch.rand(1, device=device) * cumsum[-1]
            idx = (cumsum > threshold).nonzero(as_tuple=True)[0][0].item()
            centroids[i] = X[idx]

        labels = torch.zeros(n_points, dtype=torch.long, device=device)
        for _ in range(n_iter):
            dists = torch.cdist(X, centroids)
            new_labels = dists.argmin(dim=1)

            counts = new_labels.bincount(minlength=k)
            empty = (counts == 0).nonzero(as_tuple=True)[0]
            if len(empty) > 0:
                fill_idx = torch.randperm(n_points, device=device)[:len(empty)]
                for cluster_id, point_idx in zip(empty.tolist(), fill_idx.tolist()):
                    new_labels[point_idx] = cluster_id

            new_centroids = torch.zeros_like(centroids)
            for cluster_id in range(k):
                mask = new_labels == cluster_id
                if mask.any().item():
                    new_centroids[cluster_id] = X[mask].mean(dim=0)

            change = (new_centroids - centroids).norm() / (centroids.norm() + 1e-8)
            centroids = new_centroids
            labels = new_labels
            if change < tol:
                break

        return labels.cpu().numpy()

    def _spectral_cluster(
        self,
        features_t: torch.Tensor,
    ) -> tuple[np.ndarray, float, int]:
        landmarks_t, n_landmarks = self._sample_landmarks(features_t)
        sigma = self.sigma if self.sigma is not None else self._estimate_sigma(landmarks_t)

        embeddings = self._approximate_spectral_embedding(
            features_t,
            landmarks_t,
            sigma,
        )
        labels = self._kmeans(embeddings, self.n_clusters)
        self._synchronize(features_t.device)
        return labels, sigma, n_landmarks

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

        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_features = features[mask]
            cluster_indices = np.where(mask)[0]
            cluster_sizes[cluster_id] = mask.sum()

            centroid = cluster_features.mean(axis=0)
            dists_to_centroid = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_idx_in_cluster = np.argmin(dists_to_centroid)
            cluster_prototypes[cluster_id] = cluster_indices[closest_idx_in_cluster]

        if self.n_prototypes >= n_clusters_actual:
            selected = [cluster_prototypes[cluster_id] for cluster_id in unique_labels]
            remaining = self.n_prototypes - n_clusters_actual
            if remaining > 0:
                total_size = sum(cluster_sizes.values())
                extra_per_cluster = {}
                allocated = 0
                for cluster_id in unique_labels:
                    proportion = cluster_sizes[cluster_id] / total_size
                    extra = int(np.floor(proportion * remaining))
                    extra_per_cluster[cluster_id] = extra
                    allocated += extra

                for cluster_id in sorted(
                    unique_labels,
                    key=lambda item: cluster_sizes[item],
                    reverse=True,
                ):
                    if allocated >= remaining:
                        break
                    extra_per_cluster[cluster_id] += 1
                    allocated += 1

                for cluster_id in unique_labels:
                    extra = extra_per_cluster.get(cluster_id, 0)
                    if extra <= 0:
                        continue

                    mask = labels == cluster_id
                    cluster_indices = np.where(mask)[0]
                    cluster_features = features[mask]
                    centroid = cluster_features.mean(axis=0)
                    dists = np.linalg.norm(cluster_features - centroid, axis=1)

                    count = 0
                    for rank in np.argsort(dists):
                        candidate = cluster_indices[rank]
                        if candidate not in selected:
                            selected.append(candidate)
                            count += 1
                            if count >= extra:
                                break
        else:
            total_size = sum(cluster_sizes.values())
            weights = np.array(
                [cluster_sizes[cluster_id] / total_size for cluster_id in unique_labels]
            )
            selected_clusters = np.random.choice(
                unique_labels,
                size=self.n_prototypes,
                replace=False,
                p=weights,
            )
            selected = [cluster_prototypes[cluster_id] for cluster_id in selected_clusters]

        return selected

    def cluster_and_select(
        self,
        features: torch.Tensor | np.ndarray,
        X_raw: torch.Tensor | np.ndarray,
        Y_raw: torch.Tensor | np.ndarray,
    ) -> dict:
        """
        执行谱聚类 + 样本提取。
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Stage 4 requires CUDA. Please run the pipeline on a GPU-enabled environment."
            )

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

        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features_normed = features_np / norms

        device = torch.device('cuda')
        features_t = torch.from_numpy(features_normed).to(device, non_blocking=True)
        cluster_labels, sigma, n_landmarks = self._spectral_cluster(features_t)

        print(
            "  谱聚类完成 [GPU-only approx]: "
            f"{self.n_clusters} 簇, landmarks={n_landmarks}, sigma={sigma:.6f}, "
            f"chunk_size={self.chunk_size}, "
            f"大小分布: {[int((cluster_labels == i).sum()) for i in range(self.n_clusters)]}"
        )

        prototype_indices = self._select_prototypes(features_normed, cluster_labels)
        result = {
            "X": X_np[prototype_indices],
            "Y": Y_np[prototype_indices],
            "features": features_np[prototype_indices],
            "cluster_ids": cluster_labels[prototype_indices],
            "indices": np.array(prototype_indices),
        }
        return result
