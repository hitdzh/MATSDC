"""
阶段 4：谱聚类与典型样本提取

加载阶段 3 输出的特征和原始 (X, Y) 数据，执行谱聚类，
从每个簇中提取 N 个典型样本（prototype），保存最终结果。

用法:
  python scripts/stage4_prototype_extraction.py --input outputs/stage3/trained_encoder.pt
  python scripts/stage4_prototype_extraction.py --input outputs/stage3/trained_encoder.pt --n_prototypes 8
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from configs.default import PipelineConfig
from src.clustering.spectral_selector import SpectralSelector


def main():
    parser = argparse.ArgumentParser(description='Stage 4: Spectral Clustering & Prototype Extraction')
    parser.add_argument('--input', type=str, required=True, help='阶段3 输出的 .pt 文件路径')
    parser.add_argument('--n_clusters', type=int, default=None, help='谱聚类簇数')
    parser.add_argument('--n_prototypes', type=int, default=None, help='提取典型样本数 N')
    parser.add_argument('--sigma', type=float, default=None, help='RBF 核带宽 (None=自动)')
    parser.add_argument('--output_dir', type=str, default='outputs/stage4',
                        help='输出目录')
    args = parser.parse_args()

    # 加载阶段 3 数据
    data = torch.load(args.input, weights_only=False)
    all_features = data['features']  # (N, hidden_dim)
    all_X = data['X']                # (N, seq_len, feature_dim)
    all_Y = data['Y']                # (N, pre_len, feature_dim)
    cfg: PipelineConfig = data['config']

    if args.n_clusters is not None:
        cfg.n_clusters = args.n_clusters
    if args.n_prototypes is not None:
        cfg.n_prototypes = args.n_prototypes

    print(f"[Stage 4] Spectral Clustering & Prototype Extraction")
    print(f"  n_clusters={cfg.n_clusters}, n_prototypes={cfg.n_prototypes}")

    # 执行谱聚类 + 典型样本提取
    selector = SpectralSelector(
        n_clusters=cfg.n_clusters,
        n_prototypes=cfg.n_prototypes,
        sigma=args.sigma if args.sigma else cfg.spectral_sigma,
    )

    prototypes = selector.cluster_and_select(
        features=all_features,
        X_raw=all_X,
        Y_raw=all_Y,
    )

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"Prototype Extraction Results:")
    print(f"  N = {cfg.n_prototypes} prototypes")
    print(f"  X shape:          {prototypes['X'].shape}")
    print(f"  Y shape:          {prototypes['Y'].shape}")
    print(f"  features shape:   {prototypes['features'].shape}")
    print(f"  cluster_ids:      {prototypes['cluster_ids']}")
    print(f"  indices:          {prototypes['indices']}")
    print(f"{'=' * 60}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'prototypes.pt')
    torch.save({
        'prototypes': prototypes,
        'config': cfg,
    }, save_path)
    print(f"  Saved to {save_path}")


if __name__ == '__main__':
    main()
