"""
完整 Pipeline 运行脚本

按顺序执行所有 4 个阶段：
  阶段 1: 数据准备 → 切分滑动窗口 (X, Y)
  阶段 2: Y-PreEncoder + 聚类伪标签生成
  阶段 3: X 空间度量学习 → CE + Center Loss 联合训练
  阶段 4: 谱聚类 + 典型样本提取

用法:
  python scripts/run_full_pipeline.py
  python scripts/run_full_pipeline.py --dataset ETTh1 --seq_len 336 --pre_len 96
  python scripts/run_full_pipeline.py --resume_from 3  # 从阶段3开始续跑
"""

import sys
import os
import argparse
from contextlib import nullcontext

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler

from configs.default import PipelineConfig
from src.data.sliding_window import SlidingWindowDataset
from src.data.dataset_factory import TimeSeriesForecastDataset
from src.layers.PatchTSTEncoder import PatchTSTFeatureExtractor
from src.models.x_encoder import XEncoder
from src.clustering.kcenter import generate_pseudo_labels
from src.losses.center_loss import CenterLoss
from src.losses.supcon_loss import SupervisedContrastiveLoss
from src.trainers.metric_trainer import MetricTrainer
from src.clustering.spectral_selector import SpectralSelector


def _build_window_tensors(
    series_np: np.ndarray,
    seq_len: int,
    pre_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    直接从完整时序构建滑窗视图，避免 DataLoader + cat 的高开销拷贝。
    """
    series_t = torch.from_numpy(series_np)
    window_len = seq_len + pre_len
    windows = series_t.unfold(0, window_len, 1).permute(0, 2, 1)
    all_X = windows[:, :seq_len, :]
    all_Y = windows[:, seq_len:, :]
    return all_X, all_Y


def _encode_in_batches(
    encoder: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    device: str,
    num_workers: int | None = None,
) -> torch.Tensor:
    """
    分 batch 提取特征，避免全量张量一次性占满显存。
    """
    loader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=False,
        **_make_loader_kwargs(device, num_workers=num_workers),
    )

    feature_chunks = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=device == 'cuda')
            feature_chunks.append(encoder.encode(x).cpu())

    return torch.cat(feature_chunks, dim=0)


def _make_loader_kwargs(device: str, num_workers: int | None = None) -> dict:
    """
    为大张量 batch 构建更稳妥的 DataLoader 参数。
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, max(0, cpu_count // 4))

    if device != 'cuda':
        return {'num_workers': num_workers}

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return kwargs


def _get_amp_context(device: str):
    """
    为 CUDA 训练提供自动混合精度上下文，减轻激活与带宽压力。
    """
    if device != 'cuda':
        return nullcontext, False, None

    supports_bf16 = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    def _autocast():
        if hasattr(torch, 'autocast'):
            return torch.autocast(device_type='cuda', dtype=amp_dtype)
        return torch.cuda.amp.autocast(dtype=amp_dtype)

    return _autocast, True, amp_dtype


def _extract_patchtst_encoder_config(encoder: PatchTSTFeatureExtractor, seq_len: int) -> dict:
    """
    导出 PatchTSTFeatureExtractor 的最小可复现配置。
    """
    first_layer = encoder.transformer_encoder.encoder.layers[0]
    return {
        'c_in': encoder.c_in,
        'seq_len': seq_len,
        'patch_len': encoder.patch_len,
        'stride': encoder.stride,
        'd_model': encoder.patch_projection.out_features,
        'n_heads': first_layer.self_attn.num_heads,
        'e_layers': len(encoder.transformer_encoder.encoder.layers),
        'd_ff': first_layer.linear1.out_features,
        'aggregation': encoder.aggregation,
        'concat_rev_params': encoder.concat_rev_params,
        'output_dim': encoder.get_output_dim(),
    }


def _save_y_encoder(encoder: PatchTSTFeatureExtractor, cfg: PipelineConfig, save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'encoder_config': _extract_patchtst_encoder_config(encoder, cfg.pre_len),
        'training_info': {
            'dataset': cfg.dataset,
            'seq_len': cfg.seq_len,
            'pre_len': cfg.pre_len,
            'pretrain_epochs': cfg.pretrain_epochs,
            'pretrain_batch_size': cfg.pretrain_batch_size,
            'num_workers': cfg.num_workers,
        },
    }, save_path)
    return save_path


def _save_x_encoder(
    encoder: XEncoder,
    cfg: PipelineConfig,
    save_path: str,
    center_loss: nn.Module | None = None,
    loss_history=None,
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'encoder_config': {
            'feature_dim': cfg.feature_dim,
            'hidden_dim': cfg.x_hidden_dim,
            'num_classes': cfg.K,
            'seq_len': cfg.seq_len,
            'patch_len': cfg.x_patch_len,
            'stride': cfg.x_stride,
            'd_model': cfg.x_d_model,
            'n_heads': cfg.x_n_heads,
            'e_layers': cfg.x_e_layers,
            'd_ff': cfg.x_d_ff,
            'aggregation': cfg.x_aggregation,
            'concat_rev_params': cfg.x_concat_rev_params,
        },
        'center_loss_state_dict': center_loss.state_dict() if center_loss is not None else None,
        'loss_history': loss_history,
        'training_info': {
            'dataset': cfg.dataset,
            'epochs': cfg.epochs,
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_workers,
        },
    }, save_path)
    return save_path


def _build_stage3_sampling(valid_labels: np.ndarray, num_classes: int, device: str):
    """
    为 Stage 3 构建类别均衡采样器与类别权重。
    """
    class_counts = np.bincount(valid_labels.astype(np.int64), minlength=num_classes)
    safe_class_counts = np.where(class_counts > 0, class_counts, 1)

    sample_weights = 1.0 / safe_class_counts[valid_labels.astype(np.int64)]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    class_weights = np.zeros(num_classes, dtype=np.float32)
    present_mask = class_counts > 0
    class_weights[present_mask] = 1.0 / class_counts[present_mask]
    if present_mask.any():
        class_weights[present_mask] = (
            class_weights[present_mask]
            / class_weights[present_mask].sum()
            * present_mask.sum()
        )

    return sampler, torch.tensor(class_weights, dtype=torch.float32, device=device), class_counts


def _ensure_cfg_defaults(cfg: PipelineConfig) -> PipelineConfig:
    defaults = PipelineConfig()
    for key, value in defaults.__dict__.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _normalize_kcenter_method(method: str | None, allow_removed: bool = False) -> str:
    if method in (None, 'greedy'):
        return 'kcenter'
    if method == 'robust':
        if allow_removed:
            return 'robust'
        raise ValueError("robust k-center has been removed; please use 'kcenter'.")
    if method != 'kcenter':
        raise ValueError(f"Unknown k-center method: {method}")
    return method


def _resolve_label_len(label_len: int | None, seq_len: int) -> int:
    resolved = seq_len // 2 if label_len is None else label_len
    if resolved <= 0 or resolved > seq_len:
        raise ValueError(
            f"label_len must be in [1, seq_len], got label_len={resolved}, seq_len={seq_len}."
        )
    return resolved


def _normalize_cfg(
    cfg: PipelineConfig,
    allow_removed_kcenter: bool = False,
) -> PipelineConfig:
    cfg = _ensure_cfg_defaults(cfg)
    cfg.kcenter_method = _normalize_kcenter_method(
        getattr(cfg, 'kcenter_method', None),
        allow_removed=allow_removed_kcenter,
    )
    cfg.label_len = _resolve_label_len(getattr(cfg, 'label_len', None), cfg.seq_len)
    return cfg


def _format_float_for_path(value: float | None) -> str:
    if value is None:
        return 'auto'
    return f"{value:g}".replace('-', 'm').replace('.', 'p')


def _stage1_artifact_id(cfg: PipelineConfig) -> str:
    return f"{cfg.dataset}_{cfg.seq_len}_{cfg.pre_len}"


def _stage2_artifact_id(cfg: PipelineConfig) -> str:
    return f"{_stage1_artifact_id(cfg)}_K{cfg.K}_{cfg.kcenter_method}"


def _stage3_artifact_id(cfg: PipelineConfig) -> str:
    return _stage2_artifact_id(cfg)


def _stage4_artifact_id(cfg: PipelineConfig) -> str:
    sigma_tag = _format_float_for_path(cfg.spectral_sigma)
    return (
        f"{_stage3_artifact_id(cfg)}_NC{cfg.n_clusters}_NP{cfg.n_prototypes}"
        f"_SG{sigma_tag}_LL{cfg.label_len}"
    )


def _stage1_output_path(cfg: PipelineConfig) -> str:
    return os.path.join('outputs', 'stage1', f'{_stage1_artifact_id(cfg)}_data.pt')


def _stage2_output_path(cfg: PipelineConfig) -> str:
    return os.path.join('outputs', 'stage2', f'{_stage2_artifact_id(cfg)}_pseudo_labels.pt')


def _stage3_output_path(cfg: PipelineConfig) -> str:
    return os.path.join('outputs', 'stage3', f'{_stage3_artifact_id(cfg)}_trained_encoder.pt')


def _stage4_output_path(cfg: PipelineConfig) -> str:
    return os.path.join('outputs', 'stage4', f'{_stage4_artifact_id(cfg)}_prototypes.pt')


def _condensed_output_path(cfg: PipelineConfig) -> str:
    return os.path.join(
        'outputs',
        'CondensedDatasets',
        f'Condensed_{_stage4_artifact_id(cfg)}.pt',
    )


def _y_encoder_output_path(cfg: PipelineConfig) -> str:
    return os.path.join('outputs', 'Y_Encoder', f'{_stage2_artifact_id(cfg)}_y_encoder.pt')


def _x_encoder_output_path(cfg: PipelineConfig) -> str:
    return os.path.join('outputs', 'X_Encoder', f'{_stage3_artifact_id(cfg)}_x_encoder.pt')


def _load_artifact(primary_path: str, legacy_paths: list[str] | None = None) -> tuple[dict, str]:
    for path in [primary_path, *(legacy_paths or [])]:
        if os.path.exists(path):
            return torch.load(path, weights_only=False), path
    searched = [primary_path, *(legacy_paths or [])]
    raise FileNotFoundError(f"Artifact not found. Tried: {searched}")


def _assert_cfg_matches(
    current_cfg: PipelineConfig,
    saved_cfg: PipelineConfig,
    keys: list[str],
    artifact_path: str,
) -> None:
    mismatches = []
    for key in keys:
        current_value = getattr(current_cfg, key)
        saved_value = getattr(saved_cfg, key)
        if current_value != saved_value:
            mismatches.append(f"{key}: current={current_value}, saved={saved_value}")

    if mismatches:
        mismatch_text = '\n'.join(f'  - {item}' for item in mismatches)
        raise ValueError(
            f"Resume config mismatch for {artifact_path}:\n{mismatch_text}"
        )


def run_stage1(cfg: PipelineConfig, args) -> tuple[torch.Tensor, torch.Tensor]:
    """阶段 1"""
    print(f"[Stage 1] Data Preparation — dataset={cfg.dataset}")

    if cfg.dataset == 'dummy':
        np.random.seed(42)
        time_series = np.random.randn(args.T, cfg.feature_dim).astype(np.float32)
        dataset = SlidingWindowDataset(time_series, cfg.seq_len, cfg.pre_len)
        full_series = dataset.time_series
    else:
        dataset = TimeSeriesForecastDataset(
            dataset_name=cfg.dataset,
            split='train',
            seq_len=cfg.seq_len,
            pre_len=cfg.pre_len,
        )
        cfg.feature_dim = dataset.feature_dim
        full_series = dataset.get_full_series()

    n_samples = len(dataset)
    bytes_per_value = np.dtype(np.float32).itemsize
    x_gib = (
        n_samples * cfg.seq_len * cfg.feature_dim * bytes_per_value / 1024 ** 3
    )
    y_gib = (
        n_samples * cfg.pre_len * cfg.feature_dim * bytes_per_value / 1024 ** 3
    )
    print(
        f"  Window build plan: samples={n_samples}, "
        f"estimated X={x_gib:.2f} GiB, Y={y_gib:.2f} GiB"
    )

    all_X, all_Y = _build_window_tensors(full_series, cfg.seq_len, cfg.pre_len)

    print(f"  X shape: {all_X.shape}")
    print(f"  Y shape: {all_Y.shape}")
    print(f"  Total samples: {all_X.shape[0]}")

    save_path = _stage1_output_path(cfg)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'X': all_X, 'Y': all_Y, 'config': cfg}, save_path)
    print(f"  Saved Stage 1 artifact to {save_path}")

    return all_X, all_Y


class _ReconDecoder(nn.Module):
    """线性解码器"""
    def __init__(self, feat_dim, seq_len, c_in):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, seq_len * c_in),
        )

    def forward(self, features):
        B = features.size(0)
        return self.net(features).reshape(B, self.seq_len, self.c_in)


def _pretrain_on_y(encoder, Y_data, epochs, batch_size, lr, train_ratio, device, num_workers=None):
    """自监督重建训练编码器。"""
    feat_dim = encoder.get_output_dim()
    seq_len = Y_data.size(1)
    c_in = Y_data.size(2)
    decoder = _ReconDecoder(feat_dim, seq_len, c_in).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()

    # 自监督处理
    N = Y_data.size(0)
    n_train = int(N * train_ratio)
    idx = torch.randperm(N)
    train_indices = idx[:n_train].tolist()
    val_indices = idx[n_train:].tolist()

    base_dataset = TensorDataset(Y_data)
    loader_kwargs = _make_loader_kwargs(device, num_workers=num_workers)
    train_loader = DataLoader(
        Subset(base_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(base_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    autocast_context, amp_enabled, amp_dtype = _get_amp_context(device)
    if device == 'cuda':
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    else:
        scaler = None

    print(f"\nTraining encoder ({epochs} epochs, train={n_train})")
    print(
        f"  Loader: batch_size={batch_size}, train_batches={len(train_loader)}, "
        f"val_batches={len(val_loader)}, workers={loader_kwargs.get('num_workers', 0)}, "
        f"pin_memory={loader_kwargs.get('pin_memory', False)}"
    )
    if amp_enabled:
        print(f"  AMP enabled on CUDA ({amp_dtype})")

    best_val = float('inf')
    best_state = None
    for epoch in range(epochs):
        encoder.train(); decoder.train()
        t_loss, n = 0.0, 0
        for batch_idx, b in enumerate(train_loader, start=1):
            x = b[0].to(device, non_blocking=amp_enabled)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                recon = decoder(encoder(x))
                loss = criterion(recon, x)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            t_loss += loss.item(); n += 1
        t_loss /= max(n, 1)

        encoder.eval(); decoder.eval()
        v_loss, n = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                x = b[0].to(device, non_blocking=amp_enabled)
                with autocast_context():
                    recon = decoder(encoder(x))
                    loss = criterion(recon, x)
                v_loss += loss.item(); n += 1
        v_loss /= max(n, 1)

        mark = ''
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            mark = ' *'
        print(f"    [{epoch+1}/{epochs}] Train: {t_loss:.6f}  Val: {v_loss:.6f}{mark}")

    if best_state:
        encoder.load_state_dict(best_state)
    encoder.eval()
    print(f"  Pretraining done. Best val loss: {best_val:.6f}")
    return encoder


def _build_y_encoder(cfg, device, pretrained_path=None, Y_data=None):
    """获取y_encoder"""
    # 读取预训练的 y_encoder (如果存在)
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, weights_only=False)
        ec = ckpt['encoder_config']
        encoder = PatchTSTFeatureExtractor(
            c_in=ec['c_in'], seq_len=ec['seq_len'], patch_len=ec['patch_len'],
            stride=ec['stride'], d_model=ec['d_model'], n_heads=ec['n_heads'],
            e_layers=ec['e_layers'], d_ff=ec['d_ff'], activation='gelu',
            aggregation=ec['aggregation'], concat_rev_params=ec['concat_rev_params'],
        ).to(device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        encoder.eval()
        print(f"  Loaded pretrained encoder: {pretrained_path}")
        cfg.y_hidden_dim = ec['output_dim']
        return encoder

    # 就地训练
    assert Y_data is not None
    patch_len = min(16, cfg.pre_len)
    stride = max(1, patch_len // 2)
    n_heads = max(1, cfg.y_hidden_dim // 32)
    encoder = PatchTSTFeatureExtractor(
        c_in=cfg.feature_dim, seq_len=cfg.pre_len,
        patch_len=patch_len, stride=stride,
        d_model=cfg.y_hidden_dim, n_heads=n_heads,
        e_layers=1, d_ff=cfg.y_hidden_dim * 2,
        dropout=0.1, activation='gelu',
        aggregation='max', concat_rev_params=True,
    ).to(device)
    return _pretrain_on_y(
        encoder,
        Y_data,
        cfg.pretrain_epochs,
        cfg.pretrain_batch_size,
        cfg.pretrain_lr,
        0.8,
        device,
        cfg.num_workers,
    )


def run_stage2(all_X, all_Y, cfg, device, pretrained_encoder=None) -> np.ndarray:
    """阶段 2"""
    print(f"\n[Stage 2] Pseudo Labeling")

    y_encoder = _build_y_encoder(cfg, device, pretrained_encoder, Y_data=all_Y)

    pseudo_labels, centers_idx = generate_pseudo_labels(
        Y_data=all_Y,
        encoder=y_encoder,
        K=cfg.K,
        method=cfg.kcenter_method,
        batch_size=cfg.pretrain_batch_size,
        num_workers=cfg.num_workers or 0,
    )

    label_dist = [int((pseudo_labels == k).sum()) for k in range(cfg.K)]
    print(f"  Label distribution: {label_dist}")

    save_path = _stage2_output_path(cfg)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {'pseudo_labels': pseudo_labels, 'centers_idx': centers_idx, 'config': cfg},
        save_path,
    )
    print(f"  Saved Stage 2 artifact to {save_path}")
    y_encoder_path = _save_y_encoder(y_encoder, cfg, _y_encoder_output_path(cfg))
    print(f"  Saved Y-Encoder to {y_encoder_path}")

    return pseudo_labels


def run_stage3(all_X, all_Y, pseudo_labels, cfg: PipelineConfig, device: str) -> torch.Tensor:
    """阶段 3"""
    print(f"\n[Stage 3] X-Encoder Metric Learning Training")

    # 兼容旧产物中可能存在的 -1 标签，新路径下 kcenter 仅产生非负类别
    valid_mask = pseudo_labels >= 0
    valid_X = all_X[valid_mask]
    valid_labels = pseudo_labels[valid_mask]
    if valid_labels.size == 0:
        raise ValueError("Stage 3 received no valid pseudo labels; all labels are -1.")
    sampler, class_weights, class_counts = _build_stage3_sampling(valid_labels, cfg.K, device)
    print(f"  Valid samples: {valid_X.shape[0]} / {all_X.shape[0]}")
    print(f"  Class counts: {class_counts.tolist()}")

    # 构建模型 (PatchTST backbone)
    x_encoder = XEncoder(
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.x_hidden_dim,
        num_classes=cfg.K,
        seq_len=cfg.seq_len,
        patch_len=cfg.x_patch_len,
        stride=cfg.x_stride,
        d_model=cfg.x_d_model,
        n_heads=cfg.x_n_heads,
        e_layers=cfg.x_e_layers,
        d_ff=cfg.x_d_ff,
        aggregation=cfg.x_aggregation,
        concat_rev_params=cfg.x_concat_rev_params,
    )

    center_loss = CenterLoss(
        num_classes=cfg.K,
        feat_dim=cfg.x_hidden_dim,
    )

    supcon_loss = SupervisedContrastiveLoss(
        temperature=cfg.supcon_temperature,
    )

    trainer = MetricTrainer(
        encoder=x_encoder,
        center_loss=center_loss,
        supcon_loss=supcon_loss,
        lr_encoder=cfg.lr_encoder,
        lr_centers=cfg.lr_centers,
        center_loss_weight=cfg.center_loss_weight,
        supcon_loss_weight=cfg.supcon_loss_weight,
        class_weights=class_weights,
        min_lr=cfg.min_lr,
        max_grad_norm=cfg.max_grad_norm,
        device=device,
    )

    # 训练
    train_labels = torch.from_numpy(valid_labels).long()
    train_dataset = TensorDataset(valid_X, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        **_make_loader_kwargs(device, num_workers=cfg.num_workers),
    )

    loss_history = trainer.fit(
        dataloader=train_loader,
        pseudo_labels=valid_labels,
        epochs=cfg.epochs,
    )

    # 提取全量特征
    x_encoder.eval()
    print(f"  Extracting X features in batches: batch_size={cfg.batch_size}")
    all_features = _encode_in_batches(
        encoder=x_encoder,
        data=all_X,
        batch_size=cfg.batch_size,
        device=device,
        num_workers=cfg.num_workers,
    )

    save_path = _stage3_output_path(cfg)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'encoder_state_dict': x_encoder.state_dict(),
        'center_loss_state_dict': center_loss.state_dict(),
        'loss_history': loss_history,
        'features': all_features.cpu(),
        'config': cfg,
        'X': all_X,
        'Y': all_Y,
    }, save_path)
    print(f"  Saved Stage 3 artifact to {save_path}")
    x_encoder_path = _save_x_encoder(
        x_encoder,
        cfg,
        _x_encoder_output_path(cfg),
        center_loss=center_loss,
        loss_history=loss_history,
    )
    print(f"  Saved X-Encoder to {x_encoder_path}")

    return all_features


def run_stage4(all_X, all_Y, all_features, cfg: PipelineConfig) -> dict:
    """阶段 4"""
    print(f"\n[Stage 4] Spectral Clustering & Prototype Extraction")

    selector = SpectralSelector(
        n_clusters=cfg.n_clusters,
        n_prototypes=cfg.n_prototypes,
        sigma=cfg.spectral_sigma,
        n_landmarks=cfg.spectral_landmarks,
        chunk_size=cfg.spectral_chunk_size,
        seed=cfg.spectral_seed,
    )

    prototypes = selector.cluster_and_select(
        features=all_features,
        X_raw=all_X,
        Y_raw=all_Y,
    )

    save_path = _stage4_output_path(cfg)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'prototypes': prototypes, 'config': cfg}, save_path)
    print(f"  Saved Stage 4 artifact to {save_path}")

    x_enc = torch.from_numpy(prototypes['X'])
    y_enc = torch.from_numpy(prototypes['Y'])
    x_dec = torch.cat([x_enc[:, -cfg.label_len:, :], y_enc], dim=1)

    train_data = {
        'x_enc': x_enc,
        'x_dec': x_dec,
        'y_enc': y_enc,
        'features': torch.from_numpy(prototypes['features']),
        'cluster_ids': torch.from_numpy(prototypes['cluster_ids']).long(),
        'indices': torch.from_numpy(prototypes['indices']).long(),
        'label_len': cfg.label_len,
        'config': cfg,
    }

    condensed_path = _condensed_output_path(cfg)
    os.makedirs(os.path.dirname(condensed_path), exist_ok=True)
    torch.save(train_data, condensed_path)
    print(
        f"\n  Saved condensed dataset to {condensed_path} "
        f"— x_enc: {x_enc.shape}, x_dec: {x_dec.shape}"
    )

    return prototypes


def main():
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description='Full Pipeline: Time Series Prototype Mining')
    parser.add_argument('--dataset', type=str, default=defaults.dataset,
                        choices=['dummy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                                 'electricity', 'traffic', 'weather'])
    parser.add_argument('--seq_len', type=int, default=defaults.seq_len)
    parser.add_argument('--pre_len', type=int, default=defaults.pre_len)
    parser.add_argument('--label_len', type=int, default=None,
                        help='decoder 输入长度；不指定时自动使用 seq_len // 2')
    parser.add_argument('--feature_dim', type=int, default=3, help='仅 dummy')
    parser.add_argument('--T', type=int, default=1000, help='仅 dummy')
    parser.add_argument('--K', type=int, default=defaults.K, help='K-Center 聚类数')
    parser.add_argument('--kcenter_method', type=str, default=defaults.kcenter_method,
                        choices=['kcenter'], help='伪标签聚类算法')
    parser.add_argument('--n_clusters', type=int, default=defaults.n_clusters, help='谱聚类簇数')
    parser.add_argument('--n_prototypes', type=int, default=defaults.n_prototypes, help='提取典型样本数')
    parser.add_argument('--spectral_sigma', type=float, default=defaults.spectral_sigma,
                        help='谱聚类 RBF 核带宽参数')
    parser.add_argument('--spectral_landmarks', type=int, default=defaults.spectral_landmarks,
                        help='近似谱聚类 landmark 数量')
    parser.add_argument('--spectral_chunk_size', type=int, default=defaults.spectral_chunk_size,
                        help='近似谱聚类分块大小')
    parser.add_argument('--spectral_seed', type=int, default=defaults.spectral_seed,
                        help='近似谱聚类随机种子')
    parser.add_argument('--epochs', type=int, default=defaults.epochs, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=defaults.batch_size)
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader 工作进程数；不指定则自动推断')
    parser.add_argument('--pretrain_epochs', type=int, default=defaults.pretrain_epochs, help='Y-Encoder 预训练轮数')
    parser.add_argument('--pretrain_lr', type=float, default=defaults.pretrain_lr, help='Y-Encoder 预训练学习率')
    parser.add_argument('--pretrain_batch_size', type=int, default=defaults.pretrain_batch_size, help='Y-Encoder 预训练批大小')
    parser.add_argument('--resume_from', type=int, default=1, choices=[1, 2, 3, 4],
                        help='从指定阶段开始（需有前序输出）')
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                        help='预训练 Y-Encoder 路径 (pretrain_y_encoder.py 输出) ')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('scripts/run_full_pipeline.py requires CUDA. CPU execution is not supported.')
    device = 'cuda'
    print(f"Device: {device}")

    # 构建配置
    cfg = PipelineConfig(
        dataset=args.dataset,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        label_len=args.label_len,
        feature_dim=args.feature_dim,
        K=args.K,
        kcenter_method=args.kcenter_method,
        n_clusters=args.n_clusters,
        n_prototypes=args.n_prototypes,
        spectral_sigma=args.spectral_sigma,
        spectral_landmarks=args.spectral_landmarks,
        spectral_chunk_size=args.spectral_chunk_size,
        spectral_seed=args.spectral_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_batch_size=args.pretrain_batch_size,
    )
    cfg = _normalize_cfg(cfg)

    # ===== 阶段 1 =====
    if args.resume_from <= 1:
        all_X, all_Y = run_stage1(cfg, args)
    else:
        print(f"\n[Stage 1] Skipped (resume_from={args.resume_from})")
        data, loaded_path = _load_artifact(
            _stage1_output_path(cfg),
            legacy_paths=[f'outputs/stage1/{cfg.dataset}_data.pt'],
        )
        all_X, all_Y = data['X'], data['Y']
        saved_cfg = _normalize_cfg(data['config'], allow_removed_kcenter=True)
        _assert_cfg_matches(cfg, saved_cfg, ['dataset', 'seq_len', 'pre_len'], loaded_path)
        cfg.feature_dim = all_X.shape[-1]
        print(f"  Loaded Stage 1 artifact from {loaded_path}")

    # ===== 阶段 2 =====
    if args.resume_from <= 2:
        pseudo_labels = run_stage2(all_X, all_Y, cfg, device,
                                   pretrained_encoder=args.pretrained_encoder)
    else:
        print(f"\n[Stage 2] Skipped (resume_from={args.resume_from})")
        data, loaded_path = _load_artifact(
            _stage2_output_path(cfg),
            legacy_paths=['outputs/stage2/pseudo_labels.pt'],
        )
        pseudo_labels = data['pseudo_labels']
        saved_cfg = _normalize_cfg(data['config'], allow_removed_kcenter=True)
        _assert_cfg_matches(
            cfg,
            saved_cfg,
            ['dataset', 'seq_len', 'pre_len', 'K', 'kcenter_method'],
            loaded_path,
        )
        print(f"  Loaded Stage 2 artifact from {loaded_path}")

    # ===== 阶段 3 =====
    if args.resume_from <= 3:
        all_features = run_stage3(all_X, all_Y, pseudo_labels, cfg, device)
    else:
        print(f"\n[Stage 3] Skipped (resume_from={args.resume_from})")
        data, loaded_path = _load_artifact(
            _stage3_output_path(cfg),
            legacy_paths=['outputs/stage3/trained_encoder.pt'],
        )
        all_features = data['features']
        all_X, all_Y = data['X'], data['Y']
        saved_cfg = _normalize_cfg(data['config'], allow_removed_kcenter=True)
        _assert_cfg_matches(
            cfg,
            saved_cfg,
            ['dataset', 'seq_len', 'pre_len', 'K', 'kcenter_method'],
            loaded_path,
        )
        cfg.feature_dim = all_X.shape[-1]
        print(f"  Loaded Stage 3 artifact from {loaded_path}")

    # ===== 阶段 4 =====
    prototypes = run_stage4(all_X, all_Y, all_features, cfg)

    # 最终结果
    print("\n")
    print(f"  N = {cfg.n_prototypes} prototypes extracted")
    print(f"  X shape:          {prototypes['X'].shape}")
    print(f"  Y shape:          {prototypes['Y'].shape}")
    print(f"  features shape:   {prototypes['features'].shape}")
    print(f"  cluster_ids:      {prototypes['cluster_ids']}")
    print(f"  indices:          {prototypes['indices']}")


if __name__ == '__main__':
    main()
