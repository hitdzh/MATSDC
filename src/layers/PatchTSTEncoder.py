import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)

    Normalizes each instance independently by subtracting mean and dividing by std.
    This forces the Transformer to learn pure morphological details rather than
    absolute scale information, which is crucial for capturing microscopic patterns.

    The affine parameters (mean, std) can be optionally concatenated to the final
    feature vector to make the representation scale-aware for K-Center.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_features = x.shape

        # Compute mean and std along sequence dimension for each instance and feature
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        std = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, C)

        # Normalize
        normalized = (x - mean) / std

        # Apply learnable affine transformation if enabled
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias

        return normalized, mean, std

    def inverse(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        return x * std + mean


class Patching(nn.Module):
    """
    Overlapping Micro-Patching Layer

    Creates overlapping patches from input time series to preserve microscopic details
    at patch boundaries. This is critical for capturing local mutations and high-
    frequency glitches that would be lost with non-overlapping patches.
    """

    def __init__(self, patch_len: int = 16, stride: int = 8):
        super(Patching, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to (possibly overlapping) patches.

        Overlapping patches are extracted via manual slicing to avoid
        PyTorch unfold's per-channel stride semantics.

        Args:
            x: shape (batch_size, seq_len, c_in)

        Returns:
            patches: shape (batch_size, num_patches, patch_len, c_in)
        """
        batch_size, seq_len, c_in = x.shape
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        # Manual loop for reliability — supports arbitrary (stride < patch_len) overlap
        patches_list = []
        for start in range(0, num_patches * self.stride, self.stride):
            if start + self.patch_len > seq_len:
                break
            patch = x[:, start:start + self.patch_len, :]  # (batch, patch_len, c_in)
            patches_list.append(patch)

        # Stack: (batch, num_patches, patch_len, c_in)
        patches = torch.stack(patches_list, dim=1)
        return patches


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for patches

    Unlike sinusoidal encoding, learnable encoding can adapt to the specific
    temporal patterns in the data during training.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_patches = x.size(1)
        return x + self.position_embedding[:num_patches, :].unsqueeze(0)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder with multi-head self-attention

    Processes patchified time series to extract high-level features while
    preserving local details through overlapping patches.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super(TransformerEncoder, self).__init__()

        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=e_layers,
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AggregationHead(nn.Module):
    """
    Aggregation strategies for converting patch-level features to single vector

    CRITICAL: No Global Average Pooling (GAP) as it acts like a low-pass filter
    and smooths out local anomalies. Instead, we provide two strategies:

    1. Max Pooling: Extracts most significant local activation features
    2. Flatten + Linear: Preserves absolute temporal position of micro-features
    """

    def __init__(
        self,
        strategy: str = 'max',
        num_patches: int = None,
        d_model: int = None,
        target_dim: int = None
    ):
        super(AggregationHead, self).__init__()

        self.strategy = strategy

        if strategy == 'max':
            pass  # No parameters needed

        elif strategy == 'flatten':
            if num_patches is None or d_model is None or target_dim is None:
                raise ValueError(
                    "num_patches, d_model, and target_dim must be provided for 'flatten' strategy"
                )
            self.projection = nn.Linear(num_patches * d_model, target_dim)

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.strategy == 'max':
            # Max pooling over sequence dimension
            return x.max(dim=1)[0]  # (batch_size, d_model)

        elif self.strategy == 'flatten':
            batch_size = x.size(0)
            x_flat = x.reshape(batch_size, -1)  # (batch_size, num_patches * d_model)
            return self.projection(x_flat)  # (batch_size, target_dim)


class PatchTSTFeatureExtractor(nn.Module):
    """
    Architecture:
        1. RevIN: Normalize to remove macro-level mean and variance
        2. Patching: Create overlapping patches (stride < patch_len)
        3. Projection: Linear mapping to d_model dimension
        4. Positional Encoding: Add learnable position information
        5. Transformer Encoder: Multi-layer self-attention processing
        6. Aggregation: Max pooling or Flatten+Linear
        7. RevIN Concatenation (optional): Append mean and std to output
        8. L2 Normalization: Normalize for cosine distance in K-Center
    """

    def __init__(
        self,
        c_in: int = 21,              # Input feature dimension
        seq_len: int = 336,          # Input sequence length (pre_len as Y window)
        patch_len: int = 16,         # Patch length (small for fine-grained analysis)
        stride: int = 8,             # Stride (must be < patch_len for overlap)
        d_model: int = 256,          # Model dimension
        n_heads: int = 8,            # Number of attention heads
        e_layers: int = 4,           # Number of encoder layers
        d_ff: int = 1024,             # Feed-forward network dimension
        dropout: float = 0.1,        # Dropout rate
        activation: str = 'gelu',    # Activation function
        aggregation: str = 'max',    # Aggregation strategy: 'max' or 'flatten'
        target_dim: int = 256,       # Target output dimension (for 'flatten' strategy)
        concat_rev_params: bool = True  # Whether to concatenate RevIN parameters
    ):
        super(PatchTSTFeatureExtractor, self).__init__()

        self.c_in = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.aggregation = aggregation
        self.concat_rev_params = concat_rev_params

        # Validate parameters
        if stride >= patch_len:
            raise ValueError(f"stride ({stride}) must be < patch_len ({patch_len}) for overlapping patches")

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # 1. RevIN: Reversible Instance Normalization
        self.revin = RevIN(num_features=c_in, eps=1e-5, affine=False)

        # 2. Patching: overlapping patch extraction logic is inline in forward()
        # (Patching class is provided for standalone use; see forward() for actual implementation)

        # 3. Projection: Map each patch to d_model dimension
        self.patch_projection = nn.Linear(patch_len * c_in, d_model)

        # 4. Positional Encoding: Learnable position embeddings
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=self.num_patches)

        # 5. Transformer Encoder: Multi-layer self-attention
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )

        # 6. Aggregation Head
        if aggregation == 'max':
            agg_output_dim = d_model
        elif aggregation == 'flatten':
            agg_output_dim = target_dim
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        self.aggregation_head = AggregationHead(
            strategy=aggregation,
            num_patches=self.num_patches,
            d_model=d_model,
            target_dim=target_dim
        )

        # 7. Optional: Concatenate RevIN parameters
        if concat_rev_params:
            final_output_dim = agg_output_dim + 2 * c_in
        else:
            final_output_dim = agg_output_dim

        self.final_output_dim = final_output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features from multivariate time series

        Args:
            x: Input time series of shape (batch_size, seq_len, c_in)

        Returns:
            features: L2-normalized feature vector, shape (batch_size, final_output_dim)
        """
        batch_size = x.size(0)

        # Step 1: Apply RevIN normalization
        x_normalized, revin_mean, revin_std = self.revin(x)

        # Step 2: Create overlapping patches via manual slicing
        # (explicit loop avoids PyTorch unfold's per-channel stride semantics)
        seq_len = x_normalized.size(1)
        patches_list = []
        for start in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x_normalized[:, start:start + self.patch_len, :]  # (B, patch_len, c_in)
            patches_list.append(patch)
        patches = torch.stack(patches_list, dim=1)  # (B, actual_num_patches, patch_len, c_in)

        # Step 3: Project patches to d_model dimension
        patches_flat = patches.reshape(batch_size, patches.size(1), -1)  # (B, num_patches, patch_len*c_in)
        patch_embeddings = self.patch_projection(patches_flat)

        # Step 4: Add positional encoding
        patch_embeddings = self.pos_encoding(patch_embeddings)

        # Step 5: Process through Transformer Encoder
        encoded = self.transformer_encoder(patch_embeddings)

        # Step 6: Aggregate patch features to single vector
        features = self.aggregation_head(encoded)

        # Step 7: Concatenate RevIN parameters
        if self.concat_rev_params:
            revin_mean = revin_mean.squeeze(1)
            revin_std = revin_std.squeeze(1)
            features = torch.cat([features, revin_mean, revin_std], dim=1)

        # Step 8: L2 Normalization
        features = F.normalize(features, p=2, dim=1)

        return features

    def get_output_dim(self) -> int:
        return self.final_output_dim


def create_patchtst_encoder(
    c_in: int = 21,
    seq_len: int = 96,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 128,
    n_heads: int = 4,
    e_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
    activation: str = 'gelu',
    aggregation: str = 'max',
    target_dim: int = 256,
    concat_rev_params: bool = True,
    pretrained_path: str = None,
    device: str = 'cuda'
) -> PatchTSTFeatureExtractor:
    """
    Factory function to create a PatchTST Feature Extractor (Y-PreEncoder)

    Args:
        c_in: Number of input features
        seq_len: Input sequence length (pre_len)
        patch_len: Patch length
        stride: Stride between patches
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_ff: Feed-forward network dimension
        dropout: Dropout rate
        activation: Activation function
        aggregation: Aggregation strategy ('max' or 'flatten')
        target_dim: Target output dimension (for 'flatten' strategy)
        concat_rev_params: Whether to concatenate RevIN parameters
        pretrained_path: Path to pretrained model weights
        device: Device to load model on

    Returns:
        encoder: Configured feature extractor
    """
    encoder = PatchTSTFeatureExtractor(
        c_in=c_in,
        seq_len=seq_len,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        aggregation=aggregation,
        target_dim=target_dim,
        concat_rev_params=concat_rev_params
    ).to(device)

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=torch.device(device))
        encoder.load_state_dict(state_dict)
        print(f"Loaded pretrained encoder from {pretrained_path}")

    encoder.eval()
    return encoder
