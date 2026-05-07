"""
Modality-specific encoders for NMR-MultiFuse v3.

  - H1Encoder:  1H-NMR Set Transformer (peaks as unordered set)
  - C13Encoder: 13C-NMR Set Transformer
  - IREncoder:  IR 1D CNN + Transformer
  - MolEncoder: SMILES fingerprint -> embedding
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiheadAttentionBlock(nn.Module):
    """Standard multi-head attention with pre-norm."""

    def __init__(self, dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class InducedSetAttentionBlock(nn.Module):
    """
    ISAB from Set Transformer (Lee et al., 2019).
    Uses inducing points to reduce O(N^2) to O(N*M) where M = num_inducing.
    """

    def __init__(self, dim: int, nhead: int = 4, num_inducing: int = 32, dropout: float = 0.1):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)

        # Inducing -> Input attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)

        # Input -> Inducing attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        I = self.inducing_points.expand(B, -1, -1)

        # Step 1: Inducing points attend to input
        h = self.norm1(I)
        x_norm = self.norm1(x)
        h, _ = self.attn1(h, x_norm, x_norm, key_padding_mask=key_padding_mask)
        I = I + h

        # Step 2: Input attends to inducing points
        h = self.norm2(x)
        I_norm = self.norm2(I)
        h, _ = self.attn2(h, I_norm, I_norm)
        x = x + h

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class PoolingByMultiheadAttention(nn.Module):
    """PMA from Set Transformer — pool set to fixed-size output."""

    def __init__(self, dim: int, num_seeds: int = 1, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        S = self.seed.expand(B, -1, -1)
        h = self.norm(S)
        x_norm = self.norm(x)
        out, _ = self.attn(h, x_norm, x_norm, key_padding_mask=key_padding_mask)
        return S + out  # (B, num_seeds, dim)


# ---------------------------------------------------------------------------
# 1H-NMR Encoder (Set Transformer)
# ---------------------------------------------------------------------------

class H1Encoder(nn.Module):
    """
    1H-NMR peaks -> set embedding.

    Input: peaks (B, N, 3) [shift, intensity, mult_code], mask (B, N) bool
    Output: peak_features (B, N, D), cls_token (B, D)
    """

    def __init__(self, dim: int = 256, nhead: int = 4, num_layers: int = 4,
                 num_inducing: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(7, dim)  # H1_FEAT_DIM = 7
        self.pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # learnable, broadcast

        self.layers = nn.ModuleList([
            InducedSetAttentionBlock(dim, nhead, num_inducing, dropout)
            for _ in range(num_layers)
        ])

        self.pool = PoolingByMultiheadAttention(dim, num_seeds=1, nhead=nhead, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, peaks: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            peaks: (B, N, 7) float [shift, width, nH, mult, j1, j2, j3]
            mask: (B, N) bool — True for valid peaks
        Returns:
            features: (B, N, D)
            cls_token: (B, D)
        """
        x = self.input_proj(peaks)  # (B, N, D)
        x = x + self.pos_embed

        # Invert mask for MultiheadAttention (True = ignore)
        key_padding_mask = ~mask

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # Pool to cls token
        cls = self.pool(x, key_padding_mask=key_padding_mask)  # (B, 1, D)
        cls = self.out_norm(cls.squeeze(1))  # (B, D)

        return x, cls


# ---------------------------------------------------------------------------
# 13C-NMR Encoder (Set Transformer)
# ---------------------------------------------------------------------------

class C13Encoder(nn.Module):
    """
    13C-NMR peaks -> set embedding.

    Input: peaks (B, N, 2) [shift, intensity], mask (B, N) bool
    Output: peak_features (B, N, D), cls_token (B, D)
    """

    def __init__(self, dim: int = 256, nhead: int = 4, num_layers: int = 4,
                 num_inducing: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(3, dim)  # C13_FEAT_DIM = 3
        self.pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.layers = nn.ModuleList([
            InducedSetAttentionBlock(dim, nhead, num_inducing, dropout)
            for _ in range(num_layers)
        ])

        self.pool = PoolingByMultiheadAttention(dim, num_seeds=1, nhead=nhead, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, peaks: torch.Tensor, mask: torch.Tensor):
        """peaks: (B, N, 3) [shift, intensity, width]"""
        x = self.input_proj(peaks)
        x = x + self.pos_embed

        key_padding_mask = ~mask
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        cls = self.pool(x, key_padding_mask=key_padding_mask).squeeze(1)
        cls = self.out_norm(cls)

        return x, cls


# ---------------------------------------------------------------------------
# IR Encoder (CNN + Transformer)
# ---------------------------------------------------------------------------

class IREncoder(nn.Module):
    """
    IR spectrum -> embedding.

    Input: spectrum (B, L) float
    Output: ir_features (B, L//8, D), cls_token (B, D)
    """

    def __init__(self, dim: int = 256, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        # CNN stem: 3 layers with stride=2 each -> L//8
        self.cnn = nn.Sequential(
            nn.Conv1d(1, dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(dim // 4),
            nn.GELU(),
            nn.Conv1d(dim // 4, dim // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(dim // 2),
            nn.GELU(),
            nn.Conv1d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

        # Sinusoidal positional encoding
        self.register_buffer("_pe_cache", None)

        # Transformer layers
        self.layers = nn.ModuleList([
            MultiheadAttentionBlock(dim, nhead, dropout) for _ in range(num_layers)
        ])

        self.pool = PoolingByMultiheadAttention(dim, num_seeds=1, nhead=nhead, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def _get_pe(self, seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, D)

    def forward(self, spectrum: torch.Tensor):
        """
        Args:
            spectrum: (B, L) float
        Returns:
            features: (B, L//8, D)
            cls_token: (B, D)
        """
        x = spectrum.unsqueeze(1)  # (B, 1, L)
        x = self.cnn(x)            # (B, D, L//8)
        x = x.transpose(1, 2)      # (B, L//8, D)

        # Add positional encoding
        x = x + self._get_pe(x.size(1), x.size(2), x.device)

        for layer in self.layers:
            x = layer(x)

        cls = self.pool(x).squeeze(1)  # (B, D)
        cls = self.out_norm(cls)

        return x, cls


# ---------------------------------------------------------------------------
# Molecular Encoder (Fingerprint MLP)
# ---------------------------------------------------------------------------

class MolEncoder(nn.Module):
    """
    Morgan fingerprint -> molecular embedding.

    Input: fp (B, 2048) float
    Output: mol_repr (B, D) float
    """

    def __init__(self, dim: int = 256, fp_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp)
