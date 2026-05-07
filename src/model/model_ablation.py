"""
Ablation variants of NMRMultiFuse.

Each variant removes one structural component to isolate its contribution.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import H1Encoder, C13Encoder, IREncoder, MolEncoder
from .caps import CAPS, caps_calibration_loss
from .fusion import GatedFusion
from .losses import NMRMultiFuseLoss


class NMRMultiFuse_NoCaps(nn.Module):
    """Ablation A: No CAPS — missing modalities filled with zero vectors."""

    def __init__(self, dim=256, nhead=4, num_encoder_layers=4, num_inducing=32,
                 fp_dim=2048, dropout=0.1, mask_prob=0.5):
        super().__init__()
        self.dim = dim
        self.mask_prob = mask_prob
        self.h1_encoder = H1Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.c13_encoder = C13Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.ir_encoder = IREncoder(dim, nhead, num_encoder_layers, dropout)
        self.mol_encoder = MolEncoder(dim, fp_dim, dropout)
        self.fusion = GatedFusion(dim, num_modalities=3)

    def forward(self, batch):
        B = batch["mol_fp"].size(0)
        modality_mask = batch["modality_mask"].clone()

        _, cls_1h = self.h1_encoder(batch["h_nmr_peaks"], batch["h_nmr_mask"])
        _, cls_13c = self.c13_encoder(batch["c_nmr_peaks"], batch["c_nmr_mask"])
        _, cls_ir = self.ir_encoder(batch["ir_spectrum"])
        mol_repr = self.mol_encoder(batch["mol_fp"])

        cls_tokens = [cls_1h, cls_13c, cls_ir]

        # Training-time masking (same as full model)
        if self.training and self.mask_prob > 0:
            for b in range(B):
                available = modality_mask[b].nonzero(as_tuple=True)[0]
                if len(available) <= 1:
                    continue
                n_mask = random.randint(1, min(2, len(available) - 1))
                mask_indices = available[torch.randperm(len(available))[:n_mask]]
                for idx in mask_indices:
                    if random.random() < self.mask_prob:
                        modality_mask[b, idx] = False

        # NO CAPS: just zero out missing modalities
        for m in range(3):
            missing = ~modality_mask[:, m]
            if missing.any():
                cls_tokens[m] = cls_tokens[m].clone()
                cls_tokens[m][missing] = 0.0

        fused_repr = self.fusion(cls_tokens)
        return {
            "fused_repr": fused_repr,
            "mol_repr": mol_repr,
            "cls_tokens": cls_tokens,
            "modality_mask": modality_mask,
            "calib_loss": torch.tensor(0.0, device=fused_repr.device),
        }


class NMRMultiFuse_MMPOnly(nn.Module):
    """Ablation B: MMP only — cross-attention projection, no proxy token fallback."""

    def __init__(self, dim=256, nhead=4, num_encoder_layers=4, num_inducing=32,
                 fp_dim=2048, dropout=0.1, mask_prob=0.5):
        super().__init__()
        self.dim = dim
        self.mask_prob = mask_prob
        self.h1_encoder = H1Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.c13_encoder = C13Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.ir_encoder = IREncoder(dim, nhead, num_encoder_layers, dropout)
        self.mol_encoder = MolEncoder(dim, fp_dim, dropout)
        self.fusion = GatedFusion(dim, num_modalities=3)

        # Cross-attention only (no proxy, no gate)
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.query_tokens = nn.Parameter(torch.randn(3, dim) * 0.02)

    def forward(self, batch):
        B = batch["mol_fp"].size(0)
        modality_mask = batch["modality_mask"].clone()

        _, cls_1h = self.h1_encoder(batch["h_nmr_peaks"], batch["h_nmr_mask"])
        _, cls_13c = self.c13_encoder(batch["c_nmr_peaks"], batch["c_nmr_mask"])
        _, cls_ir = self.ir_encoder(batch["ir_spectrum"])
        mol_repr = self.mol_encoder(batch["mol_fp"])

        cls_tokens = [cls_1h, cls_13c, cls_ir]

        if self.training and self.mask_prob > 0:
            for b in range(B):
                available = modality_mask[b].nonzero(as_tuple=True)[0]
                if len(available) <= 1:
                    continue
                n_mask = random.randint(1, min(2, len(available) - 1))
                mask_indices = available[torch.randperm(len(available))[:n_mask]]
                for idx in mask_indices:
                    if random.random() < self.mask_prob:
                        modality_mask[b, idx] = False

        # Zero out missing
        for m in range(3):
            missing = ~modality_mask[:, m]
            if missing.any():
                cls_tokens[m] = cls_tokens[m].clone()
                cls_tokens[m][missing] = 0.0

        # MMP only: project missing from available via cross-attention
        completed = list(cls_tokens)
        for m in range(3):
            missing = ~modality_mask[:, m]
            if not missing.any():
                continue

            avail_indices = [j for j in range(3) if j != m]
            for b_idx in range(B):
                if not missing[b_idx]:
                    continue
                avail = []
                for j in avail_indices:
                    if modality_mask[b_idx, j]:
                        avail.append(cls_tokens[j][b_idx])
                if not avail:
                    continue  # no available modalities, stay zero
                kv = torch.stack(avail).unsqueeze(0)  # (1, num_avail, D)
                q = self.query_tokens[m].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                proj, _ = self.cross_attn(q, self.cross_norm(kv), self.cross_norm(kv))
                completed[m] = completed[m].clone()
                completed[m][b_idx] = proj.squeeze(0).squeeze(0)

        fused_repr = self.fusion(completed)
        return {
            "fused_repr": fused_repr,
            "mol_repr": mol_repr,
            "cls_tokens": completed,
            "modality_mask": modality_mask,
            "calib_loss": torch.tensor(0.0, device=fused_repr.device),
        }


class NMRMultiFuse_ProxyOnly(nn.Module):
    """Ablation C: Proxy only — static learnable proxy tokens, no cross-attention."""

    def __init__(self, dim=256, nhead=4, num_encoder_layers=4, num_inducing=32,
                 fp_dim=2048, dropout=0.1, mask_prob=0.5):
        super().__init__()
        self.dim = dim
        self.mask_prob = mask_prob
        self.h1_encoder = H1Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.c13_encoder = C13Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.ir_encoder = IREncoder(dim, nhead, num_encoder_layers, dropout)
        self.mol_encoder = MolEncoder(dim, fp_dim, dropout)
        self.fusion = GatedFusion(dim, num_modalities=3)
        self.proxy_tokens = nn.Parameter(torch.randn(3, dim) * 0.02)

    def forward(self, batch):
        B = batch["mol_fp"].size(0)
        modality_mask = batch["modality_mask"].clone()

        _, cls_1h = self.h1_encoder(batch["h_nmr_peaks"], batch["h_nmr_mask"])
        _, cls_13c = self.c13_encoder(batch["c_nmr_peaks"], batch["c_nmr_mask"])
        _, cls_ir = self.ir_encoder(batch["ir_spectrum"])
        mol_repr = self.mol_encoder(batch["mol_fp"])

        cls_tokens = [cls_1h, cls_13c, cls_ir]

        if self.training and self.mask_prob > 0:
            for b in range(B):
                available = modality_mask[b].nonzero(as_tuple=True)[0]
                if len(available) <= 1:
                    continue
                n_mask = random.randint(1, min(2, len(available) - 1))
                mask_indices = available[torch.randperm(len(available))[:n_mask]]
                for idx in mask_indices:
                    if random.random() < self.mask_prob:
                        modality_mask[b, idx] = False

        # Proxy only: replace missing with static learned proxy
        completed = list(cls_tokens)
        for m in range(3):
            missing = ~modality_mask[:, m]
            if missing.any():
                completed[m] = completed[m].clone()
                completed[m][missing] = self.proxy_tokens[m].unsqueeze(0).expand(missing.sum(), -1)

        fused_repr = self.fusion(completed)
        return {
            "fused_repr": fused_repr,
            "mol_repr": mol_repr,
            "cls_tokens": completed,
            "modality_mask": modality_mask,
            "calib_loss": torch.tensor(0.0, device=fused_repr.device),
        }


class NMRMultiFuse_DataTypeDropout(nn.Module):
    """
    Ablation D: SPECTRE-style data-type dropout.
    During training, randomly drop modalities (like SPECTRE).
    During inference, missing modalities filled with a learned default token per modality.
    No cross-attention, no confidence gating — just dropout + learned defaults.
    This is SPECTRE's core missing-modality mechanism adapted to our framework.
    """

    def __init__(self, dim=256, nhead=4, num_encoder_layers=4, num_inducing=32,
                 fp_dim=2048, dropout=0.1, mask_prob=0.5):
        super().__init__()
        self.dim = dim
        self.mask_prob = mask_prob
        self.h1_encoder = H1Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.c13_encoder = C13Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.ir_encoder = IREncoder(dim, nhead, num_encoder_layers, dropout)
        self.mol_encoder = MolEncoder(dim, fp_dim, dropout)
        self.fusion = GatedFusion(dim, num_modalities=3)
        # Learned default tokens (SPECTRE's approach)
        self.default_tokens = nn.Parameter(torch.randn(3, dim) * 0.02)

    def forward(self, batch):
        B = batch["mol_fp"].size(0)
        modality_mask = batch["modality_mask"].clone()

        _, cls_1h = self.h1_encoder(batch["h_nmr_peaks"], batch["h_nmr_mask"])
        _, cls_13c = self.c13_encoder(batch["c_nmr_peaks"], batch["c_nmr_mask"])
        _, cls_ir = self.ir_encoder(batch["ir_spectrum"])
        mol_repr = self.mol_encoder(batch["mol_fp"])

        cls_tokens = [cls_1h, cls_13c, cls_ir]

        # SPECTRE-style: random dropout during training
        if self.training and self.mask_prob > 0:
            for b in range(B):
                available = modality_mask[b].nonzero(as_tuple=True)[0]
                if len(available) <= 1:
                    continue
                n_mask = random.randint(1, min(2, len(available) - 1))
                mask_indices = available[torch.randperm(len(available))[:n_mask]]
                for idx in mask_indices:
                    if random.random() < self.mask_prob:
                        modality_mask[b, idx] = False

        # Replace missing with learned default tokens (not zero, not projected)
        for m in range(3):
            missing = ~modality_mask[:, m]
            if missing.any():
                cls_tokens[m] = cls_tokens[m].clone()
                cls_tokens[m][missing] = self.default_tokens[m].unsqueeze(0).expand(missing.sum(), -1)

        fused_repr = self.fusion(cls_tokens)
        return {
            "fused_repr": fused_repr,
            "mol_repr": mol_repr,
            "cls_tokens": cls_tokens,
            "modality_mask": modality_mask,
            "calib_loss": torch.tensor(0.0, device=fused_repr.device),
        }


ABLATION_REGISTRY = {
    "no_caps": NMRMultiFuse_NoCaps,
    "mmp_only": NMRMultiFuse_MMPOnly,
    "proxy_only": NMRMultiFuse_ProxyOnly,
    "spectre_dropout": NMRMultiFuse_DataTypeDropout,
}


def build_ablation_model(ablation_name, cfg):
    """Build an ablation variant."""
    cls = ABLATION_REGISTRY[ablation_name]
    model = cls(
        dim=cfg.get("dim", 256),
        nhead=cfg.get("nhead", 4),
        num_encoder_layers=cfg.get("num_encoder_layers", 4),
        num_inducing=cfg.get("num_inducing", 32),
        fp_dim=cfg.get("fp_dim", 2048),
        dropout=cfg.get("dropout", 0.1),
        mask_prob=cfg.get("mask_prob", 0.5),
    )
    criterion = NMRMultiFuseLoss(
        temperature=cfg.get("contrastive_temp", 0.07),
        inter_weight=cfg.get("inter_weight", 1.0),
        calib_weight=0.0,  # ablations don't use calib
    )
    return model, criterion
