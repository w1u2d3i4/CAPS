"""
NMRMultiFuse: Complete multi-modal spectroscopy model.

Architecture:
  3 modality encoders -> CAPS missing-modality completion -> Gated Fusion -> outputs
"""

import random

import torch
import torch.nn as nn

from .encoders import H1Encoder, C13Encoder, IREncoder, MolEncoder
from .caps import CAPS, caps_calibration_loss
from .fusion import GatedFusion
from .losses import NMRMultiFuseLoss


class NMRMultiFuse(nn.Module):
    """
    Multi-modal spectroscopy model with CAPS.

    Forward returns fused_repr for contrastive learning and retrieval.
    """

    def __init__(
        self,
        dim: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_inducing: int = 32,
        fp_dim: int = 2048,
        dropout: float = 0.1,
        mask_prob: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.mask_prob = mask_prob

        # Modality encoders
        self.h1_encoder = H1Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.c13_encoder = C13Encoder(dim, nhead, num_encoder_layers, num_inducing, dropout)
        self.ir_encoder = IREncoder(dim, nhead, num_encoder_layers, dropout)
        self.mol_encoder = MolEncoder(dim, fp_dim, dropout)

        # CAPS fusion
        self.caps = CAPS(dim, num_modalities=3, nhead=nhead, dropout=dropout)
        self.fusion = GatedFusion(dim, num_modalities=3)

    def encode_modalities(self, batch: dict) -> tuple[list, torch.Tensor]:
        """
        Encode all available modalities.

        Returns:
            cls_tokens: list of 3 tensors (B, D)
            modality_mask: (B, 3) bool (after training-time random masking)
        """
        device = batch["mol_fp"].device
        B = batch["mol_fp"].size(0)
        modality_mask = batch["modality_mask"].clone()  # (B, 3) bool

        # Encode each modality
        _, cls_1h = self.h1_encoder(batch["h_nmr_peaks"], batch["h_nmr_mask"])
        _, cls_13c = self.c13_encoder(batch["c_nmr_peaks"], batch["c_nmr_mask"])
        _, cls_ir = self.ir_encoder(batch["ir_spectrum"])

        cls_tokens = [cls_1h, cls_13c, cls_ir]

        # Training-time random modality masking
        if self.training and self.mask_prob > 0:
            for b in range(B):
                # Don't mask all modalities — keep at least 1
                available = modality_mask[b].nonzero(as_tuple=True)[0]
                if len(available) <= 1:
                    continue
                # Random number of modalities to mask (1 or 2)
                n_mask = random.randint(1, min(2, len(available) - 1))
                mask_indices = available[torch.randperm(len(available))[:n_mask]]
                for idx in mask_indices:
                    if random.random() < self.mask_prob:
                        modality_mask[b, idx] = False

        # Zero out masked modalities
        for m in range(3):
            missing = ~modality_mask[:, m]
            if missing.any():
                cls_tokens[m] = cls_tokens[m].clone()
                cls_tokens[m][missing] = 0.0

        return cls_tokens, modality_mask

    def forward(self, batch: dict) -> dict:
        """
        Full forward pass.

        Args:
            batch: dict from collate_multimodal
        Returns:
            dict with fused_repr, mol_repr, cls_tokens, modality_mask, caps_details
        """
        # 1. Encode modalities (with training-time masking)
        # Save pre-mask tokens for calibration loss
        with torch.no_grad():
            _, gt_cls_1h = self.h1_encoder(batch["h_nmr_peaks"], batch["h_nmr_mask"])
            _, gt_cls_13c = self.c13_encoder(batch["c_nmr_peaks"], batch["c_nmr_mask"])
            _, gt_cls_ir = self.ir_encoder(batch["ir_spectrum"])
        gt_tokens = [gt_cls_1h, gt_cls_13c, gt_cls_ir]

        cls_tokens, modality_mask = self.encode_modalities(batch)

        # 2. CAPS: complete missing modalities
        completed_tokens, caps_details = self.caps(cls_tokens, modality_mask)

        # 3. Gated fusion
        fused_repr = self.fusion(completed_tokens)  # (B, D)

        # 4. Molecular embedding
        mol_repr = self.mol_encoder(batch["mol_fp"])  # (B, D)

        # 5. Calibration loss
        calib_loss = caps_calibration_loss(caps_details, gt_tokens)

        return {
            "fused_repr": fused_repr,
            "mol_repr": mol_repr,
            "cls_tokens": completed_tokens,
            "modality_mask": modality_mask,
            "calib_loss": calib_loss,
            "caps_details": caps_details,
        }

    def encode_spectra(self, batch: dict) -> torch.Tensor:
        """Encode spectra for retrieval (inference only)."""
        self.eval()
        with torch.no_grad():
            cls_tokens, modality_mask = self.encode_modalities(batch)
            completed_tokens, _ = self.caps(cls_tokens, modality_mask)
            fused = self.fusion(completed_tokens)
        return fused


def build_model(cfg: dict) -> tuple[NMRMultiFuse, NMRMultiFuseLoss]:
    """Build model and loss from config dict."""
    model = NMRMultiFuse(
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
        calib_weight=cfg.get("calib_weight", 0.1),
        learnable_temp=cfg.get("learnable_temp", False),
    )

    return model, criterion
