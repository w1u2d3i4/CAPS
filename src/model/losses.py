"""
Contrastive learning losses for NMR-MultiFuse v3.

  - SpectMolContrastiveLoss: align fused spectra repr with molecular repr (InfoNCE)
  - InterModalContrastiveLoss: align different modality cls tokens of the same molecule
  - CAPSCalibrationLoss: wrapper around caps_calibration_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectMolContrastiveLoss(nn.Module):
    """
    InfoNCE loss: align fused spectral representation with molecular representation.

    Positive pairs: (spectra_i, molecule_i)
    Negative pairs: all other molecules in the batch
    """

    def __init__(self, temperature: float = 0.07, learnable_temp: bool = False):
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def forward(self, spec_repr: torch.Tensor, mol_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec_repr: (B, D) normalized spectral embeddings
            mol_repr:  (B, D) normalized molecular embeddings
        Returns:
            loss: scalar
        """
        spec_repr = F.normalize(spec_repr, dim=-1)
        mol_repr = F.normalize(mol_repr, dim=-1)

        # Similarity matrix
        logits = spec_repr @ mol_repr.T / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_s2m = F.cross_entropy(logits, labels)
        loss_m2s = F.cross_entropy(logits.T, labels)
        return (loss_s2m + loss_m2s) / 2


class InterModalContrastiveLoss(nn.Module):
    """
    InfoNCE loss between different modality cls tokens of the same molecule.

    For each pair of available modalities (i, j):
      positive: (cls_i[k], cls_j[k])  same molecule k
      negative: (cls_i[k], cls_j[l])  different molecules
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.register_buffer("log_temp", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def forward(
        self,
        cls_tokens: list[torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_tokens: list of 3 tensors (B, D)
            modality_mask: (B, 3) bool
        Returns:
            loss: scalar
        """
        num_mod = len(cls_tokens)
        losses = []

        for i in range(num_mod):
            for j in range(i + 1, num_mod):
                # Find samples where both modalities are available
                both_avail = modality_mask[:, i] & modality_mask[:, j]  # (B,)
                if both_avail.sum() < 2:
                    continue

                a = F.normalize(cls_tokens[i][both_avail], dim=-1)
                b = F.normalize(cls_tokens[j][both_avail], dim=-1)

                logits = a @ b.T / self.temperature
                labels = torch.arange(logits.size(0), device=logits.device)

                loss_ab = F.cross_entropy(logits, labels)
                loss_ba = F.cross_entropy(logits.T, labels)
                losses.append((loss_ab + loss_ba) / 2)

        if not losses:
            return torch.tensor(0.0, device=cls_tokens[0].device)
        return torch.stack(losses).mean()


class NMRMultiFuseLoss(nn.Module):
    """
    Combined loss for NMR-MultiFuse v3.

    L = L_spect_mol + w_inter * L_inter_modal + w_calib * L_calib
    """

    def __init__(
        self,
        temperature: float = 0.07,
        inter_weight: float = 1.0,
        calib_weight: float = 0.1,
        learnable_temp: bool = False,
    ):
        super().__init__()
        self.spect_mol_loss = SpectMolContrastiveLoss(temperature, learnable_temp)
        self.inter_modal_loss = InterModalContrastiveLoss(temperature)
        self.inter_weight = inter_weight
        self.calib_weight = calib_weight

    def forward(
        self,
        fused_repr: torch.Tensor,
        mol_repr: torch.Tensor,
        cls_tokens: list[torch.Tensor],
        modality_mask: torch.Tensor,
        calib_loss: torch.Tensor,
    ) -> dict:
        l_sm = self.spect_mol_loss(fused_repr, mol_repr)
        l_inter = self.inter_modal_loss(cls_tokens, modality_mask)
        l_calib = calib_loss

        total = l_sm + self.inter_weight * l_inter + self.calib_weight * l_calib

        return {
            "loss": total,
            "loss_spect_mol": l_sm.item(),
            "loss_inter_modal": l_inter.item(),
            "loss_calib": l_calib.item(),
        }
