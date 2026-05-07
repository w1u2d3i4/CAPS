"""
Gated fusion module for combining multi-modal cls tokens.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated fusion of modality cls tokens.

    Input: list of 3 cls tokens (B, D), each possibly CAPS-completed
    Output: fused representation (B, D)
    """

    def __init__(self, dim: int = 256, num_modalities: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, dim),
            nn.ReLU(),
            nn.Linear(dim, num_modalities),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.num_modalities = num_modalities

    def forward(self, cls_tokens: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            cls_tokens: list of num_modalities tensors, each (B, D)
        Returns:
            fused: (B, D)
        """
        stacked = torch.stack(cls_tokens, dim=1)  # (B, M, D)
        concat = torch.cat(cls_tokens, dim=-1)      # (B, M*D)

        weights = self.gate(concat)  # (B, M)
        weights = weights.unsqueeze(-1)  # (B, M, 1)

        fused = (weights * stacked).sum(dim=1)  # (B, D)
        fused = self.out_proj(fused)
        return fused
