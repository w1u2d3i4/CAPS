"""
CAPS: Confidence-Aware Adaptive Projection-Proxy Switching.

Core innovation of NMR-MultiFuse v3.
For each missing modality:
  1. MMP path: cross-attention projection from available modalities
  2. Confidence: entropy of attention weights
  3. Gate: learned switch between MMP projection and static proxy token
  4. Calibration loss: confidence should correlate with projection quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAPS(nn.Module):
    """
    Confidence-Aware Adaptive Projection-Proxy Switching.

    Args:
        dim: embedding dimension
        num_modalities: number of modalities (default 3: 1H, 13C, IR)
        nhead: attention heads for cross-attention
        dropout: dropout rate
    """

    def __init__(self, dim: int = 256, num_modalities: int = 3, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.nhead = nhead

        # Learnable proxy tokens: one per modality, with a refinement MLP
        self.proxy_tokens = nn.Parameter(torch.randn(num_modalities, dim) * 0.02)
        self.proxy_refine = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Cross-attention: project missing modality from available ones
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)

        # Gate network: confidence + context -> alpha
        # Input: [confidence_score, num_available/num_total, info_score]
        self.gate_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Initialize gate bias so alpha starts at ~0.3 (favor proxy initially,
        # let the model learn when projection is trustworthy)
        nn.init.constant_(self.gate_net[-1].bias, -0.85)  # sigmoid(-0.85) ≈ 0.3

    def compute_confidence(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute projection confidence from attention weight entropy.

        Args:
            attn_weights: (B, nhead, 1, num_kv) attention weights
        Returns:
            confidence: (B,) in [0, 1], higher = more confident
        """
        # Average over heads
        w = attn_weights.mean(dim=1).squeeze(2)  # (B, num_kv)
        # Entropy
        eps = 1e-8
        entropy = -(w * (w + eps).log()).sum(dim=-1)  # (B,)
        max_entropy = torch.log(torch.tensor(w.size(-1), dtype=torch.float, device=w.device))
        # Confidence = 1 - normalized entropy
        confidence = 1.0 - entropy / (max_entropy + eps)
        return confidence.clamp(0, 1)

    def forward(
        self,
        cls_tokens: list[torch.Tensor],
        modality_mask: torch.Tensor,
        return_details: bool = False,
    ) -> tuple[list[torch.Tensor], dict]:
        """
        Complete missing modalities via CAPS.

        Args:
            cls_tokens: list of 3 tensors, each (B, D) or None if modality unavailable.
                        Actually always provided but may be zero vectors for missing modalities.
            modality_mask: (B, 3) bool — True if modality is available
        Returns:
            completed_tokens: list of 3 tensors (B, D), all filled in
            details: dict with confidence scores, alphas, calibration info
        """
        B = modality_mask.size(0)
        device = modality_mask.device
        completed = list(cls_tokens)  # copy the list
        details = {"confidences": [], "alphas": [], "proj_tokens": [], "modality_indices": []}

        for m in range(self.num_modalities):
            # Find samples where modality m is missing
            missing = ~modality_mask[:, m]  # (B,) bool
            if not missing.any():
                continue

            # Collect available modality tokens for missing samples
            # Stack available cls tokens: (B_missing, num_avail, D)
            available_indices = [j for j in range(self.num_modalities) if j != m]
            avail_tokens = []
            avail_mask_per_sample = []

            for b_idx in range(B):
                if not missing[b_idx]:
                    continue
                tokens_b = []
                mask_b = []
                for j in available_indices:
                    tokens_b.append(cls_tokens[j][b_idx])
                    mask_b.append(modality_mask[b_idx, j].item())
                avail_tokens.append(torch.stack(tokens_b))  # (num_other, D)
                avail_mask_per_sample.append(mask_b)

            if not avail_tokens:
                continue

            kv = torch.stack(avail_tokens)  # (B_miss, num_other, D)
            kv_mask = torch.tensor(avail_mask_per_sample, dtype=torch.bool, device=device)
            kv_padding_mask = ~kv_mask  # True = ignore

            # Query: proxy token for modality m
            query = self.proxy_tokens[m].unsqueeze(0).unsqueeze(0).expand(len(avail_tokens), 1, -1)
            # (B_miss, 1, D)

            # Cross-attention
            kv_normed = self.cross_norm(kv)
            proj_token, attn_weights = self.cross_attn(
                query, kv_normed, kv_normed,
                key_padding_mask=kv_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            proj_token = proj_token.squeeze(1)  # (B_miss, D)

            # Confidence
            confidence = self.compute_confidence(attn_weights)  # (B_miss,)

            # Gate inputs — ensure all are 1D (B_miss,)
            num_available = kv_mask.sum(dim=-1).float().view(-1)  # (B_miss,)
            info_score = kv.norm(dim=-1).mean(dim=-1).view(-1)   # (B_miss,)
            confidence = confidence.view(-1)                      # (B_miss,)
            gate_input = torch.stack([
                confidence,
                num_available / self.num_modalities,
                info_score / (info_score.mean() + 1e-8),
            ], dim=-1)  # (B_miss, 3)

            alpha = torch.sigmoid(self.gate_net(gate_input)).squeeze(-1)  # (B_miss,)

            # Refine proxy: condition on available modality context
            proxy_raw = self.proxy_tokens[m].unsqueeze(0).expand(len(avail_tokens), -1)  # (B_miss, D)
            # Average available tokens as context for proxy refinement
            avail_context = (kv * kv_mask.unsqueeze(-1).float()).sum(dim=1) / (kv_mask.sum(dim=-1, keepdim=True).float() + 1e-8)
            proxy_refined = proxy_raw + self.proxy_refine(torch.cat([proxy_raw, avail_context], dim=-1))

            # Mix: alpha * projection + (1-alpha) * refined_proxy
            mixed = alpha.unsqueeze(-1) * proj_token + (1 - alpha).unsqueeze(-1) * proxy_refined

            # Write back into completed tokens
            miss_indices = missing.nonzero(as_tuple=True)[0]
            completed[m] = completed[m].clone()
            completed[m][miss_indices] = mixed

            # Store details for calibration loss
            details["confidences"].append(confidence)
            details["alphas"].append(alpha)
            details["proj_tokens"].append(proj_token)
            details["modality_indices"].append((m, miss_indices))

        return completed, details


def caps_calibration_loss(details: dict, gt_tokens: list[torch.Tensor]) -> torch.Tensor:
    """
    Calibration loss: confidence should positively correlate with projection quality.

    L_calib = -mean(PearsonCorr(confidence, cosine_sim(proj, gt)))

    Only computed during training when gt_tokens are available (before masking).
    """
    if not details["confidences"]:
        return torch.tensor(0.0)

    losses = []
    for conf, proj, (m_idx, b_indices) in zip(
        details["confidences"], details["proj_tokens"], details["modality_indices"]
    ):
        gt = gt_tokens[m_idx][b_indices]  # (B_miss, D)
        # Cosine similarity
        cos_sim = F.cosine_similarity(proj, gt, dim=-1)  # (B_miss,)

        if len(conf) < 3:
            # Not enough samples for correlation
            continue

        # Pearson correlation
        conf_centered = conf - conf.mean()
        sim_centered = cos_sim - cos_sim.mean()
        denom = (conf_centered.norm() * sim_centered.norm() + 1e-8)
        corr = (conf_centered * sim_centered).sum() / denom

        # We want positive correlation -> minimize -corr
        losses.append(-corr)

    if not losses:
        return torch.tensor(0.0, device=gt_tokens[0].device)
    return torch.stack(losses).mean()
