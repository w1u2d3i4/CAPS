#!/usr/bin/env python3
"""Analyze CAPS gate values (alpha) and confidence across MMS-Bench scenarios."""

import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from src.model.model import build_model
from torch.utils.data import DataLoader

MMS_SCENARIOS = {
    "S1": [True, True, True],
    "S2": [True, False, False],
    "S3": [False, True, False],
    "S4": [True, True, False],
    "S5": [True, False, True],
    "S6": [False, True, True],
    "S7": [False, False, True],
}


@torch.no_grad()
def collect_gate_values(model, dataloader, device, modality_override):
    model.eval()
    all_alphas = {0: [], 1: [], 2: []}  # per modality
    all_confs = {0: [], 1: [], 2: []}

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        B = batch["mol_fp"].size(0)
        batch["modality_mask"] = torch.tensor([modality_override] * B, dtype=torch.bool, device=device)

        old_mp = model.mask_prob
        model.mask_prob = 0.0

        # Encode
        cls_tokens, mod_mask = model.encode_modalities(batch)
        completed, details = model.caps(cls_tokens, mod_mask)

        model.mask_prob = old_mp

        for conf, alpha, (m_idx, b_indices) in zip(
            details.get("confidences", []),
            details.get("alphas", []),
            details.get("modality_indices", [])
        ):
            all_alphas[m_idx].extend(alpha.cpu().numpy().tolist())
            all_confs[m_idx].extend(conf.cpu().numpy().tolist())

    return all_alphas, all_confs


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "checkpoints/autoresearch/exp_mask0.5_calib0.1_temp0.07_inter1.0_lr2e-4/best.pt"
    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model, _ = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    dataset = FastMultimodalDataset("data/processed", split="test", max_samples=5000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2,
                           collate_fn=fast_collate, pin_memory=True)

    modality_names = {0: "1H-NMR", 1: "13C-NMR", 2: "IR"}
    results = {}

    for scenario_id, mask in MMS_SCENARIOS.items():
        print(f"  {scenario_id}: {mask}")
        alphas, confs = collect_gate_values(model, dataloader, device, mask)
        results[scenario_id] = {"alphas": alphas, "confs": confs}

    # Generate figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("CAPS Gate Values (alpha) Across Missing-Modality Scenarios", fontsize=12)

    for i, (sid, mask) in enumerate(MMS_SCENARIOS.items()):
        if i >= 7:
            break
        row, col = i // 4, i % 4
        ax = axes[row][col]

        for m in range(3):
            vals = results[sid]["alphas"][m]
            if vals:
                ax.hist(vals, bins=30, alpha=0.6, label=modality_names[m], density=True)

        ax.set_title(f"{sid}: {['1H','13C','IR'][0] if mask[0] else ''}"
                     f"{'+13C' if mask[1] else ''}{'+IR' if mask[2] else ''}".replace('++','+').strip('+'),
                     fontsize=9)
        ax.set_xlabel("alpha", fontsize=8)
        ax.set_xlim(0, 1)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)

    # Hide unused subplot
    axes[1][3].axis('off')

    # Summary stats
    summary_text = "Mean alpha per (scenario, missing modality):\n"
    for sid in ["S2", "S3", "S7"]:
        mask = MMS_SCENARIOS[sid]
        for m in range(3):
            if not mask[m]:  # this modality is missing
                vals = results[sid]["alphas"][m]
                if vals:
                    summary_text += f"  {sid}, {modality_names[m]}: alpha={np.mean(vals):.3f} (std={np.std(vals):.3f})\n"
    axes[1][3].text(0.1, 0.5, summary_text, fontsize=8, transform=axes[1][3].transAxes,
                    verticalalignment='center', family='monospace')

    plt.tight_layout()
    plt.savefig("paper/figures/gate_analysis.pdf", dpi=300)
    plt.savefig("paper/figures/gate_analysis.png", dpi=300)
    print("Saved: paper/figures/gate_analysis.pdf")

    # Save JSON
    summary = {}
    for sid in MMS_SCENARIOS:
        mask = MMS_SCENARIOS[sid]
        summary[sid] = {}
        for m in range(3):
            vals = results[sid]["alphas"][m]
            if vals:
                summary[sid][modality_names[m]] = {
                    "mean_alpha": round(float(np.mean(vals)), 4),
                    "std_alpha": round(float(np.std(vals)), 4),
                    "mean_conf": round(float(np.mean(results[sid]["confs"][m])), 4),
                    "n_samples": len(vals),
                    "is_missing": not mask[m],
                }
    with open("results/gate_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: results/gate_analysis.json")


if __name__ == "__main__":
    main()
