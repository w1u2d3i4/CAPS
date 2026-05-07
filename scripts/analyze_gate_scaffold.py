#!/usr/bin/env python3
"""Analyze CAPS gate values on scaffold-trained CAPS checkpoint."""
import json, sys, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from src.model.model import build_model
from torch.utils.data import DataLoader

MMS = {
    "S1": [True, True, True], "S2": [True, False, False],
    "S3": [False, True, False], "S4": [True, True, False],
    "S5": [True, False, True], "S6": [False, True, True],
    "S7": [False, False, True],
}
NAMES = {0: "1H", 1: "13C", 2: "IR"}


@torch.no_grad()
def collect(model, dl, device, mask_override):
    model.eval()
    alphas = {0: [], 1: [], 2: []}
    confs  = {0: [], 1: [], 2: []}
    for b in dl:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        B = b["mol_fp"].size(0)
        b["modality_mask"] = torch.tensor([mask_override]*B, dtype=torch.bool, device=device)
        old = model.mask_prob
        model.mask_prob = 0.0
        cls_tokens, mod_mask = model.encode_modalities(b)
        completed, details = model.caps(cls_tokens, mod_mask)
        model.mask_prob = old
        for conf, alpha, (m_idx, _) in zip(
            details.get("confidences", []),
            details.get("alphas", []),
            details.get("modality_indices", [])
        ):
            alphas[m_idx].extend(alpha.cpu().numpy().tolist())
            confs[m_idx].extend(conf.cpu().numpy().tolist())
    return alphas, confs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/scaffold_full_caps_100ep/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model, _ = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)

    ds = FastMultimodalDataset("data/processed", split="test",
                               split_file="scaffold_split.json", max_samples=5000)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, collate_fn=fast_collate, pin_memory=True)

    summary = {"checkpoint": ckpt_path, "n_samples": len(ds)}
    for sid, mask in MMS.items():
        a, c = collect(model, dl, device, mask)
        scen = {}
        for m in range(3):
            if a[m]:
                scen[NAMES[m]] = {
                    "mean_alpha": round(float(np.mean(a[m])), 4),
                    "std_alpha":  round(float(np.std(a[m])),  4),
                    "mean_conf":  round(float(np.mean(c[m])), 4),
                    "is_missing": not mask[m],
                    "n": len(a[m]),
                }
        summary[sid] = scen
        print(f"{sid}: " + ", ".join(f"{NAMES[m]}_a={np.mean(a[m]):.3f}" for m in range(3) if a[m]))

    out = "results/gate_analysis_scaffold.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
