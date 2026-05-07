#!/usr/bin/env python3
"""Bootstrap CI on scaffold-split test set for both scaffold-trained checkpoints."""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import torch
import torch.nn.functional as F
from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from src.model.model import build_model
from torch.utils.data import DataLoader

MMS = {
    "S1": [True, True, True], "S2": [True, False, False],
    "S3": [False, True, False], "S4": [True, True, False],
    "S5": [True, False, True], "S6": [False, True, True],
    "S7": [False, False, True],
}


@torch.no_grad()
def encode(model, dl, device, mask_override):
    model.eval()
    all_s, all_m = [], []
    for b in dl:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        B = b["mol_fp"].size(0)
        b["modality_mask"] = torch.tensor([mask_override]*B, dtype=torch.bool, device=device)
        if hasattr(model, "mask_prob"):
            model.mask_prob = 0.0
        out = model(b)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    return torch.cat(all_s), torch.cat(all_m)


def top1_acc(spec, mol):
    N = spec.size(0)
    correct = 0
    for i in range(0, N, 1000):
        end = min(i+1000, N)
        sim = spec[i:end] @ mol.T
        ranks = (sim >= sim[torch.arange(end-i), torch.arange(i,end)].unsqueeze(1)).sum(1)
        correct += (ranks <= 1).sum().item()
    return correct / N


def bootstrap(spec, mol, n_boot=300, seed=42, device="cuda"):
    """Bootstrap CI by resampling QUERIES against a FIXED full library (GPU)."""
    rng = np.random.RandomState(seed)
    N = spec.size(0)
    spec_g = spec.to(device)
    mol_g  = mol.to(device)
    sim_full = spec_g @ mol_g.T  # (N, N)
    diag_full = sim_full.diag()
    # rank_i = #candidates with sim(i, j) >= sim(i, i)
    ranks_full = (sim_full >= diag_full.unsqueeze(1)).sum(1)  # (N,) on GPU
    correct_mask = (ranks_full <= 1).cpu().numpy().astype(np.int8)  # (N,)
    accs = []
    for _ in range(n_boot):
        q = rng.choice(N, N, replace=True)
        accs.append(float(correct_mask[q].mean()))
    return float(np.mean(accs)), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {
        "scaffold_spectre_dropout": "checkpoints/scaffold_spectre_dropout_100ep/best.pt",
        "scaffold_full_caps":       "checkpoints/scaffold_full_caps_100ep/best.pt",
    }
    out_path = "results/bootstrap_scaffold.json"
    out = {}
    for name, path in runs.items():
        print(f"\n=== {name} ===")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        ablation = ckpt.get("args", {}).get("ablation") or cfg.get("ablation")
        if ablation:
            from src.model.model_ablation import build_ablation_model
            model, _ = build_ablation_model(ablation, cfg)
        else:
            model, _ = build_model(cfg)
        model.load_state_dict(ckpt["model"], strict=False)
        model = model.to(device)

        ds = FastMultimodalDataset("data/processed", split="test",
                                   split_file="scaffold_split.json", max_samples=20000)
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2, collate_fn=fast_collate, pin_memory=True)

        results = {}
        for sid, mask in MMS.items():
            spec, mol = encode(model, dl, device, mask)
            mean, lo, hi = bootstrap(spec, mol, n_boot=300)
            ci = (hi - lo) / 2
            print(f"  {sid}: {mean:.4f} [{lo:.4f}, {hi:.4f}] (±{ci:.4f})")
            results[sid] = {"mean": round(mean,4), "ci95_lo": round(lo,4), "ci95_hi": round(hi,4), "ci_half": round(ci,4)}
        out[name] = results

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
