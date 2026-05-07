#!/usr/bin/env python3
"""Q5 (round 3 review): Bootstrap CIs on the FULL 79K test pool
(vs the 20K subsample used in the original paper).

Output: results/q5_bootstrap_79K.json
"""
import json, sys, os
from pathlib import Path
import numpy as np

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

CKPTS = {
    "scaffold_full_caps":       "checkpoints/scaffold_full_caps_100ep/best.pt",
    "scaffold_spectre_dropout": "checkpoints/scaffold_spectre_dropout_100ep/best.pt",
}

N_BOOT = 300
SEED = 42


@torch.no_grad()
def encode(model, dl, device, mask_override):
    model.eval()
    if hasattr(model, "mask_prob"):
        model.mask_prob = 0.0
    all_s, all_m = [], []
    for b in dl:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        B = b["mol_fp"].size(0)
        b["modality_mask"] = torch.tensor([mask_override]*B, dtype=torch.bool, device=device)
        out = model(b)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    return torch.cat(all_s), torch.cat(all_m)


def correct_mask_full79k(spec, mol, device, chunk=2048):
    """Returns boolean array correct[i] = True iff query i is top-1 against the full library."""
    s = spec.to(device); m = mol.to(device)
    N = s.size(0)
    correct = np.zeros(N, dtype=np.int8)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        sim = s[i:end] @ m.T  # (chunk, N)
        diag = sim[torch.arange(end-i), torch.arange(i, end)].unsqueeze(1)
        ranks = (sim >= diag).sum(1)
        correct[i:end] = (ranks <= 1).cpu().numpy()
    return correct


def bootstrap(correct_mask, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    N = len(correct_mask)
    accs = []
    for _ in range(n_boot):
        q = rng.choice(N, N, replace=True)
        accs.append(float(correct_mask[q].mean()))
    return float(np.mean(accs)), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FastMultimodalDataset("data/processed", split="test",
                               split_file="scaffold_split.json", max_samples=None)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2,
                    collate_fn=fast_collate, pin_memory=True)
    print(f"N_test = {len(ds)}")

    out = {"n_test": len(ds), "n_boot": N_BOOT, "results": {}}
    for name, path in CKPTS.items():
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

        scen = {}
        for sid, mask in MMS.items():
            spec, mol = encode(model, dl, device, mask)
            correct = correct_mask_full79k(spec, mol, device)
            mean, lo, hi = bootstrap(correct, N_BOOT)
            half = (hi - lo) / 2
            scen[sid] = {
                "mean_top1": round(mean, 4),
                "ci95_lo":   round(lo, 4),
                "ci95_hi":   round(hi, 4),
                "ci_half":   round(half, 4),
                "n_correct": int(correct.sum()),
            }
            print(f"  {sid}: {mean:.4f} [{lo:.4f}, {hi:.4f}] (±{half:.4f})")
        out["results"][name] = scen

    out_path = "results/q5_bootstrap_79K.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
