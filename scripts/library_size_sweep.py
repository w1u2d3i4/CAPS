#!/usr/bin/env python3
"""E3: Library-size sensitivity sweep on the scaffold-split test set.

For the scaffold-trained Full CAPS checkpoint, encode the entire test set once
under each scenario in {S1 full, S4 1H+13C, S7 IR-only}, then for each library
size L in {1000, 5000, 20000, 79000}, randomly subsample L test molecules to
form the candidate pool and compute top-1 / top-5 retrieval over that pool.
Average over 5 random subsamples per (scenario, L).

Output: results/library_size_sweep.json
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

CKPT_PATH = "checkpoints/scaffold_full_caps_100ep/best.pt"

SCENARIOS = {
    "S1_full":    [True, True, True],
    "S4_1H_13C":  [True, True, False],
    "S7_IR_only": [False, False, True],
}

LIB_SIZES = [1000, 5000, 20000, 79000]
N_REPEATS = 5
SEED = 42


@torch.no_grad()
def encode_all(model, dl, device, mask_override):
    model.eval()
    if hasattr(model, "mask_prob"):
        model.mask_prob = 0.0
    all_s, all_m = [], []
    for b in dl:
        b_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        B = b_dev["mol_fp"].size(0)
        b_dev["modality_mask"] = torch.tensor([mask_override]*B, dtype=torch.bool, device=device)
        out = model(b_dev)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    return torch.cat(all_s), torch.cat(all_m)


def topk_subset(spec, mol, idx_subset, device, k_list=(1, 5), chunk=2048):
    """top-k retrieval where queries = library = the molecules in idx_subset.
    Chunks queries to avoid an L×L similarity matrix on GPU."""
    s = spec[idx_subset].to(device)  # (L, d)
    m = mol[idx_subset].to(device)   # (L, d)
    L = s.size(0)
    correct = {k: 0 for k in k_list}
    for i in range(0, L, chunk):
        end = min(i + chunk, L)
        sim_chunk = s[i:end] @ m.T               # (chunk, L)
        diag_chunk = sim_chunk[torch.arange(end - i), torch.arange(i, end)].unsqueeze(1)
        ranks_chunk = (sim_chunk >= diag_chunk).sum(1)  # (chunk,)
        for k in k_list:
            correct[k] += int((ranks_chunk <= k).sum().item())
    out = {f"top{k}": correct[k] / L for k in k_list}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model, _ = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)

    print("Loading scaffold test set (full)…")
    ds = FastMultimodalDataset("data/processed", split="test",
                               split_file="scaffold_split.json", max_samples=None)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2,
                    collate_fn=fast_collate, pin_memory=True)
    N = len(ds)
    print(f"  N_test = {N}")

    rng = np.random.RandomState(SEED)
    out = {"checkpoint": CKPT_PATH, "n_test": N, "n_repeats": N_REPEATS,
           "lib_sizes": LIB_SIZES, "scenarios": {}}

    for sid, mask in SCENARIOS.items():
        print(f"\n=== {sid} ===")
        spec, mol = encode_all(model, dl, device, mask)  # (N, d) on CPU
        scen = {}
        for L in LIB_SIZES:
            L_eff = min(L, N)
            top1s, top5s = [], []
            for r in range(N_REPEATS):
                idx = rng.choice(N, L_eff, replace=False)
                m = topk_subset(spec, mol, idx, device)
                top1s.append(m["top1"]); top5s.append(m["top5"])
            mean_top1 = float(np.mean(top1s)); std_top1 = float(np.std(top1s))
            mean_top5 = float(np.mean(top5s)); std_top5 = float(np.std(top5s))
            scen[str(L_eff)] = {
                "lib_size": L_eff,
                "top1_mean": round(mean_top1, 4), "top1_std": round(std_top1, 4),
                "top5_mean": round(mean_top5, 4), "top5_std": round(std_top5, 4),
            }
            print(f"  L={L_eff:>5d}: top1={mean_top1:.4f}±{std_top1:.4f}  top5={mean_top5:.4f}±{std_top5:.4f}")
        out["scenarios"][sid] = scen

    out_path = "results/library_size_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
