#!/usr/bin/env python3
"""Q10 (round 3 review): Feature normalization on cross-distribution data.

Re-runs cross-distribution evaluation (real NMR) under the scaffold Full CAPS
checkpoint with two pre-processing variants:
  (a) raw features (current paper baseline)
  (b) z-score normalised peak features per channel using statistics from
      the in-distribution scaffold train set.

Tests whether simple distribution-matching at inference recovers any of
the cross-distribution gap WITHOUT label-supervised fine-tuning.

Output: results/q10_feature_norm.json
"""
import json, sys, os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from src.model.model import build_model

CKPT_PATH = "checkpoints/scaffold_full_caps_100ep/best.pt"
REAL_DIR  = "data/real_nmr"


class RealEvalDataset(torch.utils.data.Dataset):
    def __init__(self, root, indices=None):
        root = Path(root)
        self.h_peaks = np.load(root / "h_nmr_peaks.npy", mmap_mode="r")
        self.h_lens  = np.load(root / "h_nmr_lengths.npy", mmap_mode="r")
        self.c_peaks = np.load(root / "c_nmr_peaks.npy", mmap_mode="r")
        self.c_lens  = np.load(root / "c_nmr_lengths.npy", mmap_mode="r")
        self.ir      = np.load(root / "ir_spectra.npy", mmap_mode="r")
        self.fps     = np.load(root / "mol_fps.npy", mmap_mode="r")
        with open(root / "smiles.json") as f:
            self.smiles = json.load(f)
        self.indices = indices if indices is not None else np.arange(len(self.smiles))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        h_len, c_len = int(self.h_lens[i]), int(self.c_lens[i])
        return {
            "h_nmr_peaks": np.array(self.h_peaks[i, :max(h_len,1)]),
            "c_nmr_peaks": np.array(self.c_peaks[i, :max(c_len,1)]),
            "ir_spectrum": np.array(self.ir[i]),
            "mol_fp": np.array(self.fps[i]),
            "smiles": self.smiles[i],
            "modality_mask": np.array([h_len>0, c_len>0, False], dtype=np.bool_),
        }


def dedup_indices(smiles_list):
    from rdkit import Chem
    canon = {}
    for i, smi in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            cs = Chem.MolToSmiles(m)
        except Exception:
            continue
        if cs not in canon:
            canon[cs] = i
    return np.array(sorted(canon.values()), dtype=np.int64)


@torch.no_grad()
def compute_train_stats(n_samples=10000):
    """Compute per-channel mean/std for 1H and 13C peak features on scaffold train."""
    ds = FastMultimodalDataset("data/processed", split="train",
                               split_file="scaffold_split.json", max_samples=n_samples)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, collate_fn=fast_collate)
    h_acc, c_acc = [], []
    for b in dl:
        for hk in ("h_nmr_peaks", "h_peaks"):
            if hk in b and b[hk].numel() > 0:
                h = b[hk].view(-1, b[hk].size(-1))
                h_acc.append(h[h.abs().sum(-1) > 1e-6])
                break
        for ck in ("c_nmr_peaks", "c_peaks"):
            if ck in b and b[ck].numel() > 0:
                c = b[ck].view(-1, b[ck].size(-1))
                c_acc.append(c[c.abs().sum(-1) > 1e-6])
                break
    stats = {}
    if h_acc:
        H = torch.cat(h_acc, 0)
        stats["h_mean"] = H.mean(0).tolist()
        stats["h_std"]  = (H.std(0) + 1e-6).tolist()
    if c_acc:
        C = torch.cat(c_acc, 0)
        stats["c_mean"] = C.mean(0).tolist()
        stats["c_std"]  = (C.std(0) + 1e-6).tolist()
    return stats


def make_normalizer(stats):
    h_mean = torch.tensor(stats["h_mean"]).view(1,1,-1) if "h_mean" in stats else None
    h_std  = torch.tensor(stats["h_std"]).view(1,1,-1)  if "h_std"  in stats else None
    c_mean = torch.tensor(stats["c_mean"]).view(1,1,-1) if "c_mean" in stats else None
    c_std  = torch.tensor(stats["c_std"]).view(1,1,-1)  if "c_std"  in stats else None
    def norm(b):
        for hk in ("h_nmr_peaks", "h_peaks"):
            if hk in b and b[hk].numel() > 0 and h_mean is not None:
                v = (b[hk].abs().sum(-1) > 1e-6).unsqueeze(-1).float()
                b[hk] = ((b[hk] - h_mean.to(b[hk].device)) / h_std.to(b[hk].device)) * v
                break
        for ck in ("c_nmr_peaks", "c_peaks"):
            if ck in b and b[ck].numel() > 0 and c_mean is not None:
                v = (b[ck].abs().sum(-1) > 1e-6).unsqueeze(-1).float()
                b[ck] = ((b[ck] - c_mean.to(b[ck].device)) / c_std.to(b[ck].device)) * v
                break
        return b
    return norm


@torch.no_grad()
def eval_dl(model, dl, device, mask, normalizer=None):
    model.eval()
    if hasattr(model, "mask_prob"): model.mask_prob = 0.0
    all_s, all_m = [], []
    for b in dl:
        if normalizer is not None:
            b = normalizer(b)
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        B = b["mol_fp"].size(0)
        b["modality_mask"] = torch.tensor([mask]*B, dtype=torch.bool, device=device)
        out = model(b)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    return torch.cat(all_s), torch.cat(all_m)


def topk(spec, mol, ks=(1,5,10)):
    N = spec.size(0)
    s = spec.cuda(); m = mol.cuda()
    sim = s @ m.T
    diag = sim.diag().unsqueeze(1)
    ranks = (sim >= diag).sum(1)
    return {f"top{k}": float((ranks <= k).float().mean().item()) for k in ks}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model, _ = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)

    print("Computing in-distribution train statistics ...")
    stats = compute_train_stats(n_samples=10000)
    print(f"  h_mean (first 3): {[round(x,3) for x in stats.get('h_mean', [])[:3]]}")
    print(f"  c_mean (first 3): {[round(x,3) for x in stats.get('c_mean', [])[:3]]}")

    out = {"checkpoint": CKPT_PATH, "stats_n_samples": 10000, "results": {}}

    # Real NMR S4 (1H + 13C)
    if Path(REAL_DIR).exists():
        print(f"\nLoading real NMR from {REAL_DIR} ...")
        ds_full = RealEvalDataset(REAL_DIR)
        keep = dedup_indices(ds_full.smiles)
        h_lens = ds_full.h_lens[keep]; c_lens = ds_full.c_lens[keep]
        s4 = keep[(h_lens > 0) & (c_lens > 0)]
        print(f"  S4 (1H+13C present): {len(s4)} unique mols")
        ds_s4 = RealEvalDataset(REAL_DIR, indices=s4)
        dl_s4 = DataLoader(ds_s4, batch_size=128, shuffle=False, num_workers=2, collate_fn=fast_collate)

        normalizer = make_normalizer(stats)

        for label, n in [("raw", None), ("z_norm", normalizer)]:
            print(f"\n=== Real NMR S4 ({label}) ===")
            spec, mol = eval_dl(model, dl_s4, device, [True, True, False], normalizer=n)
            r = topk(spec, mol)
            print(f"  {r}")
            out["results"].setdefault("real_nmr_S4", {})[label] = r
        # delta
        a = out["results"]["real_nmr_S4"]["raw"]
        b = out["results"]["real_nmr_S4"]["z_norm"]
        out["results"]["real_nmr_S4"]["delta_top1_pp"] = round((b["top1"] - a["top1"]) * 100, 3)
        print(f"  Δtop1 (z_norm − raw): {out['results']['real_nmr_S4']['delta_top1_pp']:+.3f} pp")
    else:
        print(f"  SKIP real NMR (path {REAL_DIR} missing)")

    out_path = "results/q10_feature_norm.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
