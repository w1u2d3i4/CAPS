#!/usr/bin/env python3
"""Zero-shot real-NMR (nmrshiftdb2 + SDBS) eval with both scaffold-trained checkpoints."""
import json, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from torch.utils.data import DataLoader

DATA_DIR = "data/real_nmr"

CKPTS = {
    "scaffold_spectre_dropout": "checkpoints/scaffold_spectre_dropout_100ep/best.pt",
    "scaffold_full_caps":       "checkpoints/scaffold_full_caps_100ep/best.pt",
}


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


def collate(batch):
    from src.data.fast_dataset import fast_collate
    return fast_collate(batch)


@torch.no_grad()
def encode_with_mask(model, dl, device, mask_override):
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


def topk_acc(spec, mol, ks=(1, 5, 10)):
    N = spec.size(0)
    out = {k: 0 for k in ks}
    for i in range(0, N, 1000):
        end = min(i+1000, N)
        sim = spec[i:end] @ mol.T
        diag = sim[torch.arange(end-i), torch.arange(i, end)].unsqueeze(1)
        ranks = (sim >= diag).sum(1)
        for k in ks:
            out[k] += (ranks <= k).sum().item()
    return {k: out[k] / N for k in ks}


def dedup_indices(smiles_list, h_lens, c_lens):
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
    keep = sorted(canon.values())
    print(f"  dedup: {len(smiles_list)} -> {len(keep)}")
    return np.array(keep, dtype=np.int64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load
    with open(Path(DATA_DIR) / "smiles.json") as f:
        all_smiles = json.load(f)
    h_lens = np.load(Path(DATA_DIR) / "h_nmr_lengths.npy")
    c_lens = np.load(Path(DATA_DIR) / "c_nmr_lengths.npy")
    N = len(all_smiles)
    print(f"Total real NMR samples: {N}")

    # Dedup
    keep = dedup_indices(all_smiles, h_lens, c_lens)
    h_lens_d = h_lens[keep]
    c_lens_d = c_lens[keep]

    # S2/S3/S4 subset masks
    both_idx  = keep[(h_lens_d > 0) & (c_lens_d > 0)]
    only_h    = keep[(h_lens_d > 0) & (c_lens_d == 0)]
    only_c    = keep[(h_lens_d == 0) & (c_lens_d > 0)]
    print(f"  S4(1H+13C): {len(both_idx)}, S2(1H only): {len(only_h)}, S3(13C only): {len(only_c)}")

    out = {}
    for name, ckpt_path in CKPTS.items():
        print(f"\n=== {name} ===")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        ablation = ckpt.get("args", {}).get("ablation") or cfg.get("ablation")
        if ablation:
            from src.model.model_ablation import build_ablation_model
            model, _ = build_ablation_model(ablation, cfg)
        else:
            from src.model.model import build_model
            model, _ = build_model(cfg)
        model.load_state_dict(ckpt["model"], strict=False)
        model = model.to(device)

        ckpt_results = {}
        for sid, idx_set, mask in [
            ("S2_1h_only",  only_h,   [True, False, False]),
            ("S3_13c_only", only_c,   [False, True, False]),
            ("S4_1h_13c",   both_idx, [True, True, False]),
        ]:
            if len(idx_set) < 50:
                print(f"  skip {sid}: only {len(idx_set)} samples")
                continue
            ds = RealEvalDataset(DATA_DIR, indices=idx_set)
            dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)
            spec, mol = encode_with_mask(model, dl, device, mask)
            ks = topk_acc(spec, mol)
            print(f"  {sid} (N={len(idx_set)}): top1={ks[1]:.4f} top5={ks[5]:.4f} top10={ks[10]:.4f}")
            ckpt_results[sid] = {"N": int(len(idx_set)), **{f"top{k}": round(v, 4) for k, v in ks.items()}}
        out[name] = ckpt_results

    with open("results/eval_real_nmr_scaffold.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/eval_real_nmr_scaffold.json")


if __name__ == "__main__":
    main()
