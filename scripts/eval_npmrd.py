#!/usr/bin/env python3
"""Zero-shot eval scaffold checkpoints on NP-MRD predicted spectra."""
import json, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from torch.utils.data import DataLoader, Dataset
from src.data.fast_dataset import fast_collate

CKPTS = {
    "scaffold_spectre_dropout": "checkpoints/scaffold_spectre_dropout_100ep/best.pt",
    "scaffold_full_caps":       "checkpoints/scaffold_full_caps_100ep/best.pt",
}
DATA_DIR = "data/np_mrd_processed"


class NPDataset(Dataset):
    def __init__(self, root, indices=None):
        root = Path(root)
        self.h_peaks = np.load(root/"h_nmr_peaks.npy", mmap_mode="r")
        self.h_lens  = np.load(root/"h_nmr_lengths.npy", mmap_mode="r")
        self.c_peaks = np.load(root/"c_nmr_peaks.npy", mmap_mode="r")
        self.c_lens  = np.load(root/"c_nmr_lengths.npy", mmap_mode="r")
        self.ir      = np.load(root/"ir_spectra.npy", mmap_mode="r")
        self.fps     = np.load(root/"mol_fps.npy", mmap_mode="r")
        with open(root/"smiles.json") as f:
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


@torch.no_grad()
def encode(model, dl, device, mask):
    model.eval()
    if hasattr(model, "mask_prob"):
        model.mask_prob = 0.0
    all_s, all_m = [], []
    for b in dl:
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
    return {f"top{k}": ((ranks <= k).float().mean().item()) for k in ks}


def main():
    device = torch.device("cuda")
    h_lens = np.load(Path(DATA_DIR)/"h_nmr_lengths.npy")
    c_lens = np.load(Path(DATA_DIR)/"c_nmr_lengths.npy")
    has_h = h_lens > 0
    has_c = c_lens > 0
    print(f"Total: {len(h_lens)}, with H: {has_h.sum()}, with C: {has_c.sum()}, with both: {(has_h & has_c).sum()}")

    both = np.where(has_h & has_c)[0]
    only_h = np.where(has_h & ~has_c)[0]
    only_c = np.where(~has_h & has_c)[0]

    out = {"n_total": int(len(h_lens)), "n_h": int(has_h.sum()), "n_c": int(has_c.sum())}
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

        ckpt_out = {}
        for sid, idx, mask in [
            ("S2_1H_only",  only_h, [True, False, False]),
            ("S3_13C_only", only_c, [False, True, False]),
            ("S4_1H_13C",   both,   [True, True, False]),
        ]:
            if len(idx) < 100:
                print(f"  skip {sid}: {len(idx)} samples")
                continue
            # Subsample to 20K for tractable retrieval pool
            if len(idx) > 20000:
                rng = np.random.RandomState(42)
                idx = rng.choice(idx, 20000, replace=False)
            ds = NPDataset(DATA_DIR, indices=idx)
            dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2, collate_fn=fast_collate, pin_memory=True)
            spec, mol = encode(model, dl, device, mask)
            ks = topk(spec, mol)
            print(f"  {sid} (N={len(idx)}): {ks}")
            ckpt_out[sid] = {"N": int(len(idx)), **{k: round(v, 4) for k, v in ks.items()}}
        out[name] = ckpt_out

    with open("results/eval_npmrd.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/eval_npmrd.json")


if __name__ == "__main__":
    main()
