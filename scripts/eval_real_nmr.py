#!/usr/bin/env python3
"""
Zero-shot eval Exp_SD1 (best.pt) on real-world NMR data (SDBS + nmrshiftdb2).
Reports top-1 / top-5 / top-10 retrieval for S2 (1H only), S3 (13C only),
S4 (1H+13C, no IR) — the scenarios applicable to data without IR.
"""
import json, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from src.model.model import build_model
from torch.utils.data import DataLoader

CKPT = "checkpoints/autoresearch/exp_SD1_spectre_dropout_mask0.5_ep200_lr2e-4/best.pt"
DATA_DIR = "data/real_nmr"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
print(f"Loading {CKPT}...")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = ckpt.get("config", {})
ablation = ckpt.get("args", {}).get("ablation") or cfg.get("ablation") or "spectre_dropout"
print(f"Ablation: {ablation}")
from src.model.model_ablation import build_ablation_model
model, _ = build_ablation_model(ablation, cfg)
model.load_state_dict(ckpt["model"], strict=False)
model = model.to(device).eval()

# Load real data dataset (no train/val/test split — use everything as test pool)
class RealEvalDataset(torch.utils.data.Dataset):
    def __init__(self, root, indices=None):
        root = Path(root)
        self.h_peaks = np.load(root / "h_nmr_peaks.npy", mmap_mode="r")
        self.h_lens = np.load(root / "h_nmr_lengths.npy", mmap_mode="r")
        self.c_peaks = np.load(root / "c_nmr_peaks.npy", mmap_mode="r")
        self.c_lens = np.load(root / "c_nmr_lengths.npy", mmap_mode="r")
        self.ir = np.load(root / "ir_spectra.npy", mmap_mode="r")
        self.fps = np.load(root / "mol_fps.npy", mmap_mode="r")
        with open(root / "smiles.json") as f:
            self.smiles = json.load(f)
        with open(root / "sources.json") as f:
            self.sources = json.load(f)
        self.indices = indices if indices is not None else np.arange(len(self.smiles))
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        h_len, c_len = int(self.h_lens[i]), int(self.c_lens[i])
        return {
            "h_nmr_peaks": np.array(self.h_peaks[i, :max(h_len,1)]),
            "c_nmr_peaks": np.array(self.c_peaks[i, :max(c_len,1)]),
            "ir_spectrum": np.array(self.ir[i]),
            "mol_fp": np.array(self.fps[i]),
            "smiles": self.smiles[i],
            "modality_mask": np.array([h_len>0, c_len>0, False], dtype=np.bool_),  # IR always missing
        }

with open(Path(DATA_DIR) / "smiles.json") as f:
    all_smiles = json.load(f)
h_lens = np.load(Path(DATA_DIR) / "h_nmr_lengths.npy")
c_lens = np.load(Path(DATA_DIR) / "c_nmr_lengths.npy")
N = len(all_smiles)
print(f"Total samples: {N}")

# Define subsets
both_idx = np.where((h_lens > 0) & (c_lens > 0))[0]
only_h_idx = np.where((h_lens > 0) & (c_lens == 0))[0]
only_c_idx = np.where((h_lens == 0) & (c_lens > 0))[0]
print(f"S4 (both): {len(both_idx)}, S2 (only 1H): {len(only_h_idx)}, S3 (only 13C): {len(only_c_idx)}")

# Deduplicate by canonical SMILES — important for retrieval
print("Deduplicating by canonical SMILES...")
try:
    from rdkit import Chem
    canon_map = {}
    for i, smi in enumerate(all_smiles):
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None: continue
            csmi = Chem.MolToSmiles(m)
            if csmi not in canon_map:
                canon_map[csmi] = i
        except Exception:
            continue
    unique_indices = sorted(canon_map.values())
    print(f"  unique molecules: {len(unique_indices)} (from {N})")
    unique_set = set(unique_indices)
    both_idx = np.array([i for i in both_idx if i in unique_set])
    only_h_idx = np.array([i for i in only_h_idx if i in unique_set])
    only_c_idx = np.array([i for i in only_c_idx if i in unique_set])
    print(f"  S4 unique: {len(both_idx)}, S2 unique: {len(only_h_idx)}, S3 unique: {len(only_c_idx)}")
except Exception as e:
    print(f"  rdkit dedup failed: {e}, using all")

@torch.no_grad()
def encode_subset(indices, scenario_mask):
    """Encode a subset with a fixed modality mask. Returns (spec_emb, mol_emb)."""
    if len(indices) == 0:
        return None, None
    ds = RealEvalDataset(DATA_DIR, indices=indices)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2,
                    collate_fn=fast_collate, pin_memory=True)
    all_s, all_m = [], []
    for b in dl:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in b.items()}
        B = b["mol_fp"].size(0)
        b["modality_mask"] = torch.tensor([scenario_mask]*B, dtype=torch.bool, device=device)
        if hasattr(model, "mask_prob"):
            model.mask_prob = 0.0
        out = model(b)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    return torch.cat(all_s), torch.cat(all_m)

def topk_acc(spec, mol, ks=(1,5,10)):
    N = spec.size(0)
    sim = spec @ mol.T
    diag = sim.diag().unsqueeze(1)
    ranks = (sim >= diag).sum(1)
    return {f"top{k}": ((ranks <= k).float().mean().item()) for k in ks}

print("\n=== S4: 1H + 13C (no IR) ===")
if len(both_idx) > 0:
    s, m = encode_subset(both_idx, [True, True, False])
    print(f"  Pool size: {s.size(0)}")
    res_s4 = topk_acc(s, m)
    print(f"  {res_s4}")
else:
    res_s4 = None

print("\n=== S2: only 1H (no 13C, no IR) ===")
if len(only_h_idx) > 0:
    s, m = encode_subset(only_h_idx, [True, False, False])
    print(f"  Pool size: {s.size(0)}")
    res_s2 = topk_acc(s, m)
    print(f"  {res_s2}")
else:
    res_s2 = None

print("\n=== S3: only 13C (no 1H, no IR) ===")
if len(only_c_idx) > 0:
    s, m = encode_subset(only_c_idx, [False, True, False])
    print(f"  Pool size: {s.size(0)}")
    res_s3 = topk_acc(s, m)
    print(f"  {res_s3}")
else:
    res_s3 = None

# Save
results = {
    "checkpoint": CKPT,
    "data": DATA_DIR,
    "S4_1H+13C_noIR": res_s4,
    "S2_only1H": res_s2,
    "S3_only13C": res_s3,
    "n_S4": int(len(both_idx)),
    "n_S2": int(len(only_h_idx)),
    "n_S3": int(len(only_c_idx)),
}
out = "results/eval_real_nmr_zeroshot.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
