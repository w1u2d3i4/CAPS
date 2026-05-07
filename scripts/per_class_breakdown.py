#!/usr/bin/env python3
"""Per-functional-group top-1 retrieval on scaffold-split test set.

Buckets each test molecule by RDKit functional group flags, then reports
top-1 per bucket for both scaffold-trained checkpoints under S1 (full)
and S4 (1H+13C, no IR) — the two most informative scenarios.
"""
import json, sys, os
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Fragments
from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from torch.utils.data import DataLoader

CKPTS = {
    "scaffold_spectre_dropout": "checkpoints/scaffold_spectre_dropout_100ep/best.pt",
    "scaffold_full_caps":       "checkpoints/scaffold_full_caps_100ep/best.pt",
}

# RDKit fragment counters → coarse functional groups
FG_FUNCS = {
    "alcohol":    Fragments.fr_Al_OH,
    "phenol":     Fragments.fr_phenol,
    "ester":      Fragments.fr_ester,
    "carboxylic": Fragments.fr_COO,
    "amide":      Fragments.fr_amide,
    "amine_pri":  Fragments.fr_NH2,
    "amine_sec":  Fragments.fr_NH1,
    "aromatic_N": Fragments.fr_pyridine,
    "halogen":    Fragments.fr_halogen,
    "nitro":      Fragments.fr_nitro,
    "sulfide":    Fragments.fr_sulfide,
    "ether":      Fragments.fr_ether,
    "ketone":     Fragments.fr_ketone,
    "aldehyde":   Fragments.fr_aldehyde,
}


def fg_buckets(smiles_list):
    """Returns dict: fg_name -> indices (in test set order). Each mol may be in multiple buckets."""
    buckets = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
        except Exception:
            continue
        for fg, func in FG_FUNCS.items():
            try:
                if func(m) > 0:
                    buckets[fg].append(i)
            except Exception:
                pass
    return buckets


@torch.no_grad()
def encode(model, dl, device, mask_override):
    model.eval()
    if hasattr(model, "mask_prob"):
        model.mask_prob = 0.0
    all_s, all_m, all_smi = [], [], []
    for b in dl:
        b_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        B = b_dev["mol_fp"].size(0)
        b_dev["modality_mask"] = torch.tensor([mask_override]*B, dtype=torch.bool, device=device)
        out = model(b_dev)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
        all_smi.extend(b["smiles"])
    return torch.cat(all_s), torch.cat(all_m), all_smi


def top1_indices(spec, mol):
    """Return for each query whether top-1 retrieval is correct."""
    N = spec.size(0)
    correct = np.zeros(N, dtype=bool)
    for i in range(0, N, 1000):
        end = min(i+1000, N)
        sim = spec[i:end] @ mol.T
        diag = sim[torch.arange(end-i), torch.arange(i, end)].unsqueeze(1)
        ranks = (sim >= diag).sum(1).cpu().numpy()
        correct[i:end] = ranks <= 1
    return correct


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FastMultimodalDataset("data/processed", split="test",
                               split_file="scaffold_split.json", max_samples=10000)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2, collate_fn=fast_collate, pin_memory=True)

    out = {}
    SCENARIOS = {"S1_full": [True, True, True], "S4_1H_13C": [True, True, False]}

    # Compute buckets once from the dataset's smiles in order
    print("Computing functional-group buckets…")
    smiles_in_order = [ds.smiles[i] for i in ds.indices]
    buckets = fg_buckets(smiles_in_order)
    print(f"  buckets: {[(k, len(v)) for k,v in buckets.items()]}")

    for ckpt_name, ckpt_path in CKPTS.items():
        print(f"\n=== {ckpt_name} ===")
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
        for sid, mask in SCENARIOS.items():
            spec, mol, smi = encode(model, dl, device, mask)
            correct = top1_indices(spec, mol)
            overall = float(correct.mean())
            scen_out = {"overall_top1": round(overall, 4), "n": int(len(correct))}
            for fg, idx_list in buckets.items():
                if len(idx_list) >= 30:
                    sub = correct[np.array(idx_list)]
                    scen_out[fg] = {"top1": round(float(sub.mean()), 4), "n": int(len(sub))}
            ckpt_out[sid] = scen_out
            print(f"  {sid} overall={overall:.4f}")
        out[ckpt_name] = ckpt_out

    with open("results/per_class_scaffold.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/per_class_scaffold.json")


if __name__ == "__main__":
    main()
