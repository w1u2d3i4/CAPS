#!/usr/bin/env python3
"""E4: Per-functional-group top-1 retrieval on scaffold-split test set,
extended to 5 scenarios (S1 full, S2 1H-only, S3 13C-only, S4 1H+13C, S7 IR-only).

Produces a 13×5 functional-group × scenario heatmap of top-1 retrieval, plus
per-bucket (group, scenario) counts. Reuses the bucket assignment from
scripts/per_class_breakdown.py.

Output: results/per_class_scaffold_5scenarios.json
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
    "scaffold_full_caps":       "checkpoints/scaffold_full_caps_100ep/best.pt",
    "scaffold_spectre_dropout": "checkpoints/scaffold_spectre_dropout_100ep/best.pt",
}

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

SCENARIOS = {
    "S1_full":    [True, True, True],
    "S2_1H_only": [True, False, False],
    "S3_13C_only":[False, True, False],
    "S4_1H_13C":  [True, True, False],
    "S7_IR_only": [False, False, True],
}


def fg_buckets(smiles_list):
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
    all_s, all_m = [], []
    for b in dl:
        b_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        B = b_dev["mol_fp"].size(0)
        b_dev["modality_mask"] = torch.tensor([mask_override]*B, dtype=torch.bool, device=device)
        out = model(b_dev)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    return torch.cat(all_s), torch.cat(all_m)


def top1_indices(spec, mol):
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
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2,
                    collate_fn=fast_collate, pin_memory=True)

    print("Computing functional-group buckets…")
    smiles_in_order = [ds.smiles[i] for i in ds.indices]
    buckets = fg_buckets(smiles_in_order)
    print(f"  buckets: {[(k, len(v)) for k,v in buckets.items()]}")

    out = {}
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
            spec, mol = encode(model, dl, device, mask)
            correct = top1_indices(spec, mol)
            overall = float(correct.mean())
            scen_out = {"overall_top1": round(overall, 4), "n": int(len(correct))}
            for fg, idx_list in buckets.items():
                if len(idx_list) >= 30:
                    sub = correct[np.array(idx_list)]
                    scen_out[fg] = {"top1": round(float(sub.mean()), 4), "n": int(len(sub))}
            ckpt_out[sid] = scen_out
            print(f"  {sid:>14s} overall={overall:.4f}")
        out[ckpt_name] = ckpt_out

    out_path = "results/per_class_scaffold_5scenarios.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
