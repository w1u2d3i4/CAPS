#!/usr/bin/env python3
"""Compute Bemis-Murcko scaffold split on processed dataset.

Saves train/val/test indices into the processed dir under scaffold_split.json.
Strategy: group by scaffold SMILES, sort groups by size descending, then
distribute groups round-robin into the largest deficit bucket so no scaffold
spans splits. Target ratios 80/10/10.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def scaffold_smiles(smi: str) -> str:
    if not smi:
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaf or ""
    except Exception:
        return ""


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    proc = Path("data/processed")
    smiles = json.load(open(proc / "smiles.json"))
    n = len(smiles)
    print(f"[scaffold] {n} smiles")

    # Compute scaffold for every sample
    groups = defaultdict(list)
    for i, smi in enumerate(tqdm(smiles, desc="scaffolds")):
        scaf = scaffold_smiles(smi)
        groups[scaf].append(i)

    print(f"[scaffold] {len(groups)} unique scaffolds")
    sizes = sorted([len(v) for v in groups.values()], reverse=True)
    print(f"[scaffold] top-10 group sizes: {sizes[:10]}")

    # Sort groups by size desc
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    target = {"train": int(0.8 * n), "val": int(0.1 * n), "test": n - int(0.8 * n) - int(0.1 * n)}
    buckets = {"train": [], "val": [], "test": []}
    deficits = dict(target)

    for grp in sorted_groups:
        # Place into bucket with largest remaining deficit
        bucket = max(deficits, key=lambda k: deficits[k])
        buckets[bucket].extend(grp)
        deficits[bucket] -= len(grp)

    out = {k: sorted(v) for k, v in buckets.items()}
    out["meta"] = {
        "method": "bemis_murcko",
        "total": n,
        "n_train": len(out["train"]),
        "n_val": len(out["val"]),
        "n_test": len(out["test"]),
        "n_scaffolds": len(groups),
    }
    out_path = proc / "scaffold_split.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"[scaffold] saved {out_path}")
    print(f"  train={out['meta']['n_train']}  val={out['meta']['n_val']}  test={out['meta']['n_test']}")


if __name__ == "__main__":
    main()
