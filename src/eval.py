#!/usr/bin/env python3
"""
MMS-Bench evaluation for NMR-MultiFuse v3.

8 scenarios testing missing-modality robustness:
  S1: 1H + 13C + IR  (full)
  S2: 1H only
  S3: 13C only
  S4: 1H + 13C
  S5: 1H + IR
  S6: 13C + IR
  S7: IR only
  S8: incremental (1H -> +13C -> +IR)

Usage:
    python src/eval.py --checkpoint checkpoints/stage3/best.pt --benchmark mms
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from src.model.model import build_model


MMS_SCENARIOS = {
    "S1": {"name": "1H+13C+IR (full)", "mask": [True, True, True]},
    "S2": {"name": "1H only",          "mask": [True, False, False]},
    "S3": {"name": "13C only",         "mask": [False, True, False]},
    "S4": {"name": "1H+13C",           "mask": [True, True, False]},
    "S5": {"name": "1H+IR",            "mask": [True, False, True]},
    "S6": {"name": "13C+IR",           "mask": [False, True, True]},
    "S7": {"name": "IR only",          "mask": [False, False, True]},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--benchmark", default="mms", choices=["mms"])
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--report_json", default="results/mms_bench.json")
    p.add_argument("--output_metrics", action="store_true",
                   help="Output METRIC lines for autoresearch")
    p.add_argument("--split_file", default=None,
                   help="Path to JSON with {train,val,test} index lists")
    return p.parse_args()


@torch.no_grad()
def encode_all(model, dataloader, device, modality_override=None):
    """
    Encode all samples, returns (spec_reprs, mol_reprs).

    modality_override: if set, force modality_mask to this pattern for all samples.
    """
    model.eval()
    all_spec = []
    all_mol = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Override modality mask if specified
        if modality_override is not None:
            B = batch["mol_fp"].size(0)
            batch["modality_mask"] = torch.tensor(
                [modality_override] * B, dtype=torch.bool, device=device
            )

        # Disable training-time random masking
        old_mask_prob = model.mask_prob
        model.mask_prob = 0.0

        out = model(batch)
        spec_repr = F.normalize(out["fused_repr"], dim=-1)
        mol_repr = F.normalize(out["mol_repr"], dim=-1)

        all_spec.append(spec_repr.cpu())
        all_mol.append(mol_repr.cpu())

        model.mask_prob = old_mask_prob

    return torch.cat(all_spec), torch.cat(all_mol)


def retrieval_accuracy(spec_reprs, mol_reprs, top_k_list):
    """
    Compute retrieval accuracy: for each spectrum, rank all molecules by cosine sim.
    """
    # spec_reprs: (N, D), mol_reprs: (N, D)
    # Similarity matrix: (N, N)
    N = spec_reprs.size(0)
    results = {}

    # Process in chunks to avoid OOM on large N
    chunk_size = 1000
    all_ranks = []

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        sim = spec_reprs[i:end] @ mol_reprs.T  # (chunk, N)
        # Rank of the correct molecule (diagonal)
        correct_sims = sim[torch.arange(end - i), torch.arange(i, end)]  # (chunk,)
        # How many molecules have higher similarity than the correct one
        ranks = (sim >= correct_sims.unsqueeze(1)).sum(dim=1)  # (chunk,)
        all_ranks.append(ranks)

    all_ranks = torch.cat(all_ranks).float()

    for k in top_k_list:
        acc = (all_ranks <= k).float().mean().item()
        results[f"top{k}"] = round(acc, 4)

    results["mean_rank"] = round(all_ranks.mean().item(), 1)
    results["median_rank"] = round(all_ranks.median().item(), 1)

    return results


def run_mms_bench(model, dataloader, device, top_k_list):
    """Run all MMS-Bench scenarios."""
    results = {}

    for scenario_id, scenario in MMS_SCENARIOS.items():
        print(f"  {scenario_id}: {scenario['name']}...", end=" ", flush=True)
        t0 = time.time()

        spec_reprs, mol_reprs = encode_all(
            model, dataloader, device,
            modality_override=scenario["mask"]
        )
        metrics = retrieval_accuracy(spec_reprs, mol_reprs, top_k_list)
        elapsed = time.time() - t0

        results[scenario_id] = {
            "name": scenario["name"],
            "mask": scenario["mask"],
            **metrics,
            "time_seconds": round(elapsed, 1),
        }
        print(f"top1={metrics['top1']:.4f} top5={metrics.get('top5', 'N/A')} ({elapsed:.0f}s)")

    # Compute summary metrics
    s1_top1 = results["S1"]["top1"]
    missing_top1s = [results[s]["top1"] for s in ["S2", "S3", "S4", "S5", "S6", "S7"]]
    avg_missing = np.mean(missing_top1s)

    results["summary"] = {
        "full_modal_top1": s1_top1,
        "avg_missing_top1": round(float(avg_missing), 4),
        "degradation_rates": {},
    }
    for s in ["S2", "S3", "S4", "S5", "S6", "S7"]:
        deg = (s1_top1 - results[s]["top1"]) / (s1_top1 + 1e-8)
        results["summary"]["degradation_rates"][s] = round(float(deg), 4)

    return results


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    ablation = ckpt.get("args", {}).get("ablation") or cfg.get("ablation")
    if ablation:
        from src.model.model_ablation import build_ablation_model
        model, _ = build_ablation_model(ablation, cfg)
        print(f"[ABLATION MODE: {ablation}]")
    else:
        model, _ = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

    # Data
    ds_kwargs = {"split_file": args.split_file} if args.split_file else {}
    dataset = FastMultimodalDataset(args.data_dir, split=args.split, max_samples=args.max_samples, **ds_kwargs)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=fast_collate,
        pin_memory=True,
    )

    # Run benchmark
    print(f"\nRunning MMS-Bench ({len(dataset)} samples, {len(MMS_SCENARIOS)} scenarios)...")
    print("=" * 60)
    results = run_mms_bench(model, dataloader, device, args.top_k)
    print("=" * 60)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  MMS-Bench Results Summary")
    print(f"{'='*60}")
    print(f"  Full modal (S1) top-1: {results['summary']['full_modal_top1']:.4f}")
    print(f"  Avg missing modal top-1: {results['summary']['avg_missing_top1']:.4f}")
    print(f"\n  Degradation rates (lower = more robust):")
    for s, deg in results["summary"]["degradation_rates"].items():
        print(f"    {s} ({results[s]['name']}): {deg:.2%}")

    # Save JSON
    out_path = Path(args.report_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Output METRIC lines for autoresearch
    if args.output_metrics:
        print(f"\nMETRIC avg_missing_top1={results['summary']['avg_missing_top1']}")
        print(f"METRIC full_modal_top1={results['summary']['full_modal_top1']}")
        for s_id in MMS_SCENARIOS:
            print(f"METRIC {s_id.lower()}_top1={results[s_id]['top1']}")


if __name__ == "__main__":
    main()
