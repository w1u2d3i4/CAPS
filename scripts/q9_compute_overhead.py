#!/usr/bin/env python3
"""Q9 (round 3 review): CAPS vs proxy-only — compute and memory overhead.

Times forward (and forward+backward) wall-clock per batch and measures peak
GPU memory for the two strategies on identical batches at identical batch
size, on a fresh device.

Output: results/q9_compute_overhead.json
"""
import json, sys, os, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import torch
import torch.nn.functional as F
from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from src.model.model import build_model
from src.model.model_ablation import build_ablation_model
from torch.utils.data import DataLoader

CKPT_CAPS  = "checkpoints/scaffold_full_caps_100ep/best.pt"
CKPT_PROXY = "checkpoints/ablation_proxy_only_100ep/best.pt"

N_WARMUP = 5
N_TIMED  = 50
BATCH    = 128


def benchmark_one(model, batches, device, mode="fwd"):
    times = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for b in batches[:N_WARMUP]:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        b["modality_mask"] = torch.tensor([[True, False, True]]*b["mol_fp"].size(0), dtype=torch.bool, device=device)
        out = model(b)
        if mode == "fwdbwd":
            (F.normalize(out["fused_repr"], dim=-1) * F.normalize(out["mol_repr"], dim=-1)).sum().backward()
            for p in model.parameters():
                if p.grad is not None: p.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for b in batches[N_WARMUP:N_WARMUP + N_TIMED]:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        b["modality_mask"] = torch.tensor([[True, False, True]]*b["mol_fp"].size(0), dtype=torch.bool, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(b)
        if mode == "fwdbwd":
            (F.normalize(out["fused_repr"], dim=-1) * F.normalize(out["mol_repr"], dim=-1)).sum().backward()
            for p in model.parameters():
                if p.grad is not None: p.grad = None
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "ms_per_batch_mean": float(np.mean(times)),
        "ms_per_batch_std":  float(np.std(times)),
        "peak_mem_mb":       float(peak_mb),
        "n_timed":           N_TIMED,
        "batch_size":        BATCH,
    }


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FastMultimodalDataset("data/processed", split="test",
                               split_file="scaffold_split.json", max_samples=BATCH * (N_WARMUP + N_TIMED))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2, collate_fn=fast_collate)
    batches = list(dl)
    print(f"Cached {len(batches)} batches of size {BATCH}")

    out = {"batch_size": BATCH, "n_warmup": N_WARMUP, "n_timed": N_TIMED, "models": {}}

    for name, path in [("Full_CAPS", CKPT_CAPS), ("Proxy_only", CKPT_PROXY)]:
        if not Path(path).exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        print(f"\n=== {name} ===")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        ablation = ckpt.get("args", {}).get("ablation") or cfg.get("ablation")
        if ablation:
            model, _ = build_ablation_model(ablation, cfg)
        else:
            model, _ = build_model(cfg)
        model.load_state_dict(ckpt["model"], strict=False)
        model = model.to(device)
        np_total = n_params(model)
        print(f"  params = {np_total/1e6:.3f} M")

        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            fwd = benchmark_one(model, batches, device, mode="fwd")
        print(f"  forward  : {fwd['ms_per_batch_mean']:.2f} ± {fwd['ms_per_batch_std']:.2f} ms / batch  "
              f"peak {fwd['peak_mem_mb']:.0f} MB")

        torch.cuda.empty_cache()
        model.train()
        for p in model.parameters(): p.requires_grad_(True)
        fwdbwd = benchmark_one(model, batches, device, mode="fwdbwd")
        print(f"  fwd+bwd  : {fwdbwd['ms_per_batch_mean']:.2f} ± {fwdbwd['ms_per_batch_std']:.2f} ms / batch  "
              f"peak {fwdbwd['peak_mem_mb']:.0f} MB")

        out["models"][name] = {
            "params_M":   round(np_total / 1e6, 3),
            "forward":    fwd,
            "fwd_bwd":    fwdbwd,
        }

    if "Full_CAPS" in out["models"] and "Proxy_only" in out["models"]:
        c = out["models"]["Full_CAPS"]; p = out["models"]["Proxy_only"]
        out["overhead"] = {
            "params_pct":         round(100 * (c["params_M"] / p["params_M"] - 1), 2),
            "forward_ms_pct":     round(100 * (c["forward"]["ms_per_batch_mean"] / p["forward"]["ms_per_batch_mean"] - 1), 2),
            "fwdbwd_ms_pct":      round(100 * (c["fwd_bwd"]["ms_per_batch_mean"] / p["fwd_bwd"]["ms_per_batch_mean"] - 1), 2),
            "forward_mem_pct":    round(100 * (c["forward"]["peak_mem_mb"] / p["forward"]["peak_mem_mb"] - 1), 2),
            "fwdbwd_mem_pct":     round(100 * (c["fwd_bwd"]["peak_mem_mb"] / p["fwd_bwd"]["peak_mem_mb"] - 1), 2),
        }
        print(f"\n=== CAPS overhead vs proxy-only ===")
        for k, v in out["overhead"].items():
            print(f"  {k:>20s}: {v:+.2f}%")

    out_path = "results/q9_compute_overhead.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
