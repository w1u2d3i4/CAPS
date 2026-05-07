#!/usr/bin/env python3
"""
Analyze modality coverage of the Multimodal Spectroscopic Dataset.

Outputs:
  - Per-modality sample count and coverage %
  - Modality combination distribution (which subsets of modalities co-occur)
  - Data shape statistics (spectrum lengths, peak counts)
  - Report saved to results/modality_coverage_report.json

Usage:
    python scripts/analyze_modality_coverage.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/multimodal_spectroscopic/multimodal_spectroscopic_dataset")
OUTPUT_DIR = Path("results")

# The 3 modalities we care about for Agent E (NMR-MultiFuse v3)
CORE_MODALITIES = {
    "1H_NMR": "h_nmr_peaks",
    "13C_NMR": "c_nmr_peaks",
    "IR": "ir_spectra",
}

# All available modalities in the dataset
ALL_MODALITIES = {
    "1H_NMR": "h_nmr_peaks",
    "13C_NMR": "c_nmr_peaks",
    "IR": "ir_spectra",
    "HSQC": "hsqc_nmr_peaks",
    "MS_pos": "msms_positive_10ev",
    "MS_neg": "msms_negative_10ev",
    "1H_spectrum": "h_nmr_spectra",
    "13C_spectrum": "c_nmr_spectra",
}


def is_valid(value) -> bool:
    """Check if a modality value is non-empty / non-null."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if hasattr(value, "__len__"):
        return len(value) > 0
    return True


def analyze_chunk(filepath: Path) -> dict:
    """Analyze a single parquet chunk."""
    df = pd.read_parquet(filepath)
    n = len(df)

    stats = {
        "n_samples": n,
        "core_coverage": {},
        "all_coverage": {},
        "core_combos": Counter(),
        "peak_counts": defaultdict(list),
        "spectrum_lengths": defaultdict(list),
    }

    for name, col in CORE_MODALITIES.items():
        if col not in df.columns:
            stats["core_coverage"][name] = 0
            continue
        count = sum(1 for v in df[col] if is_valid(v))
        stats["core_coverage"][name] = count

    for name, col in ALL_MODALITIES.items():
        if col not in df.columns:
            stats["all_coverage"][name] = 0
            continue
        count = sum(1 for v in df[col] if is_valid(v))
        stats["all_coverage"][name] = count

    # Per-sample: which core modality combination?
    for idx in range(n):
        combo = []
        for name, col in CORE_MODALITIES.items():
            if col in df.columns and is_valid(df[col].iloc[idx]):
                combo.append(name)
        combo_key = "+".join(sorted(combo)) if combo else "NONE"
        stats["core_combos"][combo_key] += 1

    # Shape statistics (sample from first 100 rows for speed)
    sample_n = min(100, n)
    for idx in range(sample_n):
        for name, col in [("1H_NMR", "h_nmr_peaks"), ("13C_NMR", "c_nmr_peaks")]:
            if col in df.columns:
                v = df[col].iloc[idx]
                if is_valid(v):
                    stats["peak_counts"][name].append(len(v))
        for name, col in [("IR", "ir_spectra"), ("1H_spectrum", "h_nmr_spectra"), ("13C_spectrum", "c_nmr_spectra")]:
            if col in df.columns:
                v = df[col].iloc[idx]
                if is_valid(v):
                    stats["spectrum_lengths"][name].append(len(v))

    return stats


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    parquet_files = sorted(DATA_DIR.glob("aligned_chunk_*.parquet"))
    print(f"Found {len(parquet_files)} parquet chunks in {DATA_DIR}")

    # Aggregate stats
    total_samples = 0
    total_core_coverage = Counter()
    total_all_coverage = Counter()
    total_core_combos = Counter()
    all_peak_counts = defaultdict(list)
    all_spectrum_lengths = defaultdict(list)

    for i, fp in enumerate(parquet_files):
        if i % 20 == 0:
            print(f"  Processing chunk {i+1}/{len(parquet_files)}...")
        try:
            stats = analyze_chunk(fp)
        except Exception as e:
            print(f"  WARNING: Failed to process {fp.name}: {e}")
            continue

        total_samples += stats["n_samples"]
        for k, v in stats["core_coverage"].items():
            total_core_coverage[k] += v
        for k, v in stats["all_coverage"].items():
            total_all_coverage[k] += v
        for k, v in stats["core_combos"].items():
            total_core_combos[k] += v
        for k, v in stats["peak_counts"].items():
            all_peak_counts[k].extend(v)
        for k, v in stats["spectrum_lengths"].items():
            all_spectrum_lengths[k].extend(v)

    # Build report
    print(f"\n{'='*60}")
    print(f"  Multimodal Spectroscopic Dataset — Modality Coverage Report")
    print(f"{'='*60}")
    print(f"\n  Total samples: {total_samples:,}")

    print(f"\n  --- Core Modalities (Agent E focuses on these 3) ---")
    for name in CORE_MODALITIES:
        count = total_core_coverage.get(name, 0)
        pct = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {name:>10}: {count:>8,} / {total_samples:,}  ({pct:.1f}%)")

    print(f"\n  --- All Modalities ---")
    for name in ALL_MODALITIES:
        count = total_all_coverage.get(name, 0)
        pct = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {name:>15}: {count:>8,} / {total_samples:,}  ({pct:.1f}%)")

    print(f"\n  --- Core Modality Combinations ---")
    for combo, count in sorted(total_core_combos.items(), key=lambda x: -x[1]):
        pct = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {combo:>25}: {count:>8,}  ({pct:.1f}%)")

    # Key decision metrics
    three_modal = total_core_combos.get("1H_NMR+13C_NMR+IR", 0)
    any_two = sum(v for k, v in total_core_combos.items() if k.count("+") == 1)
    any_one = sum(v for k, v in total_core_combos.items() if "+" not in k and k != "NONE")
    three_pct = three_modal / total_samples * 100 if total_samples > 0 else 0

    print(f"\n  --- Decision Summary ---")
    print(f"  3-modal complete (1H+13C+IR): {three_modal:,} ({three_pct:.1f}%)")
    print(f"  2-modal (any pair):           {any_two:,}")
    print(f"  1-modal only:                 {any_one:,}")

    if three_pct > 50:
        print(f"\n  >> HIGH coverage: Stage 2 joint pre-training viable on majority of data")
    elif three_pct > 10:
        print(f"\n  >> MODERATE coverage: Stage 2 viable but pairwise (Stage 1) is critical")
    else:
        print(f"\n  >> LOW coverage: Prioritize pairwise pre-training, supplement with NMRPeak 1.8M")

    # Shape statistics
    print(f"\n  --- Data Shape Statistics (sampled) ---")
    for name in ["1H_NMR", "13C_NMR"]:
        vals = all_peak_counts.get(name, [])
        if vals:
            print(f"  {name} peak count: min={min(vals)}, median={int(np.median(vals))}, "
                  f"max={max(vals)}, mean={np.mean(vals):.1f}")
    for name in ["IR", "1H_spectrum", "13C_spectrum"]:
        vals = all_spectrum_lengths.get(name, [])
        if vals:
            print(f"  {name} length: min={min(vals)}, median={int(np.median(vals))}, "
                  f"max={max(vals)}, mean={np.mean(vals):.1f}")

    # Save JSON report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "total_samples": total_samples,
        "core_coverage": dict(total_core_coverage),
        "all_coverage": dict(total_all_coverage),
        "core_combos": dict(total_core_combos),
        "three_modal_count": three_modal,
        "three_modal_pct": round(three_pct, 2),
        "peak_stats": {
            name: {
                "min": int(min(vals)), "max": int(max(vals)),
                "median": int(np.median(vals)), "mean": round(np.mean(vals), 1),
            }
            for name, vals in all_peak_counts.items() if vals
        },
        "spectrum_stats": {
            name: {
                "min": int(min(vals)), "max": int(max(vals)),
                "median": int(np.median(vals)), "mean": round(np.mean(vals), 1),
            }
            for name, vals in all_spectrum_lengths.items() if vals
        },
    }
    report_path = OUTPUT_DIR / "modality_coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)  # cd to idea_e/
    main()
