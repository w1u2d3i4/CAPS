#!/usr/bin/env python3
"""
Preprocess parquet dataset into fast numpy format.

Converts variable-length peaks into padded numpy arrays + metadata,
stored as .npz files that load instantly.

Output structure:
  data/processed/
    h_nmr_peaks.npy     (N, max_h, 7) float32
    h_nmr_lengths.npy    (N,) int32
    c_nmr_peaks.npy     (N, max_c, 3) float32
    c_nmr_lengths.npy    (N,) int32
    ir_spectra.npy       (N, 1800) float32
    smiles.json          list of SMILES strings
    mol_fps.npy          (N, 2048) float32
    metadata.json        {total, max_h, max_c, ir_len}
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import extract_h_nmr_peaks, extract_c_nmr_peaks, extract_ir_spectrum, smiles_to_fingerprint, H1_FEAT_DIM, C13_FEAT_DIM


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    parquet_dir = Path("data/multimodal_spectroscopic/multimodal_spectroscopic_dataset")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("aligned_chunk_*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    # Pass 1: scan max peak counts
    print("Pass 1: scanning peak counts...")
    max_h, max_c = 0, 0
    total = 0
    for pf in tqdm(parquet_files):
        df = pd.read_parquet(pf, columns=["h_nmr_peaks", "c_nmr_peaks"])
        for _, row in df.iterrows():
            h = row["h_nmr_peaks"]
            c = row["c_nmr_peaks"]
            if h is not None and hasattr(h, "__len__"):
                max_h = max(max_h, len(h))
            if c is not None and hasattr(c, "__len__"):
                max_c = max(max_c, len(c))
        total += len(df)
        del df

    print(f"Total samples: {total}, max_h_peaks: {max_h}, max_c_peaks: {max_c}")

    # Pre-allocate arrays
    all_h = np.zeros((total, max_h, H1_FEAT_DIM), dtype=np.float32)
    all_h_len = np.zeros(total, dtype=np.int32)
    all_c = np.zeros((total, max_c, C13_FEAT_DIM), dtype=np.float32)
    all_c_len = np.zeros(total, dtype=np.int32)
    all_ir = np.zeros((total, 1800), dtype=np.float32)
    all_fps = np.zeros((total, 2048), dtype=np.float32)
    all_smiles = []

    # Pass 2: extract features
    print("Pass 2: extracting features...")
    idx = 0
    for pf in tqdm(parquet_files):
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            # 1H
            h = extract_h_nmr_peaks(row.get("h_nmr_peaks"))
            n_h = len(h)
            if n_h > 0:
                all_h[idx, :n_h] = h
            all_h_len[idx] = n_h

            # 13C
            c = extract_c_nmr_peaks(row.get("c_nmr_peaks"))
            n_c = len(c)
            if n_c > 0:
                all_c[idx, :n_c] = c
            all_c_len[idx] = n_c

            # IR
            ir = extract_ir_spectrum(row.get("ir_spectra"))
            if len(ir) > 0:
                all_ir[idx, :len(ir)] = ir[:1800]

            # SMILES + fingerprint
            smiles = str(row.get("smiles", ""))
            all_smiles.append(smiles)
            all_fps[idx] = smiles_to_fingerprint(smiles)

            idx += 1
        del df

    assert idx == total

    # Save
    print("Saving...")
    np.save(out_dir / "h_nmr_peaks.npy", all_h)
    np.save(out_dir / "h_nmr_lengths.npy", all_h_len)
    np.save(out_dir / "c_nmr_peaks.npy", all_c)
    np.save(out_dir / "c_nmr_lengths.npy", all_c_len)
    np.save(out_dir / "ir_spectra.npy", all_ir)
    np.save(out_dir / "mol_fps.npy", all_fps)

    with open(out_dir / "smiles.json", "w") as f:
        json.dump(all_smiles, f)

    metadata = {
        "total": total,
        "max_h_peaks": int(max_h),
        "max_c_peaks": int(max_c),
        "ir_length": 1800,
        "h_feat_dim": H1_FEAT_DIM,
        "c_feat_dim": C13_FEAT_DIM,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Report sizes
    for f in out_dir.glob("*.npy"):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"\nDone! Preprocessed {total} samples to {out_dir}/")


if __name__ == "__main__":
    main()
