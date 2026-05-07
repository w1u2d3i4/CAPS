#!/usr/bin/env python3
"""
Preprocess real-world NMR data (nmrshiftdb2 + SDBS) into numpy mmap format
compatible with FastMultimodalDataset. IR is left as zero (missing modality).
"""
import json, os, sys, glob
from pathlib import Path
import numpy as np

# Use rdkit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import smiles_to_fingerprint, H1_FEAT_DIM, C13_FEAT_DIM

REAL_ROOT = Path("/opt/data/private/NMR/NMR实测数据")
OUT_DIR = Path("/opt/data/private/xrd2c_v2/idea_e/data/real_nmr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IR_LEN = 1800

def make_h_peaks(shifts):
    """1H shifts list -> (N, 7) array. Fill defaults for missing fields."""
    if not shifts:
        return np.zeros((0, H1_FEAT_DIM), dtype=np.float32)
    arr = np.zeros((len(shifts), H1_FEAT_DIM), dtype=np.float32)
    for i, s in enumerate(shifts):
        try:
            arr[i, 0] = float(s)
            arr[i, 1] = 0.0  # width
            arr[i, 2] = 1.0  # nH
            arr[i, 3] = 4.0  # mult_code = "m"
        except (ValueError, TypeError):
            continue
    # filter rows with shift==0 if all zero
    mask = ~np.all(arr == 0, axis=1)
    return arr[mask]

def make_c_peaks(shifts, intensities=None):
    """13C shifts -> (N, 3): [shift, intensity, width]."""
    if not shifts:
        return np.zeros((0, C13_FEAT_DIM), dtype=np.float32)
    arr = np.zeros((len(shifts), C13_FEAT_DIM), dtype=np.float32)
    for i, s in enumerate(shifts):
        try:
            arr[i, 0] = float(s)
            arr[i, 1] = float(intensities[i]) if intensities and i < len(intensities) else 1.0
            arr[i, 2] = 0.0
        except (ValueError, TypeError):
            continue
    mask = ~np.all(arr == 0, axis=1)
    return arr[mask]

# ---- Load nmrshiftdb2 ----
print("Loading nmrshiftdb2...")
with open(REAL_ROOT / "nmrshiftdb2/output_with_formula.json") as f:
    nshift = json.load(f)

samples = []
for entry in nshift:
    smi = entry.get("SMILES", "").strip()
    if not smi:
        continue
    h_sh = entry.get("1H_shifts", [])
    c_sh = entry.get("13C_shifts", [])
    if not h_sh and not c_sh:
        continue
    samples.append({
        "source": "nmrshiftdb2",
        "smiles": smi,
        "h_shifts": h_sh,
        "c_shifts": c_sh,
        "c_int": None,
    })
print(f"  nmrshiftdb2 valid: {len(samples)}")

# ---- Load SDBS (match 1H + 13C by filename) ----
print("Loading SDBS...")
sdbs_13c_dir = REAL_ROOT / "sdbs/13C"
sdbs_1h_dir = REAL_ROOT / "sdbs/1H"
files_13c = {f.name: f for f in sdbs_13c_dir.glob("*.json")}
files_1h = {f.name: f for f in sdbs_1h_dir.glob("*.json")}

# Use union (process every SDBS compound, even if only one modality available)
all_names = set(files_13c.keys()) | set(files_1h.keys())
sdbs_added = 0
for fname in all_names:
    smi, h_sh, c_sh, c_int = None, [], [], []
    if fname in files_13c:
        try:
            with open(files_13c[fname]) as f:
                d = json.load(f)
            smi = d["info"].get("SMILES", "").strip() or smi
            for sh in d.get("shift", []):
                ppm = sh.get("ppm")
                try:
                    c_sh.append(float(ppm))
                    c_int.append(float(sh.get("Int", 1.0)) if sh.get("Int") else 1.0)
                except (ValueError, TypeError):
                    continue
        except Exception:
            pass
    if fname in files_1h:
        try:
            with open(files_1h[fname]) as f:
                d = json.load(f)
            smi = smi or d["info"].get("SMILES", "").strip()
            for sh in d.get("shift", []):
                # SDBS 1H format: ppm field is letter, Assign is actual ppm value
                val = sh.get("Assign", sh.get("ppm"))
                try:
                    h_sh.append(float(val))
                except (ValueError, TypeError):
                    continue
        except Exception:
            pass
    if smi and (h_sh or c_sh):
        samples.append({
            "source": "sdbs",
            "smiles": smi,
            "h_shifts": h_sh,
            "c_shifts": c_sh,
            "c_int": c_int if c_int else None,
        })
        sdbs_added += 1
print(f"  SDBS added: {sdbs_added}")
print(f"Total raw samples: {len(samples)}")

# ---- Convert + compute fingerprints ----
print("Computing fingerprints + arrays...")
h_peaks_list, c_peaks_list = [], []
fps_list, smiles_list, sources = [], [], []
n_skipped = 0
for s in samples:
    fp = smiles_to_fingerprint(s["smiles"])
    if fp is None or fp.sum() == 0:
        n_skipped += 1
        continue
    h_arr = make_h_peaks(s["h_shifts"])
    c_arr = make_c_peaks(s["c_shifts"], s["c_int"])
    if len(h_arr) == 0 and len(c_arr) == 0:
        n_skipped += 1
        continue
    h_peaks_list.append(h_arr)
    c_peaks_list.append(c_arr)
    fps_list.append(fp.astype(np.float32))
    smiles_list.append(s["smiles"])
    sources.append(s["source"])
print(f"  valid: {len(fps_list)}, skipped: {n_skipped}")

# ---- Pad to uniform shape ----
N = len(fps_list)
max_h = max((len(p) for p in h_peaks_list), default=1)
max_c = max((len(p) for p in c_peaks_list), default=1)
print(f"  max_h={max_h}, max_c={max_c}")

h_arr_full = np.zeros((N, max_h, H1_FEAT_DIM), dtype=np.float32)
c_arr_full = np.zeros((N, max_c, C13_FEAT_DIM), dtype=np.float32)
h_lens = np.zeros(N, dtype=np.int32)
c_lens = np.zeros(N, dtype=np.int32)
for i, (hp, cp) in enumerate(zip(h_peaks_list, c_peaks_list)):
    if len(hp) > 0:
        h_arr_full[i, :len(hp)] = hp
        h_lens[i] = len(hp)
    if len(cp) > 0:
        c_arr_full[i, :len(cp)] = cp
        c_lens[i] = len(cp)

ir_arr = np.zeros((N, IR_LEN), dtype=np.float32)
fps_arr = np.stack(fps_list)

# ---- Save ----
print(f"Saving to {OUT_DIR}...")
np.save(OUT_DIR / "h_nmr_peaks.npy", h_arr_full)
np.save(OUT_DIR / "h_nmr_lengths.npy", h_lens)
np.save(OUT_DIR / "c_nmr_peaks.npy", c_arr_full)
np.save(OUT_DIR / "c_nmr_lengths.npy", c_lens)
np.save(OUT_DIR / "ir_spectra.npy", ir_arr)
np.save(OUT_DIR / "mol_fps.npy", fps_arr)
with open(OUT_DIR / "smiles.json", "w") as f:
    json.dump(smiles_list, f)
with open(OUT_DIR / "sources.json", "w") as f:
    json.dump(sources, f)
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump({
        "total": N,
        "max_h": int(max_h),
        "max_c": int(max_c),
        "ir_len": IR_LEN,
        "h_dim": H1_FEAT_DIM,
        "c_dim": C13_FEAT_DIM,
        "fp_dim": 2048,
        "note": "Real-world NMR data (nmrshiftdb2 + SDBS). IR is missing (set to zero).",
    }, f, indent=2)

# Summary
print(f"\n=== Done ===")
print(f"Total: {N} samples")
print(f"Sources: nmrshiftdb2={sum(1 for s in sources if s=='nmrshiftdb2')}, sdbs={sum(1 for s in sources if s=='sdbs')}")
print(f"With 1H+13C: {sum(1 for h,c in zip(h_lens, c_lens) if h>0 and c>0)}")
print(f"Only 13C: {sum(1 for h,c in zip(h_lens, c_lens) if h==0 and c>0)}")
print(f"Only 1H:  {sum(1 for h,c in zip(h_lens, c_lens) if h>0 and c==0)}")
