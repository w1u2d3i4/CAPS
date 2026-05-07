#!/usr/bin/env python3
"""Preprocess NP-MRD into our model's input format.

NP-MRD predicted_nmrml_spectra.zip contains 4938 nmrML files; each has
either 1H OR 13C predicted spectrum (not both) for one molecule.
Filename pattern: NP{accession}_predicted_{run_id}.nmrML
Nucleus is identified inside the XML via <spectrum1D id="1D-1H"|"1D-13C">.
"""
import json, os, sys, zipfile, re
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

NP_DIR = Path("/opt/data/private/np_mrd_downloads")
OUT_DIR = Path("data/np_mrd_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H1_FEAT_DIM = 7
C13_FEAT_DIM = 3
IR_LEN = 1800
MAX_H_PEAKS = 80
MAX_C_PEAKS = 100

# 1) Load SMILES + accession from JSON shards
print("Loading metadata from JSON shards...")
all_records = []
for jf in sorted(NP_DIR.glob("extracted/npmrd_natural_products_*.json")):
    print(f"  {jf.name}")
    with open(jf) as f:
        data = json.load(f)
    for r in data["np_mrd"]["natural_product"]:
        smi = r.get("smiles") or ""
        acc = r.get("accession") or ""
        if smi and acc:
            all_records.append((acc, smi))
print(f"Loaded {len(all_records)} (accession, SMILES) records")
acc_to_idx = {acc: i for i, (acc, _) in enumerate(all_records)}
N = len(all_records)

# 2) Allocate arrays
all_h     = np.zeros((N, MAX_H_PEAKS, H1_FEAT_DIM),  dtype=np.float32)
all_h_len = np.zeros(N, dtype=np.int32)
all_c     = np.zeros((N, MAX_C_PEAKS, C13_FEAT_DIM), dtype=np.float32)
all_c_len = np.zeros(N, dtype=np.int32)
all_ir    = np.zeros((N, IR_LEN), dtype=np.float32)
all_fps   = np.zeros((N, 2048), dtype=np.float32)
smiles_list = [smi for _, smi in all_records]

# 3) Compute fingerprints
print("Computing fingerprints...")
from src.data.dataset import smiles_to_fingerprint
for i, smi in enumerate(tqdm(smiles_list)):
    all_fps[i] = smiles_to_fingerprint(smi)

# 4) Parse nmrML files
nmrml_zip = NP_DIR / "predicted_nmrml_spectra.zip"
if not (nmrml_zip.exists() and nmrml_zip.stat().st_size > 1_000_000_000):
    print(f"WARNING: nmrML zip not ready: {nmrml_zip.stat().st_size if nmrml_zip.exists() else 0}")
    sys.exit(1)

print(f"Parsing {nmrml_zip} ({nmrml_zip.stat().st_size/1e9:.2f} GB)...")
peak_re = re.compile(rb'<peak\s+center="([\d.eE+\-]+)"\s+amplitude="([\d.eE+\-]+)"')
nucleus_re = re.compile(rb'spectrum1D[^>]*id="1D-(1H|13C)"')

n_h_parsed = 0
n_c_parsed = 0
n_skipped = 0

with zipfile.ZipFile(nmrml_zip) as z:
    names = [n for n in z.namelist() if n.endswith(".nmrML")]
    print(f"  {len(names)} nmrML files")
    for name in tqdm(names, desc="parse nmrML"):
        m = re.search(r"(NP\d{7})", name)
        if not m:
            continue
        acc = m.group(1)
        if acc not in acc_to_idx:
            n_skipped += 1
            continue
        idx = acc_to_idx[acc]
        try:
            content = z.read(name)
        except Exception:
            continue

        nuc_m = nucleus_re.search(content)
        if not nuc_m:
            continue
        nucleus = nuc_m.group(1).decode()

        peaks = peak_re.findall(content)
        if not peaks:
            continue
        # keep top peaks by amplitude
        vals = np.array([(float(c), float(a)) for c, a in peaks], dtype=np.float32)
        if nucleus == "1H":
            n = min(len(vals), MAX_H_PEAKS)
            order = np.argsort(-vals[:, 1])[:n]  # by intensity desc
            sel = vals[order]
            all_h[idx, :n, 0] = sel[:, 0]
            all_h[idx, :n, 1] = sel[:, 1]
            all_h_len[idx] = n
            n_h_parsed += 1
        elif nucleus == "13C":
            n = min(len(vals), MAX_C_PEAKS)
            order = np.argsort(-vals[:, 1])[:n]
            sel = vals[order]
            all_c[idx, :n, 0] = sel[:, 0]
            all_c[idx, :n, 1] = sel[:, 1]
            all_c_len[idx] = n
            n_c_parsed += 1

print(f"\nParsed: {n_h_parsed} 1H spectra, {n_c_parsed} 13C spectra, {n_skipped} skipped (accession not in metadata)")

# 5) Save
np.save(OUT_DIR/"h_nmr_peaks.npy",   all_h)
np.save(OUT_DIR/"h_nmr_lengths.npy", all_h_len)
np.save(OUT_DIR/"c_nmr_peaks.npy",   all_c)
np.save(OUT_DIR/"c_nmr_lengths.npy", all_c_len)
np.save(OUT_DIR/"ir_spectra.npy",    all_ir)
np.save(OUT_DIR/"mol_fps.npy",       all_fps)
with open(OUT_DIR/"smiles.json", "w") as f:
    json.dump(smiles_list, f)
with open(OUT_DIR/"metadata.json", "w") as f:
    json.dump({
        "total": N,
        "max_h_peaks": MAX_H_PEAKS,
        "max_c_peaks": MAX_C_PEAKS,
        "ir_length": IR_LEN,
        "h_feat_dim": H1_FEAT_DIM,
        "c_feat_dim": C13_FEAT_DIM,
        "fp_dim": 2048,
        "source": "NP-MRD predicted nmrML",
        "n_with_h_nmr": int((all_h_len>0).sum()),
        "n_with_c_nmr": int((all_c_len>0).sum()),
        "n_with_both":  int(((all_h_len>0)&(all_c_len>0)).sum()),
        "note": "Predicted NP-MRD spectra; each nmrML has only one nucleus.",
    }, f, indent=2)
print(f"Saved to {OUT_DIR}")
print(f"  N={N}, with_H={int((all_h_len>0).sum())}, with_C={int((all_c_len>0).sum())}, with_both={int(((all_h_len>0)&(all_c_len>0)).sum())}")
