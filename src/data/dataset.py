"""
Multi-modal spectroscopy dataset for NMR-MultiFuse v3.

Reads the 790K Multimodal Spectroscopic Dataset (parquet) and provides:
  - 1H-NMR peaks (variable-length set)
  - 13C-NMR peaks (variable-length set)
  - IR spectrum (fixed-length 1D signal)
  - Molecular SMILES + Morgan fingerprint
  - Modality availability mask
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Peak extraction helpers
# ---------------------------------------------------------------------------

def _parse_j_values(j_str, max_j: int = 3) -> list:
    """Parse j_values string '4.46_7.80_' -> [4.46, 7.80, 0.0] (padded to max_j)."""
    js = []
    if j_str and j_str != "None":
        parts = [x for x in str(j_str).split("_") if x.strip()]
        for p in parts[:max_j]:
            try:
                js.append(float(p))
            except ValueError:
                pass
    # Pad to max_j
    while len(js) < max_j:
        js.append(0.0)
    return js[:max_j]


# Multiplicity encoding
MULT_MAP = {"s": 0, "d": 1, "t": 2, "q": 3, "m": 4, "dd": 5, "dt": 6, "td": 6, "ddd": 7, "tt": 8}
MAX_J = 3  # max coupling constants per peak

# 1H feature dim: shift + range_width + nH + mult_code + j1 + j2 + j3 = 7
H1_FEAT_DIM = 7


def extract_h_nmr_peaks(raw_peaks) -> np.ndarray:
    """
    Extract 1H-NMR peaks -> (N, 7):
      [chemical_shift, range_width, nH, multiplicity_code, j1, j2, j3]

    Fields from official dataset:
      centroid: chemical shift center
      rangeMax/rangeMin: peak range -> width = rangeMax - rangeMin
      nH: number of hydrogens (critical for structure)
      category: multiplicity type (s/d/t/dd/dt...)
      j_values: coupling constants (underscore-separated string)
    """
    if raw_peaks is None or (hasattr(raw_peaks, "__len__") and len(raw_peaks) == 0):
        return np.zeros((0, H1_FEAT_DIM), dtype=np.float32)

    peaks = []
    for p in raw_peaks:
        if isinstance(p, dict):
            shift = float(p.get("centroid", p.get("delta", 0)))
            range_max = float(p.get("rangeMax", shift))
            range_min = float(p.get("rangeMin", shift))
            width = range_max - range_min
            nH = float(p.get("nH", 1))
            mult_str = str(p.get("category", "m")).lower()
            mult_code = float(MULT_MAP.get(mult_str, len(MULT_MAP)))
            js = _parse_j_values(p.get("j_values"), MAX_J)
            peaks.append([shift, width, nH, mult_code] + js)
        else:
            # Fallback
            shift = float(p[0]) if len(p) > 0 else 0
            peaks.append([shift, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0])

    return np.array(peaks, dtype=np.float32) if peaks else np.zeros((0, H1_FEAT_DIM), dtype=np.float32)


# 13C feature dim: shift + intensity + width = 3
C13_FEAT_DIM = 3


def extract_c_nmr_peaks(raw_peaks) -> np.ndarray:
    """
    Extract 13C-NMR peaks -> (N, 3): [chemical_shift, intensity, width].

    Fields from official dataset:
      delta (ppm): chemical shift
      intensity: peak intensity
      width (ppm): peak width (related to relaxation / molecular environment)
    """
    if raw_peaks is None or (hasattr(raw_peaks, "__len__") and len(raw_peaks) == 0):
        return np.zeros((0, C13_FEAT_DIM), dtype=np.float32)

    peaks = []
    for p in raw_peaks:
        if isinstance(p, dict):
            shift = float(p.get("delta (ppm)", p.get("delta", p.get("centroid", 0))))
            intensity = float(p.get("intensity", p.get("integral", 1.0)))
            width = float(p.get("width (ppm)", 0.0))
        else:
            shift = float(p[0]) if len(p) > 0 else 0
            intensity = float(p[1]) if len(p) > 1 else 1.0
            width = 0.0
        peaks.append([shift, intensity, width])

    return np.array(peaks, dtype=np.float32) if peaks else np.zeros((0, C13_FEAT_DIM), dtype=np.float32)


def extract_ir_spectrum(raw_spectrum) -> np.ndarray:
    """Extract IR spectrum -> (L,) float32 array."""
    if raw_spectrum is None or (hasattr(raw_spectrum, "__len__") and len(raw_spectrum) == 0):
        return np.zeros(0, dtype=np.float32)
    return np.asarray(raw_spectrum, dtype=np.float32)


def smiles_to_fingerprint(smiles: str, nbits: int = 2048, radius: int = 2) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint."""
    if not HAS_RDKIT:
        # Fallback: hash-based pseudo-fingerprint
        h = hash(smiles)
        rng = np.random.RandomState(abs(h) % (2**31))
        return rng.randint(0, 2, size=nbits).astype(np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultimodalSpectDataset(Dataset):
    """
    Multi-modal spectroscopy dataset.

    Each sample returns a dict with:
        - h_nmr_peaks: (N_h, 3) float32  [shift, intensity, mult_code]
        - c_nmr_peaks: (N_c, 2) float32  [shift, intensity]
        - ir_spectrum: (L_ir,) float32
        - mol_fp: (2048,) float32  Morgan fingerprint
        - smiles: str
        - modality_mask: (3,) bool  [has_1h, has_13c, has_ir]
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: Optional[int] = None,
        fp_nbits: int = 2048,
    ):
        self.data_dir = Path(data_dir)
        self.fp_nbits = fp_nbits

        # Find parquet files
        parquet_dir = self.data_dir
        if (self.data_dir / "multimodal_spectroscopic_dataset").exists():
            parquet_dir = self.data_dir / "multimodal_spectroscopic_dataset"

        self.parquet_files = sorted(parquet_dir.glob("aligned_chunk_*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

        # Count samples per chunk — cache to file to avoid slow metadata scan
        cache_path = parquet_dir / "_chunk_sizes.json"
        if cache_path.exists():
            import json
            with open(cache_path) as f:
                cached = json.load(f)
            # Validate cache matches current files
            if len(cached) == len(self.parquet_files):
                self.chunk_sizes = cached
            else:
                self.chunk_sizes = self._scan_chunk_sizes(cache_path)
        else:
            self.chunk_sizes = self._scan_chunk_sizes(cache_path)
        total_samples = sum(self.chunk_sizes)

        # Build cumulative index: global_idx -> (chunk_idx, row_within_chunk)
        self.cum_sizes = np.cumsum([0] + self.chunk_sizes)

        # Split by global index
        rng = np.random.RandomState(seed)
        all_indices = rng.permutation(total_samples)
        n_train = int(total_samples * train_ratio)
        n_val = int(total_samples * val_ratio)

        if split == "train":
            self.indices = np.sort(all_indices[:n_train])
        elif split == "val":
            self.indices = np.sort(all_indices[n_train:n_train + n_val])
        elif split == "test":
            self.indices = np.sort(all_indices[n_train + n_val:])
        else:
            self.indices = np.arange(total_samples)

        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        # LRU cache for loaded chunks (keep 3 in memory at a time)
        self._chunk_cache = {}
        self._cache_order = []
        self._max_cached = 3

        print(f"[Dataset] {total_samples:,} total samples across {len(self.parquet_files)} chunks, "
              f"{split} split: {len(self.indices):,}")

    def __len__(self):
        return len(self.indices)

    def _scan_chunk_sizes(self, cache_path: Path) -> list:
        """Scan parquet metadata and cache chunk sizes."""
        import pyarrow.parquet as pq
        import json
        sizes = []
        for i, pf in enumerate(self.parquet_files):
            meta = pq.read_metadata(pf)
            sizes.append(meta.num_rows)
            if (i + 1) % 50 == 0:
                print(f"  Scanning metadata: {i+1}/{len(self.parquet_files)}")
        try:
            with open(cache_path, "w") as f:
                json.dump(sizes, f)
            print(f"  Cached chunk sizes to {cache_path}")
        except OSError:
            pass
        return sizes

    def _load_chunk(self, chunk_idx: int) -> pd.DataFrame:
        """Load a parquet chunk with LRU caching."""
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        # Evict oldest if at capacity
        if len(self._chunk_cache) >= self._max_cached:
            oldest = self._cache_order.pop(0)
            del self._chunk_cache[oldest]

        df = pd.read_parquet(self.parquet_files[chunk_idx])
        self._chunk_cache[chunk_idx] = df
        self._cache_order.append(chunk_idx)
        return df

    def _global_to_local(self, global_idx: int) -> tuple:
        """Convert global index to (chunk_idx, row_idx)."""
        chunk_idx = int(np.searchsorted(self.cum_sizes[1:], global_idx, side="right"))
        row_idx = global_idx - self.cum_sizes[chunk_idx]
        return chunk_idx, row_idx

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        chunk_idx, row_idx = self._global_to_local(global_idx)
        chunk_df = self._load_chunk(chunk_idx)
        row = chunk_df.iloc[row_idx]

        # Extract modalities
        h_peaks = extract_h_nmr_peaks(row.get("h_nmr_peaks"))
        c_peaks = extract_c_nmr_peaks(row.get("c_nmr_peaks"))
        ir_spec = extract_ir_spectrum(row.get("ir_spectra"))

        # Modality availability
        has_1h = len(h_peaks) > 0
        has_13c = len(c_peaks) > 0
        has_ir = len(ir_spec) > 0

        # Molecular fingerprint
        smiles = str(row.get("smiles", ""))
        mol_fp = smiles_to_fingerprint(smiles, nbits=self.fp_nbits)

        return {
            "h_nmr_peaks": h_peaks,        # (N_h, 3)
            "c_nmr_peaks": c_peaks,         # (N_c, 2)
            "ir_spectrum": ir_spec,          # (L_ir,)
            "mol_fp": mol_fp,               # (2048,)
            "smiles": smiles,
            "modality_mask": np.array([has_1h, has_13c, has_ir], dtype=np.bool_),
        }


# ---------------------------------------------------------------------------
# Collate function (handles variable-length peaks)
# ---------------------------------------------------------------------------

def collate_multimodal(batch: list) -> dict:
    """
    Collate variable-length peaks into padded tensors.

    Returns:
        h_nmr_peaks: (B, max_N_h, 3) float32
        h_nmr_mask:  (B, max_N_h) bool
        c_nmr_peaks: (B, max_N_c, 2) float32
        c_nmr_mask:  (B, max_N_c) bool
        ir_spectrum: (B, L_ir) float32
        mol_fp:      (B, 2048) float32
        modality_mask: (B, 3) bool
        smiles: list of str
    """
    B = len(batch)

    # Pad 1H peaks (dim = H1_FEAT_DIM = 7)
    h_feat_dim = batch[0]["h_nmr_peaks"].shape[1] if len(batch[0]["h_nmr_peaks"]) > 0 else H1_FEAT_DIM
    max_h = max(len(b["h_nmr_peaks"]) for b in batch)
    max_h = max(max_h, 1)  # at least 1
    h_peaks = np.zeros((B, max_h, h_feat_dim), dtype=np.float32)
    h_mask = np.zeros((B, max_h), dtype=np.bool_)
    for i, b in enumerate(batch):
        n = len(b["h_nmr_peaks"])
        if n > 0:
            h_peaks[i, :n] = b["h_nmr_peaks"]
            h_mask[i, :n] = True

    # Pad 13C peaks (dim = C13_FEAT_DIM = 3)
    c_feat_dim = batch[0]["c_nmr_peaks"].shape[1] if len(batch[0]["c_nmr_peaks"]) > 0 else C13_FEAT_DIM
    max_c = max(len(b["c_nmr_peaks"]) for b in batch)
    max_c = max(max_c, 1)
    c_peaks = np.zeros((B, max_c, c_feat_dim), dtype=np.float32)
    c_mask = np.zeros((B, max_c), dtype=np.bool_)
    for i, b in enumerate(batch):
        n = len(b["c_nmr_peaks"])
        if n > 0:
            c_peaks[i, :n] = b["c_nmr_peaks"]
            c_mask[i, :n] = True

    # IR spectra (all same length in this dataset, but handle variable just in case)
    ir_lens = [len(b["ir_spectrum"]) for b in batch]
    max_ir = max(ir_lens) if ir_lens else 1800
    max_ir = max(max_ir, 1)
    ir_spec = np.zeros((B, max_ir), dtype=np.float32)
    for i, b in enumerate(batch):
        n = len(b["ir_spectrum"])
        if n > 0:
            ir_spec[i, :n] = b["ir_spectrum"]

    # Fingerprints
    fps = np.stack([b["mol_fp"] for b in batch])

    # Modality mask
    mod_mask = np.stack([b["modality_mask"] for b in batch])

    # SMILES
    smiles_list = [b["smiles"] for b in batch]

    return {
        "h_nmr_peaks": torch.from_numpy(h_peaks),
        "h_nmr_mask": torch.from_numpy(h_mask),
        "c_nmr_peaks": torch.from_numpy(c_peaks),
        "c_nmr_mask": torch.from_numpy(c_mask),
        "ir_spectrum": torch.from_numpy(ir_spec),
        "mol_fp": torch.from_numpy(fps),
        "modality_mask": torch.from_numpy(mod_mask),
        "smiles": smiles_list,
    }


def get_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    ds = MultimodalSpectDataset(data_dir, split=split, max_samples=max_samples, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True,
        drop_last=(split == "train"),
    )
