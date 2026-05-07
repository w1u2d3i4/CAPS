"""
Fast dataset — reads preprocessed numpy arrays.

Instant startup, zero parsing overhead.
Use after running: python scripts/preprocess_to_numpy.py

Memory-mapped loading: arrays stay on disk, pages loaded on demand by OS.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FastMultimodalDataset(Dataset):
    """
    Fast dataset from preprocessed numpy files.

    Startup: < 1 second (memory-mapped, no parsing).
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: Optional[int] = None,
        split_file: Optional[str] = None,
    ):
        data_dir = Path(data_dir)
        if not (data_dir / "metadata.json").exists():
            raise FileNotFoundError(
                f"Preprocessed data not found in {data_dir}. "
                "Run: python scripts/preprocess_to_numpy.py"
            )

        with open(data_dir / "metadata.json") as f:
            self.meta = json.load(f)

        # Memory-mapped arrays (no RAM usage until accessed)
        self.h_peaks = np.load(data_dir / "h_nmr_peaks.npy", mmap_mode="r")
        self.h_lens = np.load(data_dir / "h_nmr_lengths.npy", mmap_mode="r")
        self.c_peaks = np.load(data_dir / "c_nmr_peaks.npy", mmap_mode="r")
        self.c_lens = np.load(data_dir / "c_nmr_lengths.npy", mmap_mode="r")
        self.ir = np.load(data_dir / "ir_spectra.npy", mmap_mode="r")
        self.fps = np.load(data_dir / "mol_fps.npy", mmap_mode="r")

        with open(data_dir / "smiles.json") as f:
            self.smiles = json.load(f)

        total = self.meta["total"]

        # Custom split file (e.g. scaffold split)
        if split_file is not None:
            sp = Path(split_file)
            if not sp.is_absolute():
                sp = data_dir / sp
            with open(sp) as f:
                splits = json.load(f)
            if split in splits:
                self.indices = np.array(splits[split], dtype=np.int64)
            else:
                self.indices = np.arange(total)
            split_label = f"{split}({sp.name})"
        else:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(total)
            n_train = int(total * train_ratio)
            n_val = int(total * val_ratio)
            if split == "train":
                self.indices = indices[:n_train]
            elif split == "val":
                self.indices = indices[n_train:n_train + n_val]
            elif split == "test":
                self.indices = indices[n_train + n_val:]
            else:
                self.indices = np.arange(total)
            split_label = split

        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        print(f"[FastDataset] {total:,} total, {split_label}: {len(self.indices):,} samples (instant load)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        h_len = int(self.h_lens[i])
        c_len = int(self.c_lens[i])

        return {
            "h_nmr_peaks": np.array(self.h_peaks[i, :max(h_len, 1)]),  # (N_h, 7)
            "c_nmr_peaks": np.array(self.c_peaks[i, :max(c_len, 1)]),  # (N_c, 3)
            "ir_spectrum": np.array(self.ir[i]),                         # (1800,)
            "mol_fp": np.array(self.fps[i]),                             # (2048,)
            "smiles": self.smiles[i],
            "modality_mask": np.array([h_len > 0, c_len > 0, True], dtype=np.bool_),
        }


def fast_collate(batch: list) -> dict:
    """Collate with padding for variable-length peaks."""
    from .dataset import H1_FEAT_DIM, C13_FEAT_DIM
    B = len(batch)

    # 1H
    max_h = max(len(b["h_nmr_peaks"]) for b in batch)
    max_h = max(max_h, 1)
    h_dim = batch[0]["h_nmr_peaks"].shape[-1] if len(batch[0]["h_nmr_peaks"]) > 0 else H1_FEAT_DIM
    h_peaks = np.zeros((B, max_h, h_dim), dtype=np.float32)
    h_mask = np.zeros((B, max_h), dtype=np.bool_)
    for i, b in enumerate(batch):
        n = len(b["h_nmr_peaks"])
        if n > 0:
            h_peaks[i, :n] = b["h_nmr_peaks"]
            h_mask[i, :n] = True

    # 13C
    max_c = max(len(b["c_nmr_peaks"]) for b in batch)
    max_c = max(max_c, 1)
    c_dim = batch[0]["c_nmr_peaks"].shape[-1] if len(batch[0]["c_nmr_peaks"]) > 0 else C13_FEAT_DIM
    c_peaks = np.zeros((B, max_c, c_dim), dtype=np.float32)
    c_mask = np.zeros((B, max_c), dtype=np.bool_)
    for i, b in enumerate(batch):
        n = len(b["c_nmr_peaks"])
        if n > 0:
            c_peaks[i, :n] = b["c_nmr_peaks"]
            c_mask[i, :n] = True

    # IR + FP + mask
    ir = np.stack([b["ir_spectrum"] for b in batch])
    fps = np.stack([b["mol_fp"] for b in batch])
    mod_mask = np.stack([b["modality_mask"] for b in batch])

    return {
        "h_nmr_peaks": torch.from_numpy(h_peaks),
        "h_nmr_mask": torch.from_numpy(h_mask),
        "c_nmr_peaks": torch.from_numpy(c_peaks),
        "c_nmr_mask": torch.from_numpy(c_mask),
        "ir_spectrum": torch.from_numpy(ir),
        "mol_fp": torch.from_numpy(fps),
        "modality_mask": torch.from_numpy(mod_mask),
        "smiles": [b["smiles"] for b in batch],
    }


def get_fast_dataloader(
    data_dir: str = "data/processed",
    split: str = "train",
    batch_size: int = 128,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    ds = FastMultimodalDataset(data_dir, split=split, max_samples=max_samples, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=fast_collate,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
