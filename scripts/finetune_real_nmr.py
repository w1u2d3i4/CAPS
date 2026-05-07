#!/usr/bin/env python3
"""
Fine-tune Exp_SD1 on real NMR data (SDBS + nmrshiftdb2).
Split 80/20 by canonical SMILES, fine-tune with low LR, eval on held-out test.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from src.data.fast_dataset import fast_collate
from src.model.model import build_model
from src.model.model_ablation import build_ablation_model
from src.model.losses import NMRMultiFuseLoss

CKPT = "checkpoints/autoresearch/exp_SD1_spectre_dropout_mask0.5_ep200_lr2e-4/best.pt"
DATA_DIR = "data/real_nmr"
SAVE_DIR = "checkpoints/finetune_real_nmr"
EPOCHS = 30
BATCH_SIZE = 256
LR = 5e-5
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, EPOCHS={EPOCHS}, BS={BATCH_SIZE}, LR={LR}")

# ===== Load data and split =====
with open(Path(DATA_DIR) / "smiles.json") as f:
    smiles = json.load(f)
N_total = len(smiles)
print(f"Total real samples: {N_total}")

# Dedup by canonical SMILES, then split
from rdkit import Chem
canon_to_idx = {}
for i, smi in enumerate(smiles):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        csmi = Chem.MolToSmiles(m)
        canon_to_idx.setdefault(csmi, []).append(i)
    except Exception:
        continue
unique_smiles = list(canon_to_idx.keys())
print(f"Unique canonical SMILES: {len(unique_smiles)}")

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(unique_smiles))
n_train = int(0.8 * len(unique_smiles))
train_smiles = set(unique_smiles[i] for i in perm[:n_train])
test_smiles = set(unique_smiles[i] for i in perm[n_train:])

train_indices = [idx for csmi, idxs in canon_to_idx.items() if csmi in train_smiles for idx in idxs]
test_indices = [idx for csmi, idxs in canon_to_idx.items() if csmi in test_smiles for idx in idxs]
print(f"Train indices: {len(train_indices)}, Test indices: {len(test_indices)}")


class RealDS(Dataset):
    def __init__(self, root, indices):
        root = Path(root)
        self.h_peaks = np.load(root / "h_nmr_peaks.npy", mmap_mode="r")
        self.h_lens = np.load(root / "h_nmr_lengths.npy", mmap_mode="r")
        self.c_peaks = np.load(root / "c_nmr_peaks.npy", mmap_mode="r")
        self.c_lens = np.load(root / "c_nmr_lengths.npy", mmap_mode="r")
        self.ir = np.load(root / "ir_spectra.npy", mmap_mode="r")
        self.fps = np.load(root / "mol_fps.npy", mmap_mode="r")
        with open(root / "smiles.json") as f:
            self.smiles = json.load(f)
        self.indices = list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        h_len, c_len = int(self.h_lens[i]), int(self.c_lens[i])
        return {
            "h_nmr_peaks": np.array(self.h_peaks[i, :max(h_len,1)]),
            "c_nmr_peaks": np.array(self.c_peaks[i, :max(c_len,1)]),
            "ir_spectrum": np.array(self.ir[i]),
            "mol_fp": np.array(self.fps[i]),
            "smiles": self.smiles[i],
            "modality_mask": np.array([h_len>0, c_len>0, False], dtype=np.bool_),
        }

train_ds = RealDS(DATA_DIR, train_indices)
test_ds = RealDS(DATA_DIR, test_indices)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                      collate_fn=fast_collate, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                     collate_fn=fast_collate, pin_memory=True)
print(f"Train batches: {len(train_dl)}, Test batches: {len(test_dl)}")

# ===== Load model =====
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = ckpt.get("config", {})
ablation = "spectre_dropout"
model, criterion = build_ablation_model(ablation, cfg)
model.load_state_dict(ckpt["model"], strict=False)
model = model.to(device)
print(f"Loaded {CKPT}")

# Override criterion if loss attributes differ (use simple InfoNCE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.1)

@torch.no_grad()
def eval_topk(model, dl):
    model.eval()
    all_s, all_m = [], []
    for b in dl:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in b.items()}
        if hasattr(model, "mask_prob"):
            saved = model.mask_prob; model.mask_prob = 0.0
        out = model(b)
        if hasattr(model, "mask_prob"):
            model.mask_prob = saved
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    s = torch.cat(all_s); m = torch.cat(all_m)
    sim = s @ m.T
    diag = sim.diag().unsqueeze(1)
    ranks = (sim >= diag).sum(1)
    return {f"top{k}": ((ranks <= k).float().mean().item()) for k in (1,5,10)}, s.size(0)

# Initial zero-shot eval on test split
print("\n=== Pre-finetune (zero-shot on test split) ===")
res0, n_pool = eval_topk(model, test_dl)
print(f"  Pool: {n_pool}, {res0}")

best_top1 = 0.0
log = {"zero_shot_test": res0, "n_pool": n_pool, "epochs": []}

for epoch in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    n_b = 0
    for b in train_dl:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in b.items()}
        out = model(b)
        s_emb = F.normalize(out["fused_repr"], dim=-1)
        m_emb = F.normalize(out["mol_repr"], dim=-1)
        # InfoNCE
        T = 0.07
        logits = s_emb @ m_emb.T / T
        labels = torch.arange(s_emb.size(0), device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_b += 1
    scheduler.step()
    avg_loss = total_loss / n_b
    elapsed = time.time() - t0
    res, _ = eval_topk(model, test_dl)
    msg = f"[Ep {epoch:3d}/{EPOCHS}] loss={avg_loss:.4f} test top1={res['top1']:.4f} top5={res['top5']:.4f} top10={res['top10']:.4f} ({elapsed:.0f}s)"
    print(msg)
    log["epochs"].append({"epoch": epoch, "loss": avg_loss, **res})
    if res["top1"] > best_top1:
        best_top1 = res["top1"]
        torch.save({"model": model.state_dict(), "config": cfg, "args": {"ablation": ablation}}, f"{SAVE_DIR}/best.pt")
        log["best_top1"] = best_top1
        log["best_epoch"] = epoch

with open(f"{SAVE_DIR}/finetune_log.json", "w") as f:
    json.dump(log, f, indent=2)
print(f"\n=== Done ===")
print(f"Best test top1: {best_top1:.4f}")
print(f"Saved: {SAVE_DIR}/best.pt and finetune_log.json")
