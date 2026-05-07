#!/usr/bin/env python3
"""Fine-tune scaffold-trained CAPS on real NMR data (nmrshiftdb2 + SDBS).

Reports zero-shot vs fine-tuned top-k on a held-out 20% real-NMR split.
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

CKPT = "checkpoints/scaffold_full_caps_100ep/best.pt"
DATA_DIR = "data/real_nmr"
SAVE_DIR = "checkpoints/finetune_scaffold_real"
EPOCHS = 20
BATCH_SIZE = 256
LR = 5e-5
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, CKPT={CKPT}")

with open(Path(DATA_DIR) / "smiles.json") as f:
    smiles = json.load(f)

from rdkit import Chem
canon_to_idx = {}
for i, smi in enumerate(smiles):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        cs = Chem.MolToSmiles(m)
        canon_to_idx.setdefault(cs, []).append(i)
    except Exception:
        continue
unique_smiles = list(canon_to_idx.keys())
print(f"Unique canonical SMILES: {len(unique_smiles)}")

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(unique_smiles))
n_tr = int(0.8 * len(unique_smiles))
tr_set = set(unique_smiles[i] for i in perm[:n_tr])
te_set = set(unique_smiles[i] for i in perm[n_tr:])
tr_idx = [i for cs, ids in canon_to_idx.items() if cs in tr_set for i in ids]
te_idx = [i for cs, ids in canon_to_idx.items() if cs in te_set for i in ids]
print(f"Train: {len(tr_idx)}, Test: {len(te_idx)}")


class RealDS(Dataset):
    def __init__(self, root, indices):
        root = Path(root)
        self.h_peaks = np.load(root/"h_nmr_peaks.npy", mmap_mode="r")
        self.h_lens  = np.load(root/"h_nmr_lengths.npy", mmap_mode="r")
        self.c_peaks = np.load(root/"c_nmr_peaks.npy", mmap_mode="r")
        self.c_lens  = np.load(root/"c_nmr_lengths.npy", mmap_mode="r")
        self.ir      = np.load(root/"ir_spectra.npy", mmap_mode="r")
        self.fps     = np.load(root/"mol_fps.npy", mmap_mode="r")
        with open(root/"smiles.json") as f:
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

train_dl = DataLoader(RealDS(DATA_DIR, tr_idx), batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=4, collate_fn=fast_collate, pin_memory=True)
test_dl  = DataLoader(RealDS(DATA_DIR, te_idx), batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=4, collate_fn=fast_collate, pin_memory=True)
print(f"Train batches: {len(train_dl)}, Test batches: {len(test_dl)}")

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = ckpt.get("config", {})
ablation = ckpt.get("args", {}).get("ablation") or cfg.get("ablation")
if ablation:
    from src.model.model_ablation import build_ablation_model
    model, _ = build_ablation_model(ablation, cfg)
else:
    model, _ = build_model(cfg)
model.load_state_dict(ckpt["model"], strict=False)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.1)


@torch.no_grad()
def eval_topk(model, dl):
    model.eval()
    if hasattr(model, "mask_prob"):
        saved = model.mask_prob; model.mask_prob = 0.0
    all_s, all_m = [], []
    for b in dl:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in b.items()}
        out = model(b)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    if hasattr(model, "mask_prob"):
        model.mask_prob = saved
    s = torch.cat(all_s); m = torch.cat(all_m)
    sim = s @ m.T
    diag = sim.diag().unsqueeze(1)
    ranks = (sim >= diag).sum(1)
    return {f"top{k}": ((ranks <= k).float().mean().item()) for k in (1,5,10)}, s.size(0)


print("\n=== Zero-shot ===")
z, n = eval_topk(model, test_dl)
print(f"  Pool={n}, {z}")

best = 0.0
log = {"checkpoint": CKPT, "zero_shot": z, "n_pool": n, "epochs": []}
for ep in range(1, EPOCHS+1):
    model.train()
    if hasattr(model, "mask_prob"):
        model.mask_prob = 0.5
    t0 = time.time()
    tot, nb = 0.0, 0
    for b in train_dl:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in b.items()}
        out = model(b)
        s_e = F.normalize(out["fused_repr"], dim=-1)
        m_e = F.normalize(out["mol_repr"], dim=-1)
        logits = s_e @ m_e.T / 0.07
        lab = torch.arange(s_e.size(0), device=device)
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += loss.item(); nb += 1
    scheduler.step()
    res, _ = eval_topk(model, test_dl)
    print(f"[Ep {ep:2d}/{EPOCHS}] loss={tot/nb:.4f} test top1={res['top1']:.4f} top5={res['top5']:.4f} ({time.time()-t0:.0f}s)")
    log["epochs"].append({"epoch": ep, "loss": tot/nb, **res})
    if res["top1"] > best:
        best = res["top1"]
        torch.save({"model": model.state_dict(), "config": cfg, "args": {"ablation": ablation}}, f"{SAVE_DIR}/best.pt")
        log["best_top1"] = best
        log["best_epoch"] = ep

with open(f"{SAVE_DIR}/finetune_log.json", "w") as f:
    json.dump(log, f, indent=2)
print(f"\nBest test top1: {best:.4f}")
print(f"Saved: {SAVE_DIR}/finetune_log.json")
