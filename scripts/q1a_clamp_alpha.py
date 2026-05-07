#!/usr/bin/env python3
"""Q1a (round 3 review): Clamp CAPS gate alpha=0 at inference.

Loads the scaffold-trained Full CAPS checkpoint, monkey-patches the CAPS
module's `forward` so alpha is forced to 0 (output = pure proxy_refined),
and re-evaluates MMS-Bench. If the central claim ("projection trains a
better proxy; gate then collapses, so inference uses proxy") holds, this
should produce the SAME numbers as default Full CAPS — directly validating
the paper's training-dynamics interpretation.

Output: results/q1a_alpha_clamped.json
"""
import json, sys, os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import torch
import torch.nn.functional as F
from src.data.fast_dataset import FastMultimodalDataset, fast_collate
from src.model.model import build_model
from torch.utils.data import DataLoader

CKPT_PATH = "checkpoints/scaffold_full_caps_100ep/best.pt"

MMS = {
    "S1": [True, True, True], "S2": [True, False, False],
    "S3": [False, True, False], "S4": [True, True, False],
    "S5": [True, False, True], "S6": [False, True, True],
    "S7": [False, False, True],
}


def patch_caps_alpha_zero(model):
    """Monkey-patch the CAPS module so the gate alpha is always 0.
    Returns a list of restore-callables.
    """
    restores = []
    if not hasattr(model, "missing_module") or model.missing_module is None:
        return restores
    caps = model.missing_module
    if not hasattr(caps, "gate_net"):
        return restores
    original_forward = caps.forward

    def patched_forward(tokens, modality_mask, *a, **kw):
        # call original — we cannot easily intercept the inner alpha
        # so monkey-patch gate_net to return -inf logits → sigmoid → 0.
        orig_gate = caps.gate_net
        class Zero(torch.nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 1), -1e9, device=x.device, dtype=x.dtype)
        caps.gate_net = Zero()
        try:
            out = original_forward(tokens, modality_mask, *a, **kw)
        finally:
            caps.gate_net = orig_gate
        return out

    caps.forward = patched_forward
    restores.append(lambda: setattr(caps, "forward", original_forward))
    return restores


@torch.no_grad()
def encode(model, dl, device, mask_override):
    model.eval()
    if hasattr(model, "mask_prob"):
        model.mask_prob = 0.0
    all_s, all_m = [], []
    for b in dl:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        B = b["mol_fp"].size(0)
        b["modality_mask"] = torch.tensor([mask_override]*B, dtype=torch.bool, device=device)
        out = model(b)
        all_s.append(F.normalize(out["fused_repr"], dim=-1).cpu())
        all_m.append(F.normalize(out["mol_repr"], dim=-1).cpu())
    return torch.cat(all_s), torch.cat(all_m)


def topk(spec, mol, k_list=(1, 5, 10), chunk=2048, device="cuda"):
    s = spec.to(device); m = mol.to(device)
    N = s.size(0)
    correct = {k: 0 for k in k_list}
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        sim = s[i:end] @ m.T
        diag = sim[torch.arange(end-i), torch.arange(i, end)].unsqueeze(1)
        ranks = (sim >= diag).sum(1)
        for k in k_list:
            correct[k] += int((ranks <= k).sum().item())
    return {f"top{k}": correct[k] / N for k in k_list}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model, _ = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)

    ds = FastMultimodalDataset("data/processed", split="test",
                               split_file="scaffold_split.json", max_samples=None)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2,
                    collate_fn=fast_collate, pin_memory=True)
    print(f"Test set: {len(ds)} samples")

    out = {"checkpoint": CKPT_PATH, "n_test": len(ds), "scenarios": {}}

    # First: default (gate as-is, should equal published Full CAPS scaffold numbers)
    print("\n=== Default (gate as learned) ===")
    for sid, mask in MMS.items():
        spec, mol = encode(model, dl, device, mask)
        m = topk(spec, mol)
        print(f"  {sid}: top1={m['top1']:.4f} top5={m['top5']:.4f}")
        out["scenarios"].setdefault("default", {})[sid] = m

    # Then: gate clamped to 0
    print("\n=== Clamped alpha=0 (no projection) ===")
    restores = patch_caps_alpha_zero(model)
    for sid, mask in MMS.items():
        spec, mol = encode(model, dl, device, mask)
        m = topk(spec, mol)
        print(f"  {sid}: top1={m['top1']:.4f} top5={m['top5']:.4f}")
        out["scenarios"].setdefault("alpha_zero", {})[sid] = m
    for r in restores: r()

    # Delta
    deltas = {}
    for sid in MMS:
        d = out["scenarios"]["alpha_zero"][sid]["top1"] - out["scenarios"]["default"][sid]["top1"]
        deltas[sid] = round(d * 100, 2)
    out["delta_top1_pp"] = deltas
    print(f"\n  Δtop1 (pp): {deltas}")
    print(f"  Mean |Δ|: {np.mean(np.abs(list(deltas.values()))):.2f} pp")

    out_path = "results/q1a_alpha_clamped.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
