#!/usr/bin/env python3
"""
Training script for NMR-MultiFuse v3.

Supports 3-stage pretraining:
  Stage 1: Pairwise contrastive alignment (1H-13C, 1H-IR, 13C-IR)
  Stage 2: Joint 3-modal contrastive (only full-modal samples)
  Stage 3: Full data + CAPS robust training (with modality masking)

Usage:
    # Stage 1
    python src/train.py --stage 1 --epochs 15 --save_dir checkpoints/stage1

    # Stage 2 (resume from stage 1)
    python src/train.py --stage 2 --resume checkpoints/stage1/best.pt --epochs 15

    # Stage 3 (resume from stage 2)
    python src/train.py --stage 3 --resume checkpoints/stage2/best.pt --epochs 20

    # Debug (quick test)
    python src/train.py --debug --max_steps 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path as _P
if (_P("data/processed/metadata.json").exists()):
    from src.data.fast_dataset import get_fast_dataloader as get_dataloader
    _USING_FAST = True
else:
    from src.data.dataset import get_dataloader
    _USING_FAST = False
from src.model.model import build_model

# Optional: autoresearch metric writer
try:
    from autoresearch.metric_writer import MetricWriter
    HAS_METRIC_WRITER = True
except ImportError:
    HAS_METRIC_WRITER = False

# Optional: wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    p = argparse.ArgumentParser(description="NMR-MultiFuse v3 Training")

    # Data
    p.add_argument("--data_dir", default="data/multimodal_spectroscopic",
                    help="Path to multimodal spectroscopic dataset")
    p.add_argument("--num_workers", type=int, default=4)

    # Training
    p.add_argument("--stage", type=int, default=3, choices=[1, 2, 3],
                    help="Training stage (1=pairwise, 2=joint, 3=robust)")
    p.add_argument("--ablation", default=None, choices=["no_caps", "mmp_only", "proxy_only", "spectre_dropout"],
                    help="Run ablation variant instead of full model")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_encoder_layers", type=int, default=4)
    p.add_argument("--num_inducing", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)

    # CAPS / Loss
    p.add_argument("--mask_prob", type=float, default=0.5)
    p.add_argument("--contrastive_temp", type=float, default=0.07)
    p.add_argument("--inter_weight", type=float, default=1.0)
    p.add_argument("--calib_weight", type=float, default=0.1)

    # Checkpointing
    p.add_argument("--resume", default=None, help="Resume from checkpoint")
    p.add_argument("--save_dir", default="checkpoints/default")
    p.add_argument("--save_every", type=int, default=5, help="Save every N epochs")

    # Logging
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--metrics_file", default="results/train_metrics.json")
    p.add_argument("--log_every", type=int, default=50, help="Log every N steps")

    # Debug
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--split_file", default=None,
                    help="Path to JSON with {train,val,test} index lists (e.g. scaffold_split.json)")
    p.add_argument("--seed", type=int, default=None,
                    help="Global seed (Python/NumPy/PyTorch) — set before model build, "
                         "DataLoader, and training loop for reproducibility / multi-seed runs.")

    return p.parse_args()


def get_stage_config(args):
    """Adjust model config based on training stage."""
    cfg = {
        "dim": args.dim,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_inducing": args.num_inducing,
        "dropout": args.dropout,
        "contrastive_temp": args.contrastive_temp,
        "inter_weight": args.inter_weight,
        "calib_weight": args.calib_weight,
    }

    if args.stage == 1:
        # Pairwise: no CAPS masking, inter-modal loss only
        cfg["mask_prob"] = 0.0
        cfg["calib_weight"] = 0.0
    elif args.stage == 2:
        # Joint: light masking to warm up CAPS
        cfg["mask_prob"] = 0.1
        cfg["calib_weight"] = 0.01
    else:
        # Stage 3: full CAPS
        cfg["mask_prob"] = args.mask_prob
        cfg["calib_weight"] = args.calib_weight

    return cfg


def train_one_epoch(model, criterion, dataloader, optimizer, scheduler, device, args, epoch, global_step):
    model.train()
    total_loss = 0.0
    total_sm = 0.0
    total_inter = 0.0
    total_calib = 0.0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(dataloader):
        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward
        out = model(batch)
        loss_dict = criterion(
            out["fused_repr"], out["mol_repr"],
            out["cls_tokens"], out["modality_mask"],
            out["calib_loss"],
        )
        loss = loss_dict["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Accumulate
        total_loss += loss.item()
        total_sm += loss_dict["loss_spect_mol"]
        total_inter += loss_dict["loss_inter_modal"]
        total_calib += loss_dict["loss_calib"]
        n_batches += 1
        global_step += 1

        # Log
        if step % args.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  [E{epoch}][{step}/{len(dataloader)}] "
                  f"loss={loss.item():.4f} sm={loss_dict['loss_spect_mol']:.4f} "
                  f"inter={loss_dict['loss_inter_modal']:.4f} calib={loss_dict['loss_calib']:.4f} "
                  f"lr={lr:.6f} {elapsed:.0f}s")

            if HAS_WANDB and args.wandb_project:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/loss_spect_mol": loss_dict["loss_spect_mol"],
                    "train/loss_inter_modal": loss_dict["loss_inter_modal"],
                    "train/loss_calib": loss_dict["loss_calib"],
                    "train/lr": lr,
                    "train/global_step": global_step,
                })

        if args.max_steps and global_step >= args.max_steps:
            break

    avg_loss = total_loss / max(n_batches, 1)
    avg_sm = total_sm / max(n_batches, 1)
    avg_inter = total_inter / max(n_batches, 1)
    avg_calib = total_calib / max(n_batches, 1)

    return {
        "loss": avg_loss,
        "loss_spect_mol": avg_sm,
        "loss_inter_modal": avg_inter,
        "loss_calib": avg_calib,
        "global_step": global_step,
    }


@torch.no_grad()
def validate(model, criterion, dataloader, device, args):
    model.eval()
    total_loss = 0.0
    total_sm = 0.0
    total_inter = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(batch)
        loss_dict = criterion(
            out["fused_repr"], out["mol_repr"],
            out["cls_tokens"], out["modality_mask"],
            out["calib_loss"],
        )
        total_loss += loss_dict["loss"].item()
        total_sm += loss_dict["loss_spect_mol"]
        total_inter += loss_dict["loss_inter_modal"]
        n_batches += 1

        if args.max_steps and n_batches >= args.max_steps:
            break

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_loss_spect_mol": total_sm / max(n_batches, 1),
        "val_loss_inter_modal": total_inter / max(n_batches, 1),
    }


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    if args.seed is not None:
        import random
        import numpy as _np
        random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[SEED] global seed = {args.seed}")

    if args.debug:
        args.max_samples = args.max_samples or 500
        args.max_steps = args.max_steps or 10
        args.log_every = 1
        args.num_workers = 0
        print("[DEBUG MODE]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build model
    cfg = get_stage_config(args)
    if args.ablation:
        from src.model.model_ablation import build_ablation_model
        model, criterion = build_ablation_model(args.ablation, cfg)
        print(f"[ABLATION MODE: {args.ablation}]")
    else:
        model, criterion = build_model(cfg)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.1f}M")
    print(f"Stage: {args.stage}, mask_prob={cfg['mask_prob']}, calib_weight={cfg['calib_weight']}")

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Resumed from {args.resume} (epoch {ckpt.get('epoch', '?')})")
        start_epoch = ckpt.get("epoch", 0)

    # Data
    data_dir = "data/processed" if _USING_FAST else args.data_dir
    print(f"Loading data from {data_dir} ({'fast numpy' if _USING_FAST else 'parquet'})...")
    extra = {}
    if args.split_file is not None and _USING_FAST:
        extra["split_file"] = args.split_file
    train_loader = get_dataloader(
        data_dir, split="train", batch_size=args.batch_size,
        num_workers=args.num_workers, max_samples=args.max_samples, **extra,
    )
    val_loader = get_dataloader(
        data_dir, split="val", batch_size=args.batch_size,
        num_workers=args.num_workers, max_samples=args.max_samples, **extra,
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Logging
    if HAS_WANDB and args.wandb_project and not args.debug:
        wandb.init(project=args.wandb_project, config=vars(args))

    metric_writer = None
    if HAS_METRIC_WRITER:
        metric_writer = MetricWriter(args.metrics_file)

    # Save dir
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump({**cfg, **vars(args)}, f, indent=2, default=str)

    # Training loop
    best_val_loss = float("inf")
    global_step = 0

    print(f"\n{'='*60}")
    print(f"  Training Stage {args.stage}: {args.epochs} epochs")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t_epoch = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler,
            device, args, epoch, global_step,
        )
        global_step = train_metrics["global_step"]

        # Validate
        val_metrics = validate(model, criterion, val_loader, device, args)

        epoch_time = time.time() - t_epoch
        print(f"\n[Epoch {epoch}] "
              f"train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['val_loss']:.4f} "
              f"time={epoch_time:.0f}s")

        # Metric writer (for autoresearch supervisor)
        if metric_writer:
            metric_writer.log(epoch, {
                "loss": train_metrics["loss"],
                "val_loss": val_metrics["val_loss"],
                "loss_spect_mol": train_metrics["loss_spect_mol"],
                "loss_inter_modal": train_metrics["loss_inter_modal"],
                "loss_calib": train_metrics["loss_calib"],
                "val_loss_spect_mol": val_metrics["val_loss_spect_mol"],
                "val_loss_inter_modal": val_metrics["val_loss_inter_modal"],
                "lr": optimizer.param_groups[0]["lr"],
            }, direction_map={
                "loss": "lower", "val_loss": "lower",
                "loss_spect_mol": "lower", "loss_inter_modal": "lower",
            })

        # WandB
        if HAS_WANDB and args.wandb_project and not args.debug:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_metrics["loss"],
                "val/loss": val_metrics["val_loss"],
                "val/loss_spect_mol": val_metrics["val_loss_spect_mol"],
                "val/loss_inter_modal": val_metrics["val_loss_inter_modal"],
            })

        # Save checkpoint
        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_metrics["val_loss"],
            "best_val_loss": best_val_loss,
            "config": cfg,
            "args": vars(args),
        }

        if is_best:
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  >> New best val_loss={best_val_loss:.4f}, saved to {save_dir}/best.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch+1}.pt")

        # Always save latest
        torch.save(ckpt, save_dir / "latest.pt")

        if args.max_steps and global_step >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping.")
            break

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to {save_dir}/")

    if HAS_WANDB and args.wandb_project and not args.debug:
        wandb.finish()


if __name__ == "__main__":
    main()
