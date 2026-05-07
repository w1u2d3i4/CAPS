#!/bin/bash
set -euo pipefail

# Scaffold-split retrain for SPECTRE-dropout (best of our methods).
# Uses Bemis-Murcko scaffold split → directly comparable to NeurIPS 2024 (73.38%).

PYTHON="/opt/data/private/multifuse/bin/python"
PROJECT="/opt/data/private/xrd2c_v2/idea_e"
STAGE2_CKPT="checkpoints/stage2/best.pt"
SPLIT_FILE="data/processed/scaffold_split.json"

cd "$PROJECT"

GPU="${GPU:-0}"
TAG="${TAG:-scaffold_spectre_dropout_100ep}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-512}"

SAVE_DIR="checkpoints/$TAG"
METRICS="results/${TAG}_metrics.json"
EVAL_OUT="results/${TAG}_mms.json"

echo "================================================================"
echo "  Scaffold-split SPECTRE-dropout: $EPOCHS ep, batch $BATCH, GPU $GPU"
echo "  $(date)"
echo "================================================================"

if [ -f "$EVAL_OUT" ]; then
  echo "SKIP: $EVAL_OUT already exists"
  exit 0
fi

if [ ! -f "$SPLIT_FILE" ]; then
  echo "ERROR: scaffold split not found at $SPLIT_FILE"
  exit 1
fi

# Train
CUDA_VISIBLE_DEVICES="$GPU" $PYTHON src/train.py \
  --stage 3 \
  --ablation spectre_dropout \
  --resume "$STAGE2_CKPT" \
  --split_file scaffold_split.json \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --lr 2e-4 \
  --mask_prob 0.5 \
  --num_workers 4 \
  --save_dir "$SAVE_DIR" \
  --metrics_file "$METRICS" \
  --log_every 500

# Eval on scaffold-split test set (this is the public-comparable number)
CUDA_VISIBLE_DEVICES="$GPU" $PYTHON src/eval.py \
  --checkpoint "${SAVE_DIR}/best.pt" \
  --benchmark mms \
  --split test \
  --split_file scaffold_split.json \
  --report_json "$EVAL_OUT" \
  --output_metrics

echo "Done: $EVAL_OUT"
echo "$(date)"
