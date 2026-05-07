#!/bin/bash
set -euo pipefail

# Q3-partial: Re-train scaffold Full CAPS with a different SEED for multi-seed variance.
# Usage: SEED=43 GPU=1 bash scripts/run_scaffold_caps_seed.sh
PYTHON="/opt/data/private/multifuse/bin/python"
PROJECT="/opt/data/private/xrd2c_v2/idea_e"
STAGE2_CKPT="checkpoints/stage2/best.pt"
SPLIT_FILE="data/processed/scaffold_split.json"

cd "$PROJECT"

GPU="${GPU:-1}"
SEED="${SEED:-43}"
TAG="${TAG:-scaffold_full_caps_100ep_seed${SEED}}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-512}"

SAVE_DIR="checkpoints/$TAG"
METRICS="results/${TAG}_metrics.json"
EVAL_OUT="results/${TAG}_mms.json"

echo "================================================================"
echo "  Q3-partial — Scaffold Full CAPS, seed=$SEED, $EPOCHS ep, GPU $GPU"
echo "  $(date)"
echo "================================================================"

if [ -f "$EVAL_OUT" ]; then
  echo "SKIP: $EVAL_OUT already exists"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" $PYTHON src/train.py \
  --stage 3 \
  --resume "$STAGE2_CKPT" \
  --split_file scaffold_split.json \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --lr 2e-4 \
  --mask_prob 0.5 \
  --calib_weight 0.1 \
  --seed "$SEED" \
  --num_workers 4 \
  --save_dir "$SAVE_DIR" \
  --metrics_file "$METRICS" \
  --log_every 500

CUDA_VISIBLE_DEVICES="$GPU" $PYTHON src/eval.py \
  --checkpoint "${SAVE_DIR}/best.pt" \
  --benchmark mms \
  --split test \
  --split_file scaffold_split.json \
  --report_json "$EVAL_OUT" \
  --output_metrics

echo "Done: $EVAL_OUT"
echo "$(date)"
