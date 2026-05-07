#!/bin/bash
set -euo pipefail

# SPECTRE-style data-type dropout baseline (C2 from review)
# Same framework, same data, same epochs — only missing-modality mechanism differs

PYTHON="/opt/data/private/multifuse/bin/python"
PROJECT="/opt/data/private/xrd2c_v2/idea_e"
STAGE2_CKPT="checkpoints/stage2/best.pt"

cd "$PROJECT"

echo "================================================================"
echo "  SPECTRE Dropout Baseline: 100 epochs"
echo "  $(date)"
echo "================================================================"

SAVE_DIR="checkpoints/ablation_spectre_dropout_100ep"
METRICS_FILE="results/ablation_spectre_dropout_100ep_metrics.json"
EVAL_FILE="results/ablation_spectre_dropout_100ep_mms.json"

if [ -f "$EVAL_FILE" ]; then
    echo "SKIP: $EVAL_FILE already exists"
    exit 0
fi

# Train
$PYTHON src/train.py \
    --stage 3 \
    --ablation spectre_dropout \
    --resume "$STAGE2_CKPT" \
    --epochs 100 \
    --batch_size 128 \
    --lr 2e-4 \
    --mask_prob 0.5 \
    --num_workers 4 \
    --save_dir "$SAVE_DIR" \
    --metrics_file "$METRICS_FILE" \
    --log_every 500

# Eval
$PYTHON src/eval.py \
    --checkpoint "${SAVE_DIR}/best.pt" \
    --benchmark mms \
    --report_json "$EVAL_FILE" \
    --output_metrics

echo "Done: SPECTRE dropout baseline -> $EVAL_FILE"
echo "$(date)"
