#!/bin/bash
set -euo pipefail

# ============================================================
# Autoresearch benchmark script
# Modify the hyperparams below, then run this script.
# Outputs METRIC lines parsed by the autoresearch agent.
# ============================================================

PYTHON="/opt/data/private/multifuse/bin/python"
PROJECT_DIR="/opt/data/private/xrd2c_v2/idea_e"
STAGE2_CKPT="checkpoints/stage2/best.pt"

# ------- HYPERPARAMS (agent modifies these) -------
ABLATION=spectre_dropout   # winning strategy from Round 2 ablation
MASK_PROB=0.5
CALIB_WEIGHT=0.0           # SPECTRE-dropout has no calib loss
CONTRASTIVE_TEMP=0.07
INTER_WEIGHT=1.0
LR=2e-4
EPOCHS=200                 # scale-push: 100→200 ep
BATCH_SIZE=1024            # H100 80GB, ~38GB used
# ------- END HYPERPARAMS -------

EXPERIMENT_NAME="exp_SD1_${ABLATION}_mask${MASK_PROB}_ep${EPOCHS}_lr${LR}"
SAVE_DIR="checkpoints/autoresearch/${EXPERIMENT_NAME}"
METRICS_FILE="results/autoresearch_train_metrics.json"

cd "$PROJECT_DIR"
mkdir -p "$SAVE_DIR" results logs

echo "=== TRAINING: $EXPERIMENT_NAME ==="
echo "mask_prob=$MASK_PROB calib=$CALIB_WEIGHT temp=$CONTRASTIVE_TEMP inter=$INTER_WEIGHT lr=$LR epochs=$EPOCHS"

# Train
$PYTHON src/train.py \
    --stage 3 \
    --ablation "$ABLATION" \
    --resume "$STAGE2_CKPT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --mask_prob "$MASK_PROB" \
    --calib_weight "$CALIB_WEIGHT" \
    --contrastive_temp "$CONTRASTIVE_TEMP" \
    --inter_weight "$INTER_WEIGHT" \
    --num_workers 4 \
    --save_dir "$SAVE_DIR" \
    --metrics_file "$METRICS_FILE" \
    --log_every 200

echo "=== EVALUATING: $EXPERIMENT_NAME ==="

# Evaluate
REPORT_JSON="results/autoresearch_eval_${EXPERIMENT_NAME}.json"
$PYTHON src/eval.py \
    --checkpoint "${SAVE_DIR}/best.pt" \
    --benchmark mms \
    --report_json "$REPORT_JSON" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 4 \
    --output_metrics

# Extract val_loss from train metrics
VAL_LOSS=$($PYTHON -c "
import json
with open('$METRICS_FILE') as f:
    m = json.load(f)
print(f'METRIC val_loss={m[\"best\"][\"val_loss\"]:.6f}')
")
echo "$VAL_LOSS"

echo "=== DONE: $EXPERIMENT_NAME ==="
