#!/bin/bash
set -euo pipefail

# Run all structural ablation experiments
# Each trains Stage 3 from Stage 2 checkpoint with one component removed

PYTHON="/opt/data/private/multifuse/bin/python"
PROJECT="/opt/data/private/xrd2c_v2/idea_e"
STAGE2_CKPT="checkpoints/stage2/best.pt"
EPOCHS=20  # Same as original baseline for fair comparison

cd "$PROJECT"

for ABLATION in no_caps mmp_only proxy_only; do
    echo ""
    echo "================================================================"
    echo "  ABLATION: $ABLATION ($EPOCHS epochs)"
    echo "================================================================"

    SAVE_DIR="checkpoints/ablation_${ABLATION}"
    METRICS_FILE="results/ablation_${ABLATION}_metrics.json"
    EVAL_FILE="results/ablation_${ABLATION}_mms.json"

    # Train
    echo "Training..."
    $PYTHON src/train.py \
        --stage 3 \
        --ablation "$ABLATION" \
        --resume "$STAGE2_CKPT" \
        --epochs "$EPOCHS" \
        --batch_size 128 \
        --lr 2e-4 \
        --mask_prob 0.5 \
        --num_workers 4 \
        --save_dir "$SAVE_DIR" \
        --metrics_file "$METRICS_FILE" \
        --log_every 200

    # Eval
    echo "Evaluating..."
    $PYTHON src/eval.py \
        --checkpoint "${SAVE_DIR}/best.pt" \
        --benchmark mms \
        --report_json "$EVAL_FILE" \
        --output_metrics

    echo "Done: $ABLATION -> $EVAL_FILE"
done

echo ""
echo "================================================================"
echo "  ALL ABLATIONS COMPLETE"
echo "================================================================"

# Print summary
$PYTHON -c "
import json, glob
print(f'{'Ablation':<15} {'S1':>6} {'S3':>6} {'S7':>6} {'Avg':>6}')
print('-' * 45)
for f in sorted(glob.glob('results/ablation_*_mms.json')):
    name = f.split('ablation_')[1].split('_mms')[0]
    with open(f) as fh:
        r = json.load(fh)
    s = r.get('summary', {})
    print(f'{name:<15} {r.get(\"S1\",{}).get(\"top1\",0):>6.3f} {r.get(\"S3\",{}).get(\"top1\",0):>6.3f} {r.get(\"S7\",{}).get(\"top1\",0):>6.3f} {s.get(\"avg_missing_top1\",0):>6.3f}')
"
