#!/bin/bash
set -euo pipefail

# 100-epoch ablation experiments (C1 from review)
# Fair comparison: same epochs as best model (100ep from stage2)

PYTHON="/opt/data/private/multifuse/bin/python"
PROJECT="/opt/data/private/xrd2c_v2/idea_e"
STAGE2_CKPT="checkpoints/stage2/best.pt"
EPOCHS=100

cd "$PROJECT"

for ABLATION in no_caps mmp_only proxy_only; do
    echo ""
    echo "================================================================"
    echo "  100EP ABLATION: $ABLATION"
    echo "  $(date)"
    echo "================================================================"

    SAVE_DIR="checkpoints/ablation_${ABLATION}_100ep"
    METRICS_FILE="results/ablation_${ABLATION}_100ep_metrics.json"
    EVAL_FILE="results/ablation_${ABLATION}_100ep_mms.json"

    # Skip if already done
    if [ -f "$EVAL_FILE" ]; then
        echo "SKIP: $EVAL_FILE already exists"
        continue
    fi

    # Train
    echo "Training $ABLATION for $EPOCHS epochs..."
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
        --log_every 500

    # Eval
    echo "Evaluating $ABLATION..."
    $PYTHON src/eval.py \
        --checkpoint "${SAVE_DIR}/best.pt" \
        --benchmark mms \
        --report_json "$EVAL_FILE" \
        --output_metrics

    echo "Done: $ABLATION -> $EVAL_FILE"
    echo "$(date)"
done

echo ""
echo "================================================================"
echo "  ALL 100EP ABLATIONS COMPLETE — $(date)"
echo "================================================================"

# Summary
$PYTHON -c "
import json, glob
print(f'{'Variant':<20} {'S1':>6} {'S3':>6} {'S7':>6} {'Avg':>6}')
print('-' * 50)
for f in sorted(glob.glob('results/ablation_*_100ep_mms.json')):
    name = f.split('ablation_')[1].split('_100ep')[0]
    with open(f) as fh:
        r = json.load(fh)
    s = r.get('summary', {})
    print(f'{name:<20} {r.get(\"S1\",{}).get(\"top1\",0):>6.4f} {r.get(\"S3\",{}).get(\"top1\",0):>6.4f} {r.get(\"S7\",{}).get(\"top1\",0):>6.4f} {s.get(\"avg_missing_top1\",0):>6.4f}')
# Add full CAPS 100ep for comparison
with open('results/mms_bench_exp7_100ep.json') as f:
    r = json.load(f)
s = r['summary']
print(f'{'full_caps_100ep':<20} {r[\"S1\"][\"top1\"]:>6.4f} {r[\"S3\"][\"top1\"]:>6.4f} {r[\"S7\"][\"top1\"]:>6.4f} {s[\"avg_missing_top1\"]:>6.4f}')
"
