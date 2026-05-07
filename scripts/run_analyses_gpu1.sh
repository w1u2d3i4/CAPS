#!/bin/bash
set -u
cd /opt/data/private/xrd2c_v2/idea_e
PY="/opt/data/private/multifuse/bin/python"

echo "=== [GPU1] $(date) eval_real_nmr_scaffold ==="
CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_real_nmr_scaffold.py 2>&1
echo "=== [GPU1] $(date) per_class_breakdown ==="
CUDA_VISIBLE_DEVICES=1 $PY scripts/per_class_breakdown.py 2>&1
echo "=== [GPU1] $(date) DONE ==="
