#!/bin/bash
set -u
cd /opt/data/private/xrd2c_v2/idea_e
PY="/opt/data/private/multifuse/bin/python"

echo "=== [GPU0] $(date) bootstrap_scaffold ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/bootstrap_scaffold.py 2>&1
echo "=== [GPU0] $(date) gate analysis scaffold ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/analyze_gate_scaffold.py 2>&1
echo "=== [GPU0] $(date) DONE ==="
