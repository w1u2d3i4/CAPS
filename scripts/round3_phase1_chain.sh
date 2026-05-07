#!/bin/bash
# Round-3 review response: Phase 1 (4 inference experiments) → Phase 2 (1 training run).
# All on GPU 1 (GPU 0 is busy with E2 calib ablation).
set -e
PYTHON="/opt/data/private/multifuse/bin/python"
cd /opt/data/private/xrd2c_v2/idea_e

echo ""
echo "=============================================================="
echo "  PHASE 1.1 — Q1a: clamp alpha=0 inference   ($(date))"
echo "=============================================================="
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/q1a_clamp_alpha.py 2>&1 || echo "Q1a FAILED"

echo ""
echo "=============================================================="
echo "  PHASE 1.2 — Q5: bootstrap CI on full 79K   ($(date))"
echo "=============================================================="
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/q5_bootstrap_full_79k.py 2>&1 || echo "Q5 FAILED"

echo ""
echo "=============================================================="
echo "  PHASE 1.3 — Q9: CAPS compute/memory overhead   ($(date))"
echo "=============================================================="
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/q9_compute_overhead.py 2>&1 || echo "Q9 FAILED"

echo ""
echo "=============================================================="
echo "  PHASE 1.4 — Q10: feature normalisation cross-distribution   ($(date))"
echo "=============================================================="
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/q10_feature_norm_crossdist.py 2>&1 || echo "Q10 FAILED"

echo ""
echo "=============================================================="
echo "  PHASE 2 — Q3-partial: Full CAPS scaffold seed=43   ($(date))"
echo "=============================================================="
CUDA_VISIBLE_DEVICES=1 GPU=1 SEED=43 bash scripts/run_scaffold_caps_seed.sh 2>&1 || echo "Q3p FAILED"

echo ""
echo "All round-3 GPU-1 tasks finished at $(date)"
