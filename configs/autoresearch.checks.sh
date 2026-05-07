#!/bin/bash
set -euo pipefail

# Correctness checks — runs after every passing benchmark.
# Failures block "keep".

PYTHON="/opt/data/private/multifuse/bin/python"

$PYTHON -c "
import json, sys

# Check eval results exist
try:
    import glob
    latest = sorted(glob.glob('results/autoresearch_eval_*.json'))[-1]
    with open(latest) as f:
        r = json.load(f)
except (IndexError, FileNotFoundError):
    print('FAIL: No eval results found')
    sys.exit(1)

# S1 must stay >= 0.90
s1 = r.get('S1', {}).get('top1', 0)
if s1 < 0.90:
    print(f'FAIL: S1 top1 = {s1:.4f} < 0.90 threshold (catastrophic forgetting)')
    sys.exit(1)

# avg_missing must be > 0 (sanity)
avg = r.get('summary', {}).get('avg_missing_top1', 0)
if avg <= 0:
    print(f'FAIL: avg_missing_top1 = {avg} (invalid)')
    sys.exit(1)

print(f'PASS: S1={s1:.4f} avg_missing={avg:.4f}')
"
