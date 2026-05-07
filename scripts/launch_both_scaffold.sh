#!/bin/bash
# Wait for scaffold_split.json then launch SPECTRE-dropout on GPU 0 and Full CAPS on GPU 1.
set -u
cd /opt/data/private/xrd2c_v2/idea_e

echo "[wait] for scaffold_split.json..."
until [ -f data/processed/scaffold_split.json ]; do
  sleep 10
done
echo "[ready] $(date)"

mkdir -p logs

GPU=0 TAG=scaffold_spectre_dropout_100ep nohup bash scripts/run_scaffold_spectre.sh \
  > logs/scaffold_spectre.log 2>&1 &
SP_PID=$!
echo "[spawn] SPECTRE GPU0 pid=$SP_PID"

sleep 8

GPU=1 TAG=scaffold_full_caps_100ep nohup bash scripts/run_scaffold_caps.sh \
  > logs/scaffold_caps.log 2>&1 &
CA_PID=$!
echo "[spawn] CAPS    GPU1 pid=$CA_PID"

sleep 60
echo "--- after 60s warmup ---"
ps -p $SP_PID > /dev/null && echo "[alive] SPECTRE" || echo "[dead]  SPECTRE"
ps -p $CA_PID > /dev/null && echo "[alive] CAPS"    || echo "[dead]  CAPS"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- SPECTRE log tail ---"
tail -30 logs/scaffold_spectre.log
echo "--- CAPS log tail ---"
tail -30 logs/scaffold_caps.log
