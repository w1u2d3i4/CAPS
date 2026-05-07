#!/bin/bash
# Wait for SPECTRE dropout eval to complete, then touch a flag file
while true; do
  if [ -f "/opt/data/private/xrd2c_v2/idea_e/results/ablation_spectre_dropout_100ep_mms.json" ]; then
    echo "SPECTRE_COMPLETE $(date)" > /opt/data/private/xrd2c_v2/idea_e/logs/spectre_done.flag
    exit 0
  fi
  sleep 600  # check every 10 min
done
