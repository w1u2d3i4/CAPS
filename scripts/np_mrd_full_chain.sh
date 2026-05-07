#!/bin/bash
# Wait for NP-MRD download → preprocess → eval → done flag.
set -u
PROJECT=/opt/data/private/xrd2c_v2/idea_e
NPDIR=/opt/data/private/np_mrd_downloads
PY=/opt/data/private/multifuse/bin/python
cd $PROJECT

echo "[chain] $(date) waiting for nmrml zip..."
until [ -f $NPDIR/predicted_nmrml_spectra.zip ] && [ $(stat -c%s $NPDIR/predicted_nmrml_spectra.zip) -gt 2890000000 ]; do
  sleep 120
done
echo "[chain] $(date) nmrml ready, size=$(stat -c%s $NPDIR/predicted_nmrml_spectra.zip)"

if [ ! -d $NPDIR/extracted ] || [ -z "$(ls $NPDIR/extracted 2>/dev/null)" ]; then
  echo "[chain] extracting JSON shards..."
  mkdir -p $NPDIR/extracted
  for f in $NPDIR/npmrd_*_json.zip; do
    unzip -q -o $f -d $NPDIR/extracted/ &
  done
  wait
fi

echo "[chain] $(date) preprocessing NP-MRD..."
$PY scripts/preprocess_npmrd.py
echo "[chain] $(date) preprocessing done"

if [ ! -f data/np_mrd_processed/metadata.json ]; then
  echo "[chain] preprocess FAILED — no metadata.json"
  exit 1
fi

echo "[chain] $(date) eval starting..."
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_npmrd.py
echo "[chain] $(date) ALL DONE"
touch $NPDIR/.full_chain_done
