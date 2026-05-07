# Missing-Modality Multi-Modal Spectroscopy — Code Supplement

This is the supplementary code package for the NeurIPS 2026 submission:

**"Missing-Modality Learning in Multi-Modal Spectroscopy: A Controlled
Study under Matched Compute"**

It contains the minimum set of code, configuration, and result files needed
to reproduce every number reported in the paper.

The trained checkpoints (~45 GB total) and the preprocessed dataset
(~13 GB of `.npy` mmaps) are **not** shipped here — they are reconstructed
from the public sources listed in `data/README.md` plus the scripts under
`scripts/`. Anything else needed to reach our reported numbers is in this
zip.

------------------------------------------------------------
## 1. Directory layout
------------------------------------------------------------

```
code_supplement/
├── README.md                  this file
├── requirements.txt           pip dependencies
├── src/                       core library (model, training, evaluation)
│   ├── train.py               3-stage pretrain + missing-modality finetune
│   ├── eval.py                MMS-Bench (S1..S7) evaluator
│   ├── data/                  Dataset and DataLoader utilities
│   └── model/                 encoders, fusion, CAPS module, losses
├── scripts/                   analysis / ablation / cross-distribution scripts
│   ├── compute_scaffold_split.py
│   ├── preprocess_to_numpy.py
│   ├── bootstrap_*.py
│   ├── q1a_clamp_alpha.py     (clamp-α=0 verification)
│   ├── q5_bootstrap_full_79k.py
│   ├── q9_compute_overhead.py
│   ├── q10_feature_norm_crossdist.py
│   ├── modality_loo.py
│   ├── library_size_sweep.py
│   ├── per_class_breakdown_5scenarios.py
│   ├── analyze_gate_*.py
│   ├── eval_real_nmr*.py / eval_npmrd.py
│   ├── finetune_real_nmr.py / finetune_scaffold_real.py
│   └── run_*.sh               turn-key launcher scripts
├── configs/                   experiment driver scripts
│   ├── autoresearch.sh        canonical training entrypoint
│   ├── autoresearch.checks.sh correctness gate (S1 ≥ 0.90)
│   └── autoresearch.md        session log of hyperparameter exploration
├── autoresearch/              minimal runtime helper (metric writer)
├── data/
│   ├── README.md              how to obtain & preprocess the raw data
│   └── data_metadata.json     preprocessed-corpus metadata
└── results/                   the JSON files cited by every number in the paper
```

------------------------------------------------------------
## 2. Environment
------------------------------------------------------------

Python 3.10, CUDA 12.x, 2×NVIDIA H100 80 GB used for the reported numbers
(any single H100 / A100 also works for inference and for finetuning).

```bash
conda create -n mms-caps python=3.10 -y
conda activate mms-caps
pip install -r requirements.txt
```

------------------------------------------------------------
## 3. Reproducing the paper end-to-end
------------------------------------------------------------

```bash
# (a) Acquire raw data (see data/README.md)
#     - Multimodal Spectroscopic Dataset (Alberts et al., NeurIPS 2024)
#     - nmrshiftdb2 + SDBS (real-NMR transfer)
#     - NP-MRD (cross-distribution probe)

# (b) Preprocess raw parquet → numpy mmaps + scaffold split
python scripts/preprocess_to_numpy.py
python scripts/compute_scaffold_split.py

# (c) Stage-3 main run (Full CAPS, scaffold split, seed 42)
bash scripts/run_scaffold_caps.sh           # 12-14 h on 2×H100

# (d) Five-strategy ablation (zero-fill, MMP, proxy-only,
#     SPECTRE-dropout, Full CAPS) × {random, scaffold} split
bash scripts/run_ablations_100ep.sh
bash scripts/launch_both_scaffold.sh

# (e) Multi-seed reproducibility (seed 43)
bash scripts/run_scaffold_caps_seed.sh

# (f) Mechanism / interpretability analyses
bash scripts/run_analyses_gpu0.sh           # gate, clamp-α, LOO, overhead
bash scripts/run_analyses_gpu1.sh           # bootstrap, library sweep, per-FG

# (g) Cross-distribution transfer
bash scripts/np_mrd_full_chain.sh

# Per-experiment metrics land in results/*.json — see Section 5 below.
```

A single command from scratch:

```bash
bash configs/autoresearch.sh
```

reproduces the headline Full CAPS run. The `configs/autoresearch.md`
file records the full hyperparameter exploration log; only the final
`HYPERPARAMS` block in `configs/autoresearch.sh` is needed to hit the
reported numbers.

------------------------------------------------------------
## 4. Headline hyperparameters (final config)
------------------------------------------------------------

```
ablation         = full_caps
mask_prob        = 0.5    (modality dropout per training step)
calib_weight     = 0.10   (KL of CAPS-imputed vs encoder embedding)
contrastive_temp = 0.07
inter_weight     = 1.0
lr               = 2e-4
epochs           = 100    (Stage 3, resumed from Stage 2 checkpoint)
batch_size       = 1024
optimizer        = AdamW, weight_decay 1e-4
scheduler        = CosineAnnealingLR
seed             = 42 (paper) / 43 (replication)
fp32             = (no AMP — exact reproduction)
total params     = 13.79 M
```

Stage 1 → 15 epochs pairwise contrastive; Stage 2 → 15 epochs joint
3-modal contrastive on full-modal samples; Stage 3 → 100 epochs on full
data with modality masking.

------------------------------------------------------------
## 5. Result-file ↔ paper-claim map
------------------------------------------------------------

Every JSON in `results/` is the canonical evidence for one of the paper's
claims. Tables / figures in the paper are produced from these files.

| Result file | Claim / where used in paper |
|-------------|------------------------------|
| `mms_bench_exp7_100ep.json` | Headline 5×7 strategy×scenario table (Sec. 5.3 Table 2 / Sec. 5.4) |
| `mms_bench_stage3_baseline.json` | Stage-3 baseline reference run |
| `bootstrap_scaffold.json`, `bootstrap_ci.json` | 95% CI on 20K-subsample (Sec. 5.4) |
| `q5_bootstrap_79K.json` | 95% CI on full 79 K test pool (Appendix) |
| `bootstrap_full_caps_100ep.json` | Bootstrap on the headline 100-epoch model |
| `gate_analysis.json`, `gate_analysis_scaffold.json` | Gate-value distribution (Sec. 5.5) |
| `q1a_alpha_clamped.json` | Clamp-α=0 verification (Sec. 5.5) |
| `q9_compute_overhead.json` | Compute / memory overhead profile (Sec. 5.5) |
| `q10_feature_norm.json` | z-norm cross-distribution probe (Sec. 5.6 / Appendix) |
| `modality_loo.json` | Modality leave-one-out marginal (Sec. 5.5) |
| `library_size_sweep.json` | Library-size sensitivity (Appendix) |
| `per_class_scaffold_5scenarios.json`, `per_class_scaffold.json` | Per-functional-group breakdown (Appendix) |
| `eval_real_nmr_zeroshot.json` | Zero-shot transfer to real NMR (Sec. 5.6) |
| `eval_real_nmr_scaffold.json` | Real-NMR finetuned eval, scaffold-pretrained model |
| `eval_npmrd.json` | NP-MRD cross-distribution probe (Sec. 5.6) |
| `modality_coverage_report.json` | Corpus statistics: 794,403 samples, 3-modal coverage |
| `ablation_*_mms.json`, `ablation_*_metrics.json` | 4 alternative strategies for the 5-strategy ablation |
| `autoresearch_eval_exp_*.json` | Hyperparameter exploration results (autoresearch trail) |

------------------------------------------------------------
## 6. Hardware & wall-clock
------------------------------------------------------------

| Stage | GPUs | Wall-clock |
|-------|------|------------|
| Stage 1 (pairwise contrastive, 15 ep) | 2 × H100 80 GB | ~3 h |
| Stage 2 (3-modal joint, 15 ep)        | 2 × H100 80 GB | ~3 h |
| Stage 3 (full data + CAPS, 100 ep)    | 2 × H100 80 GB | 12–14 h |
| Eval on 79 K MMS-Bench pool           | 1 × H100       | ~3 min |
| Real-NMR finetune (30 ep)             | 1 × H100       | ~25 min |

Peak GPU memory ≈ 38 GB at batch 1024 fp32.

------------------------------------------------------------
## 7. Licence and data attribution
------------------------------------------------------------

The code in this supplement is released under the MIT licence (see
`LICENSE` if attached at upload time, or the licence statement in the
paper's reproducibility checklist).

Datasets are not redistributed — see `data/README.md` for upstream
sources and licences.
