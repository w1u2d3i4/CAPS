# Autoresearch: NMR-MultiFuse CAPS Tuning

## Objective
Maximize missing-modality retrieval accuracy (avg of S2-S7 top-1) for NMR-MultiFuse v3.
The model uses CAPS (Confidence-Aware Adaptive Projection-Proxy Switching) to handle missing modalities.
We tune Stage 3 training hyperparameters starting from the Stage 2 checkpoint.

## Metrics
- **Primary**: `avg_missing_top1` (%, higher is better) — mean top-1 retrieval accuracy across 6 missing-modality scenarios (S2-S7)
- **Secondary**:
  - `full_modal_top1` (S1) — must not degrade below 0.90
  - `s3_top1` — 13C-only, weakest scenario, watch closely
  - `s7_top1` — IR-only, second weakest
  - `val_loss` — training convergence indicator

## Baseline (Stage 3 default config)
- avg_missing_top1 = 0.7518
- S1 = 0.9219, S2 = 0.7408, S3 = 0.5501, S4 = 0.8362, S5 = 0.8944, S6 = 0.8700, S7 = 0.6191

## How to Run
`./autoresearch.sh` — trains Stage 3 for N epochs, then evaluates MMS-Bench. Outputs `METRIC name=value` lines.

## Files in Scope
- `src/train.py` — training script, reads all hyperparams from CLI args
- `autoresearch.sh` — benchmark script, modify the hyperparams here
- `src/model/caps.py` — CAPS module, can modify architecture if needed
- `src/model/model.py` — full model, controls mask_prob
- `src/model/losses.py` — loss functions, controls temperature/weights
- `src/model/fusion.py` — fusion module, can swap fusion strategy
- `src/model/encoders.py` — encoders, can adjust depth/width

## Off Limits
- `src/data/` — data pipeline is stable, do not modify
- `src/eval.py` — evaluation script must stay consistent for fair comparison
- `checkpoints/stage1/`, `checkpoints/stage2/` — prior stage checkpoints
- `data/` — all data files

## Constraints
- S1 (full modal) top-1 must stay >= 0.90 (no catastrophic forgetting)
- Single GPU (H100 80GB), batch_size <= 256
- Each experiment: 10 epochs max for search, 20 epochs for final

## What's Been Tried

### Baseline (Stage 3 default)
- mask_prob=0.5, calib_weight=0.1, temp=0.07, inter_weight=1.0, lr=2e-4
- Result: avg_missing=0.7518, S3=0.5501 (worst), S7=0.6191
- Observation: calib_loss went very negative (-0.93), might be too strong

### Exp1: mask_prob=0.7, calib_weight=0.01 -> DISCARD
- Result: avg_missing=0.7383 (worse than baseline 0.7518), ALL scenarios degraded
- S1=0.9168, S3=0.5263 (was 0.5501), S7=0.6059 (was 0.6191)
- Insight: changed two params at once, both hurt. calib=0.01 too weak for CAPS to self-calibrate. mask=0.7 too aggressive without proper calibration support.
- Lesson: change ONE variable at a time.

### Exp2: mask_prob=0.7, calib_weight=0.1 -> DISCARD
- Result: avg_missing=0.7438 (baseline 0.7518), still worse but less than Exp1
- Confirms: mask=0.7 itself is harmful. More masking != better robustness.
- The issue is not exposure frequency but projection quality under single-modal input.

### Exp3: contrastive_temp=0.05 -> DISCARD
- Result: avg_missing=0.7298 (WORST so far), S1=0.9128 near red line
- Sharper temp amplifies CAPS projection errors under missing modalities
- Lower temp = harder negatives = good for full modal, but catastrophic when CAPS produces approximate projections

## Key Insight After 3 Failures
The baseline hyperparams (mask=0.5, calib=0.1, temp=0.07) may already be near-optimal for coarse tuning. All single-param changes in both directions hurt. The bottleneck is likely STRUCTURAL: the CAPS cross-attention with only 1 available modality token as kv has too little information to produce a good projection for the 2 missing modalities. Consider:
1. Stronger inter-modal loss to make source tokens richer
2. Architectural change: add a learned "modality interaction" layer before CAPS
3. More training epochs (baseline had 20, experiments only 10)

## Ideas Backlog
- ~~mask=0.7 + calib=0.01~~ FAILED
- ~~mask=0.7 + calib=0.1~~ FAILED
- ~~temp=0.05~~ FAILED (worst)
### Exp4: inter_weight=2.0 -> DISCARD
- Result: avg_missing=0.7390, S3=0.5384, S7=0.6024
- Stronger inter-modal loss didn't help either.

**PIVOT: All 4 hyperparam experiments failed. Moving to code changes.**

### Exp5 (in progress): CAPS architecture change
- Refined proxy: proxy_refine MLP conditions proxy on available modality context
- Gate bias init: alpha starts ~0.3 (favor proxy initially, learn when projection is good)
- Rationale: with 1 available modality, cross-attention has only 1 kv token — projection quality is inherently low. A context-aware proxy is a better fallback than a static learned vector.

### Exp5: CAPS refined proxy 10ep -> DISCARD
- Result: avg_missing=0.7407, new proxy_refine params randomly initialized, 10ep not enough

### Exp6: CAPS refined proxy 20ep -> DISCARD (barely)
- Result: avg_missing=0.7488, delta only -0.4% from baseline. S6 improved +0.4%.
- KEY: SPECTRE=300ep, NMR2Struct=1500ep. Our total=50ep. All configs undertrained.

## Ideas Backlog
- [DONE] Refined CAPS + 100 epochs → CAPS=0.788, SPECTRE-dropout=0.797 (CAPS refuted)

---

# PIVOT (2026-04-24): SOTA Push on SPECTRE-dropout

## New Objective
Push SPECTRE-dropout from 0.797 → ≥0.81 avg_missing_top1 to claim competitive performance.
Winning strategy: SPECTRE-dropout (simplest, best). Focus tuning there.

## Constraints
- No new data (NP-MRD deferred — needs download)
- 13C DEPT deferred (needs new feature extraction pipeline)
- Must use `--ablation spectre_dropout` in all Stage 3 runs

## Search Space (new experiments)
| Idea | Est. impact | Cost | Priority |
|------|-------------|------|----------|
| **200ep longer training** (scale dominates hypothesis) | +1-2% | 16-20h | HIGH — start first |
| IR encoder stride=1 (Reviewer B) | +0.5-1% on S7 | 12h | HIGH |
| Cosine LR schedule + warmup | +0.3-0.8% | 10h | MED |
| Larger model dim=384 | +0.5-1.5% | 20h | MED |
| Tune mask_prob for SPECTRE specifically (0.3 / 0.4 / 0.6) | +0.3-0.8% | 10h each | LOW |
| EMA model weights | +0.2-0.5% | negligible overhead | LOW |

## What's Been Tried (new round, SPECTRE-dropout focus)
_(none yet)_

## Running Experiment
**Exp_SD1**: SPECTRE-dropout 200ep on CUDA:0 — test scale hypothesis
