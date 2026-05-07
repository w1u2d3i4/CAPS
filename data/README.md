# Data acquisition and preprocessing

The preprocessed numpy mmaps used by the training code total ~13 GB and
are not redistributed in this supplement. They are reconstructed from the
public sources below using `scripts/preprocess_to_numpy.py` and
`scripts/compute_scaffold_split.py`.

------------------------------------------------------------
## Required corpus
------------------------------------------------------------

### Multimodal Spectroscopic Dataset (NeurIPS 2024)
- Reference: Alberts et al., "Unraveling Molecular Structure: A
  Multi-Modal Spectroscopic Dataset for Chemistry," *NeurIPS 2024*.
- Modalities used here: simulated ¹H NMR peak list, ¹³C NMR peak list,
  IR spectrum (1800-bin).
- After preprocessing: 794,403 molecules with all three modalities.
- Splits: 635,522 / 79,440 / 79,441 (train / val / test).

### Optional: real-NMR transfer corpus
- nmrshiftdb2 + SDBS public spectra. Used for cross-distribution
  finetuning (`scripts/preprocess_real_nmr.py`,
  `scripts/finetune_real_nmr.py`).

### Optional: NP-MRD (natural products)
- Used to test the simulated-vs-real hypothesis on a different
  chemical distribution (`scripts/preprocess_npmrd.py`,
  `scripts/eval_npmrd.py`).

------------------------------------------------------------
## Preprocessed format
------------------------------------------------------------

After `scripts/preprocess_to_numpy.py` finishes, `data/processed/`
contains:

```
data/processed/
├── h_nmr_peaks.npy           (N, 28, 7)  fp32 ¹H peak features
├── h_nmr_lengths.npy         (N,)        int   number of valid peaks
├── c_nmr_peaks.npy           (N, 74, 3)  fp32 ¹³C peak features
├── c_nmr_lengths.npy         (N,)        int
├── ir_spectra.npy            (N, 1800)   fp32 IR vector
├── mol_fps.npy               (N, 2048)   uint8 ECFP4 reference fingerprint
├── smiles.json                            list[str]  canonical SMILES per row
├── scaffold_split.json                    Bemis–Murcko scaffold split indices
└── metadata.json                          {total, max_h_peaks, max_c_peaks,
                                            ir_length, h_feat_dim, c_feat_dim}
```

`metadata.json` for the corpus we used is shipped alongside this README
as `data_metadata.json` for reference:

```json
{
  "total": 794403,
  "max_h_peaks": 28,
  "max_c_peaks": 74,
  "ir_length": 1800,
  "h_feat_dim": 7,
  "c_feat_dim": 3
}
```

Random and scaffold splits are both produced by
`scripts/compute_scaffold_split.py`. The scaffold split partitions
molecules by Bemis–Murcko scaffold so train / val / test do not share
any scaffold — this is the protocol used for every "scaffold" number in
the paper.
