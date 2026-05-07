"""
Data augmentation for NMR and IR spectra.

Applied during training to improve robustness.
"""

import numpy as np


def augment_nmr_peaks(peaks: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Augment NMR peaks (works for both 1H and 13C).

    Args:
        peaks: (N, D) where D=3 for 1H or D=2 for 13C
        p: probability of applying each augmentation
    """
    if len(peaks) == 0:
        return peaks

    peaks = peaks.copy()
    rng = np.random

    # Chemical shift perturbation: +/- 0.05 ppm
    if rng.random() < p:
        peaks[:, 0] += rng.normal(0, 0.02, size=len(peaks))

    # Intensity scaling: 0.8-1.2x
    if rng.random() < p:
        scale = rng.uniform(0.8, 1.2)
        peaks[:, 1] *= scale

    # Random peak dropout: remove 5-10% of peaks
    if rng.random() < p and len(peaks) > 2:
        n_drop = max(1, int(len(peaks) * rng.uniform(0.05, 0.1)))
        keep_idx = rng.choice(len(peaks), size=len(peaks) - n_drop, replace=False)
        keep_idx.sort()
        peaks = peaks[keep_idx]

    return peaks


def augment_ir_spectrum(spectrum: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Augment IR spectrum.

    Args:
        spectrum: (L,) 1D IR spectrum
        p: probability of applying each augmentation
    """
    if len(spectrum) == 0:
        return spectrum

    spectrum = spectrum.copy()
    rng = np.random
    L = len(spectrum)

    # Gaussian noise
    if rng.random() < p:
        sigma = rng.uniform(0.005, 0.02)
        spectrum += rng.normal(0, sigma, size=L).astype(np.float32)

    # Baseline drift: low-order polynomial
    if rng.random() < p:
        x = np.linspace(-1, 1, L)
        order = rng.randint(1, 4)
        coeffs = rng.normal(0, 0.01, size=order + 1)
        baseline = np.polyval(coeffs, x).astype(np.float32)
        spectrum += baseline

    # Intensity scaling
    if rng.random() < p:
        scale = rng.uniform(0.9, 1.1)
        spectrum *= scale

    return spectrum


def augment_sample(sample: dict, p: float = 0.5) -> dict:
    """Apply all augmentations to a sample dict."""
    sample = dict(sample)  # shallow copy

    if sample["modality_mask"][0]:  # has 1H
        sample["h_nmr_peaks"] = augment_nmr_peaks(sample["h_nmr_peaks"], p)
    if sample["modality_mask"][1]:  # has 13C
        sample["c_nmr_peaks"] = augment_nmr_peaks(sample["c_nmr_peaks"], p)
    if sample["modality_mask"][2]:  # has IR
        sample["ir_spectrum"] = augment_ir_spectrum(sample["ir_spectrum"], p)

    return sample
