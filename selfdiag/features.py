"""
The thirteen handcrafted features. Order is frozen and must match
config.FEATURE_NAMES exactly; the ESP32 C port and the ONNX model both depend
on positional order, so reordering here silently breaks deployment.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import skew, kurtosis

import config

_EPS = 1e-12


def _spectrum(w: np.ndarray, fs: int):
    """One-sided magnitude-squared FFT spectrum and its frequency axis."""
    n = w.size
    win = np.hanning(n)
    spec = np.fft.rfft(w * win)
    psd = (np.abs(spec) ** 2) / (np.sum(win ** 2) * fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, psd


def extract(w: np.ndarray, fs: int) -> np.ndarray:
    """Return a float32 vector of length 13 for one window."""
    w = np.asarray(w, dtype=np.float64)

    # ---- time domain (8)
    mean = float(np.mean(w))
    variance = float(np.var(w))
    rms = float(np.sqrt(np.mean(w ** 2)))
    kurt = float(kurtosis(w, fisher=True, bias=False)) if w.size > 3 else 0.0
    # zero crossings of the mean-removed window, normalised by length
    wc = w - mean
    zcr = float(np.mean(np.abs(np.diff(np.signbit(wc).astype(np.int8)))))
    p2p = float(np.max(w) - np.min(w))
    skw = float(skew(w, bias=False)) if w.size > 2 else 0.0
    mad = float(np.mean(np.abs(w - mean)))

    # ---- frequency domain (5)
    freqs, psd = _spectrum(w, fs)
    total = float(np.sum(psd)) + _EPS

    dominant_freq = float(freqs[int(np.argmax(psd))])

    p_norm = psd / total
    p_norm = np.clip(p_norm, _EPS, None)
    spectral_entropy = float(-np.sum(p_norm * np.log2(p_norm)) / np.log2(p_norm.size))

    lo, hi = config.PPG_BAND_HZ
    band_mask = (freqs >= lo) & (freqs <= hi)
    ppg_band_power = float(np.sum(psd[band_mask]))

    blo, bhi = config.BROAD_BAND_HZ
    broad_mask = (freqs >= blo) & (freqs <= min(bhi, freqs[-1]))
    broad_power = float(np.sum(psd[broad_mask])) + _EPS
    power_ratio = float(ppg_band_power / broad_power)

    spectral_centroid = float(np.sum(freqs * psd) / total)

    out = np.array([
        mean, variance, rms, kurt, zcr, p2p, skw, mad,
        dominant_freq, spectral_entropy, ppg_band_power,
        power_ratio, spectral_centroid,
    ], dtype=np.float32)

    if out.size != config.N_FEATURES:
        raise RuntimeError("feature vector length drifted from config")
    return out


def extract_batch(windows: np.ndarray, fs: int) -> np.ndarray:
    """(n, win) -> (n, 13) float32."""
    windows = np.asarray(windows, dtype=np.float64)
    out = np.empty((windows.shape[0], config.N_FEATURES), dtype=np.float32)
    for i in range(windows.shape[0]):
        out[i] = extract(windows[i], fs)
    return out
