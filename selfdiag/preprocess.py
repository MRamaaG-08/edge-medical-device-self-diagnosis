"""
Filtering, normalisation and windowing.

THE PIPELINE ORDER IS LOAD BEARING. Read this before changing anything.

    SYNTHETIC   raw -> resample -> band-pass -> z-score -> window -> INJECT -> features
    REAL        raw -> resample -> band-pass -> calibrate -> window -> features

Degradation is injected AFTER normalisation. If you inject before, the z-score
divides the attenuation gain straight back out and the ATTENUATED class becomes
mathematically indistinguishable from CLEAN:

    max | zscore(x) - zscore(0.15 * x) |  ==  0.0

That is an exact identity, not an approximation. tests/test_pipeline_order.py
pins it. The same mechanism damages DRIFTED, because the 0.5 Hz high-pass is
designed to remove exactly the baseline wander the class is defined by.

On the real path you cannot reorder anything, so normalisation has to use a
fixed scale measured once from a clean baseline (Calibration below) rather than
per-window statistics. Skipping that makes ATTENUATED silently undetectable in
deployment while still scoring perfectly offline.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from fractions import Fraction

import config


def resample_to(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resample. Exact rational ratio, no interpolation drift."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    ratio = Fraction(int(fs_out), int(fs_in)).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)


def bandpass(x: np.ndarray, fs: int,
             lo: float = config.HP_CUTOFF_HZ,
             hi: float = config.LP_CUTOFF_HZ,
             order: int = config.FILTER_ORDER) -> np.ndarray:
    """
    Two-stage Butterworth, applied zero-phase.

    Implemented as second-order sections. The ba form at these cutoffs on a
    64 Hz signal is numerically marginal and silently changes the coefficients
    between SciPy versions, which is why requirements.txt pins SciPy.
    """
    nyq = fs / 2.0
    sos_hp = butter(order, lo / nyq, btype="highpass", output="sos")
    sos_lp = butter(order, min(hi / nyq, 0.99), btype="lowpass", output="sos")
    y = sosfiltfilt(sos_hp, np.asarray(x, dtype=np.float64))
    y = sosfiltfilt(sos_lp, y)
    return y


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


class Calibration:
    """
    Fixed-scale normalisation for the real-acquisition path.

    Fitted once on a clean baseline segment (the protocol asks for 120 s), then
    applied unchanged to every later window. Because the scale does not track
    the window, a genuine drop in signal amplitude survives normalisation and
    stays visible to the classifier.

    Phase 1 is synthetic and does not use this. It is here so the Phase 1 and
    Phase 2 code paths share one reference implementation.
    """

    def __init__(self) -> None:
        self.mu: float | None = None
        self.sigma: float | None = None

    def fit(self, baseline: np.ndarray) -> "Calibration":
        b = np.asarray(baseline, dtype=np.float64)
        if b.size < 2:
            raise ValueError("calibration baseline too short")
        self.mu = float(b.mean())
        self.sigma = float(b.std())
        if self.sigma <= 0:
            raise ValueError("calibration baseline has zero variance")
        return self

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Calibration.fit must be called first")
        return (np.asarray(x, dtype=np.float64) - self.mu) / self.sigma


def make_windows(x: np.ndarray, fs: int, win_s: float, shift_s: float) -> np.ndarray:
    """
    Slice into overlapping windows. Returns (n_windows, win_samples).

    Incomplete trailing windows are dropped rather than zero-padded; padding
    would create artificial low-variance windows that look like ATTENUATED.
    """
    x = np.asarray(x, dtype=np.float64)
    win = int(round(win_s * fs))
    shift = int(round(shift_s * fs))
    if x.size < win:
        return np.empty((0, win), dtype=np.float64)
    starts = np.arange(0, x.size - win + 1, shift)
    out = np.empty((starts.size, win), dtype=np.float64)
    for i, s in enumerate(starts):
        out[i] = x[s:s + win]
    return out


def prepare_record(raw: np.ndarray, fs_in: int, fs_out: int,
                   win_s: float, shift_s: float) -> np.ndarray:
    """
    Full synthetic-path front end for one record, up to but NOT including
    degradation injection.

        raw -> resample -> band-pass -> z-score -> window

    Returns clean normalised windows, shape (n_windows, win_samples).
    """
    x = resample_to(raw, fs_in, fs_out)
    x = bandpass(x, fs_out)
    x = zscore(x)
    return make_windows(x, fs_out, win_s, shift_s)
