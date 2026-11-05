"""
Synthetic degradation injection.

Applied to already band-passed, z-scored windows (see preprocess docstring).
Every degrade_* function takes a unit-variance window and returns the corrupted
window plus the exact parameter used, so the manifest can report it.

Severity is kept as a separate returned value rather than folded into the class
label. Phase 1 collapsed the three levels into one label; keeping them lets
step09 report per-severity detectability, which is what explains the published
Clean/Drifted confusion.
"""
from __future__ import annotations
import numpy as np

import config


def degrade_noise(w: np.ndarray, snr_db: float, rng: np.random.Generator):
    """Additive white Gaussian noise at a specified SNR."""
    sig_power = float(np.mean(w ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=w.shape)
    return w + noise, {"snr_db": snr_db}


def degrade_drift(w: np.ndarray, amplitude: float, fs: int,
                  rng: np.random.Generator, freq_hz=None):
    """
    Additive low-frequency sinusoidal baseline wander.

    amplitude is in multiples of the window standard deviation. The window is
    already z-scored so std is 1 and amplitude is effectively absolute, which
    is what makes severities comparable with the SNR-defined noise levels.
    """
    if freq_hz is None:
        freq_hz = config.DEGRADATION["DRIFTED"]["freq_hz"]
    f = rng.uniform(freq_hz[0], freq_hz[1])
    phase = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(w.size) / float(fs)
    drift = amplitude * np.sin(2.0 * np.pi * f * t + phase)
    return w + drift, {"amplitude": amplitude, "freq_hz": float(f)}


def degrade_attenuation(w: np.ndarray, gain: float, rng: np.random.Generator):
    """
    Multiplicative amplitude loss, the stand-in for poor skin contact or a
    failing LED/photodiode pair.

    This is the class that per-window re-normalisation destroys. Nothing here
    re-normalises; that is the whole point.
    """
    return w * gain, {"gain": gain}


def degrade_spikes(w: np.ndarray, probability: float, amp_sigma: float,
                   rng: np.random.Generator):
    """Sparse bipolar impulses at amp_sigma multiples of the window std."""
    out = w.copy()
    sigma = float(np.std(w)) or 1.0
    mask = rng.random(w.shape) < probability
    signs = rng.choice([-1.0, 1.0], size=w.shape)
    out[mask] += signs[mask] * amp_sigma * sigma
    n_spikes = int(mask.sum())
    return out, {"probability": probability, "amp_sigma": amp_sigma,
                 "n_spikes": n_spikes}


def apply_degradation(w: np.ndarray, cls: str, level: int, fs: int,
                      rng: np.random.Generator, extended: bool = False):
    """
    Dispatch. Returns (degraded_window, params_dict).

    cls must be one of config.CLASSES. CLEAN returns the window untouched.
    """
    if cls == "CLEAN":
        return w.copy(), {}

    spec = config.DEGRADATION[cls]
    levels = dict(spec["levels"])
    if extended and "extended_levels" in spec:
        levels.update(spec["extended_levels"])
    if level not in levels:
        raise ValueError(f"severity level {level} not defined for {cls}")
    value = levels[level]

    if cls == "NOISY":
        return degrade_noise(w, value, rng)
    if cls == "DRIFTED":
        return degrade_drift(w, value, fs, rng)
    if cls == "ATTENUATED":
        return degrade_attenuation(w, value, rng)
    if cls == "SPIKED":
        amp = spec["amp_sigma"][level] if level in spec["amp_sigma"] else spec["amp_sigma"][3]
        return degrade_spikes(w, value, amp, rng)
    raise ValueError(f"unknown class {cls}")


def effective_snr_db(clean: np.ndarray, degraded: np.ndarray) -> float:
    """
    Signal-to-distortion ratio of a degraded window against its own
    pre-degradation reference.

    This is the number that makes the drift story quantitative: it shows that
    severity level 1 drift sits far above the level where any feature could
    separate it from a clean window, so the Clean/Drifted confusion is a
    property of the chosen parameters and not a failure of the classifier.
    """
    err = degraded - clean
    p_sig = float(np.mean(clean ** 2))
    p_err = float(np.mean(err ** 2))
    if p_err <= 0:
        return float("inf")
    return 10.0 * np.log10(p_sig / p_err)
