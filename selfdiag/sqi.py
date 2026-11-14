"""
Published signal-quality indices, used as baselines.

Reviewers 3 and 8 both asked whether a thirteen-feature Random Forest actually
improves on established simple methods. These are the established simple
methods. Each returns a scalar per window; step06 fits a decision threshold on
the training folds only and evaluates on the held-out records, so the baselines
get the same protocol as the learned models.

  S_SQI   skewness             Elgendi, "Optimal Signal Quality Index for
                               Photoplethysmogram Signals", Bioengineering,
                               3(4):21, 2016. Reported there as the best
                               single-number PPG SQI.
  R_SQI   relative power       Same paper. Ratio of power in the pulsatile band
                               to power in the broad band.
  T_SQI   template matching    Orphanidou et al., "Signal-Quality Indices for
                               the Electrocardiogram and Photoplethysmogram",
                               IEEE J. Biomed. Health Inform., 19(3), 2015.
                               Beat detection, averaged template, mean
                               correlation, plus physiological plausibility.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import skew
from scipy.signal import find_peaks, resample

_EPS = 1e-12

# Orphanidou plausibility bounds
HR_MIN_BPM = 40.0
HR_MAX_BPM = 180.0
MAX_RR_RATIO = 2.2          # longest to shortest beat interval
TEMPLATE_LEN = 64           # samples each beat is resampled to before averaging


def s_sqi(w: np.ndarray) -> float:
    """
    Skewness SQI. A clean PPG pulse train is positively skewed; additive noise
    pulls it toward zero and large bipolar spikes inflate the magnitude.
    Returned as the raw signed skewness.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size < 3:
        return 0.0
    return float(skew(w, bias=False))


def r_sqi(w: np.ndarray, fs: int,
          pulse_band=(1.0, 2.25), broad_band=(0.0, 8.0)) -> float:
    """
    Relative power SQI: band power in the pulsatile range over broad-band power.
    Elgendi's bands are used unchanged. Higher is cleaner.
    """
    w = np.asarray(w, dtype=np.float64)
    n = w.size
    win = np.hanning(n)
    psd = (np.abs(np.fft.rfft(w * win)) ** 2) / (np.sum(win ** 2) * fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    pulse = np.sum(psd[(freqs >= pulse_band[0]) & (freqs <= pulse_band[1])])
    broad = np.sum(psd[(freqs >= broad_band[0]) &
                       (freqs <= min(broad_band[1], freqs[-1]))]) + _EPS
    return float(pulse / broad)


def _detect_beats(w: np.ndarray, fs: int) -> np.ndarray:
    """Systolic peak indices, with a refractory period set by HR_MAX_BPM."""
    min_dist = max(1, int(fs * 60.0 / HR_MAX_BPM))
    prom = 0.3 * float(np.std(w))
    peaks, _ = find_peaks(w, distance=min_dist, prominence=max(prom, _EPS))
    return peaks


def t_sqi(w: np.ndarray, fs: int, return_detail: bool = False):
    """
    Template-matching SQI.

    Beats are detected, each beat segment is resampled to a common length, the
    average beat becomes the template, and the SQI is the mean correlation of
    individual beats against it. Windows failing Orphanidou's physiological
    plausibility rules score 0, which is the behaviour described in the paper:
    an implausible window is rejected outright rather than scored.
    """
    w = np.asarray(w, dtype=np.float64)
    detail = {"n_beats": 0, "hr_bpm": float("nan"), "reason": ""}

    peaks = _detect_beats(w, fs)
    detail["n_beats"] = int(peaks.size)

    if peaks.size < 3:
        detail["reason"] = "fewer than 3 beats"
        return (0.0, detail) if return_detail else 0.0

    rr = np.diff(peaks) / float(fs)
    hr = 60.0 / np.mean(rr)
    detail["hr_bpm"] = float(hr)

    if not (HR_MIN_BPM <= hr <= HR_MAX_BPM):
        detail["reason"] = "implausible heart rate"
        return (0.0, detail) if return_detail else 0.0

    if np.min(rr) <= 0 or (np.max(rr) / np.min(rr)) > MAX_RR_RATIO:
        detail["reason"] = "beat interval ratio out of range"
        return (0.0, detail) if return_detail else 0.0

    # build beats as peak-centred segments of one mean RR
    half = int(0.5 * np.mean(rr) * fs)
    beats = []
    for p in peaks:
        a, b = p - half, p + half
        if a < 0 or b > w.size:
            continue
        seg = w[a:b]
        if seg.size < 4:
            continue
        beats.append(resample(seg, TEMPLATE_LEN))

    if len(beats) < 3:
        detail["reason"] = "too few complete beats"
        return (0.0, detail) if return_detail else 0.0

    B = np.vstack(beats)
    template = B.mean(axis=0)
    if np.std(template) < _EPS:
        detail["reason"] = "degenerate template"
        return (0.0, detail) if return_detail else 0.0

    corrs = []
    for beat in B:
        if np.std(beat) < _EPS:
            corrs.append(0.0)
            continue
        corrs.append(float(np.corrcoef(beat, template)[0, 1]))

    val = float(np.nanmean(corrs))
    if not np.isfinite(val):
        val = 0.0
    return (val, detail) if return_detail else val


SQI_NAMES = ["S_SQI", "R_SQI", "T_SQI"]


def extract_sqi_batch(windows: np.ndarray, fs: int) -> np.ndarray:
    """(n, win) -> (n, 3) in SQI_NAMES order."""
    windows = np.asarray(windows, dtype=np.float64)
    out = np.empty((windows.shape[0], len(SQI_NAMES)), dtype=np.float32)
    for i in range(windows.shape[0]):
        w = windows[i]
        out[i, 0] = s_sqi(w)
        out[i, 1] = r_sqi(w, fs)
        out[i, 2] = t_sqi(w, fs)
    return out
