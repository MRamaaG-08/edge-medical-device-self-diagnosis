"""Reference heart rate for BIDMC records.

BIDMC ships a second, small "numerics" record per patient (bidmc01n, bidmc02n, ...)
holding the bedside monitor's heart rate once per second, measured from the ECG.
That is our ground truth: independent of the PPG we are judging.

Practice (surrogate) records have a known, fixed heart rate, so it is recreated.
"""
import os

import numpy as np

import config
from phase2.contracts import FS


def _surrogate_hr(rid, n_records):
    rng = np.random.default_rng(config.SEED)
    for i in range(n_records):
        hr = float(rng.uniform(55, 95))
        rng.uniform(10, 20); rng.integers(0, 2 ** 31)
        if f"surrogate{i + 1:02d}" == rid:
            return hr
    raise KeyError(rid)


def reference_hr(rid, n_records=53, duration_s=480):
    """Return (hr_per_second array, source string)."""
    if str(rid).startswith("surrogate"):
        return np.full(int(duration_s), _surrogate_hr(rid, n_records)), "surrogate_known"
    cache = os.path.join(config.BIDMC_DIR, f"{rid}_hr.npy")
    if os.path.exists(cache):
        return np.load(cache), "bidmc_numerics_ecg"
    import wfdb
    local = os.path.join(config.BIDMC_DIR, f"{rid}n")
    rec = wfdb.rdrecord(local) if os.path.exists(local + ".hea") else wfdb.rdrecord(f"{rid}n", pn_dir="bidmc/1.0.0")
    names = [s.strip().upper() for s in rec.sig_name]
    hr = np.asarray(rec.p_signal[:, names.index("HR")], dtype=float)
    good = np.isfinite(hr) & (hr > 20)
    hr = np.interp(np.arange(hr.size), np.where(good)[0], hr[good])
    np.save(cache, hr)
    return hr, "bidmc_numerics_ecg"


def window_reference(hr_per_s, start_sample, win=500, fs=FS):
    """Median monitor heart rate over the seconds a window covers."""
    a = int(start_sample / fs)
    b = max(a + 1, int(np.ceil((start_sample + win) / fs)))
    seg = hr_per_s[a:b]
    return float(np.median(seg)) if seg.size else float("nan")
