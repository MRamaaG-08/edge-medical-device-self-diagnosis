"""
Physiologically shaped synthetic PPG, used only to self-test the pipeline when
the real database is not reachable.

NOT A DATA SOURCE. It is cleaner and far more regular than real PPG and will
inflate every accuracy figure. Nothing generated here may appear in the paper.
step04 stamps results/manifest.json with the data source so a surrogate run can
never be mistaken for a real one.
"""
from __future__ import annotations
import numpy as np


def _pulse(t_rel: np.ndarray) -> np.ndarray:
    """One cardiac cycle on t_rel in [0, 1): systolic peak plus dicrotic notch."""
    systolic = np.exp(-((t_rel - 0.22) ** 2) / (2 * 0.055 ** 2))
    dicrotic = 0.38 * np.exp(-((t_rel - 0.48) ** 2) / (2 * 0.075 ** 2))
    return systolic + dicrotic


def make_surrogate(duration_s: float, fs: int, hr_bpm: float = 72.0,
                   hrv_sd: float = 0.035, resp_rate_bpm: float = 15.0,
                   seed: int = 0) -> np.ndarray:
    """Generate a PPG-like trace with heart-rate variability and respiratory modulation."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / float(fs)

    beats, pos = [], 0.0
    while pos < duration_s + 2.0:
        rr = 60.0 / hr_bpm * (1.0 + rng.normal(0.0, hrv_sd))
        rr = float(np.clip(rr, 0.35, 2.0))
        beats.append(pos)
        pos += rr

    x = np.zeros(n, dtype=np.float64)
    for i in range(len(beats) - 1):
        a, b = beats[i], beats[i + 1]
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        x[m] = _pulse((t[m] - a) / (b - a))

    resp = 1.0 + 0.12 * np.sin(2 * np.pi * (resp_rate_bpm / 60.0) * t)
    x *= resp
    x += 0.012 * rng.standard_normal(n)       # sensor noise floor
    x += 0.05 * np.sin(2 * np.pi * 0.02 * t)  # slow baseline
    return x


def make_corpus(n_records: int, duration_s: float, fs: int, seed: int = 0):
    """Yield (record_id, raw) pairs shaped like io_bidmc.load_all."""
    rng = np.random.default_rng(seed)
    for i in range(n_records):
        hr = float(rng.uniform(55, 95))
        rr = float(rng.uniform(10, 20))
        yield (f"surrogate{i + 1:02d}",
               make_surrogate(duration_s, fs, hr_bpm=hr,
                              resp_rate_bpm=rr, seed=int(rng.integers(0, 2 ** 31))))
