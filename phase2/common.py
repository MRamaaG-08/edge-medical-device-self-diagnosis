"""Small helpers shared by all Phase 2 code. Owned by Ramaa."""
import os

import numpy as np

import config
from phase2.contracts import FS, HR_BAND_HZ, SPEC


def corpus_path(spec=SPEC):
    """The Phase 1 corpus built by steps/step04_build_dataset.py."""
    return os.path.join(config.DATA, f"corpus_{spec}.npz")


def load_corpus(spec=SPEC):
    p = corpus_path(spec)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Build it first:  python steps/step04_build_dataset.py --spec {spec}")
    return np.load(p, allow_pickle=True)


def hr_estimate(w, fs=FS):
    """Heart rate (bpm) of one window: the strongest frequency between 42 and 210 bpm.

    Zero-padded to 4096 points so the answer is not stuck to 12 bpm steps.
    """
    w = np.asarray(w, dtype=float)
    w = (w - w.mean()) * np.hanning(w.size)
    spec = np.abs(np.fft.rfft(w, n=4096)) ** 2
    f = np.fft.rfftfreq(4096, d=1.0 / fs)
    band = (f >= HR_BAND_HZ[0]) & (f <= HR_BAND_HZ[1])
    return float(60.0 * f[band][np.argmax(spec[band])])


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_raw_record(rid, n_surrogate=None):
    """Raw PPG of one Phase 1 record, plus its sampling rate.

    Real BIDMC records come from the step02 cache (data/bidmc/*.npy, 125 Hz).
    Practice records called surrogateNN are regenerated exactly as step04 made them.
    """
    if str(rid).startswith("surrogate"):
        from selfdiag import surrogate
        fs = FS
        n = n_surrogate or int(str(rid)[9:])
        for r, x in surrogate.make_corpus(max(n, int(str(rid)[9:])), 480.0, fs, seed=config.SEED):
            if r == rid:
                return x, fs
        raise KeyError(rid)
    from selfdiag import io_bidmc
    return io_bidmc.load_record(str(rid)), config.BIDMC_FS_NATIVE
