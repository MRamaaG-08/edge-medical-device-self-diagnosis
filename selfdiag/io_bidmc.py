"""
BIDMC PPG and Respiration Database loader.

Pimentel et al., "Toward a Robust Estimation of Respiratory Rate from Pulse
Oximeters", IEEE Trans. Biomed. Eng., 64(8), 2017. Distributed on PhysioNet as
bidmc/1.0.0. 53 records, 8 minutes each, 125 Hz.

Phase 1 used five records. This uses all 53, which is what makes record-level
cross-validation statistically meaningful rather than a gesture.

Reads a local copy if one exists, otherwise streams from PhysioNet via wfdb.
Streaming 53 records takes a few minutes; step02 caches them as .npy so it only
happens once.
"""
from __future__ import annotations
import os
import numpy as np

import config

PN_DIR = "bidmc/1.0.0"


def _find_ppg_channel(sig_name) -> int:
    lowered = [s.strip().lower() for s in sig_name]
    for cand in config.BIDMC_PPG_CHANNEL_CANDIDATES:
        c = cand.strip().lower()
        if c in lowered:
            return lowered.index(c)
    for i, s in enumerate(lowered):
        if "pleth" in s or "ppg" in s:
            return i
    raise KeyError(f"no PPG channel among {sig_name}")


def cache_path(record: str) -> str:
    return os.path.join(config.BIDMC_DIR, f"{record}.npy")


def load_record(record: str, allow_download: bool = True) -> np.ndarray:
    """
    Return the raw PPG channel of one record at its native 125 Hz.

    NaNs are linearly interpolated; BIDMC has short gaps in some records and
    leaving them in propagates through the filter and poisons the whole record.
    """
    cp = cache_path(record)
    if os.path.exists(cp):
        return np.load(cp)

    if not allow_download:
        raise FileNotFoundError(f"{cp} missing and downloads disabled")

    import wfdb
    local_hea = os.path.join(config.BIDMC_DIR, f"{record}.hea")
    if os.path.exists(local_hea):
        rec = wfdb.rdrecord(os.path.join(config.BIDMC_DIR, record))
    else:
        rec = wfdb.rdrecord(record, pn_dir=PN_DIR)

    ch = _find_ppg_channel(rec.sig_name)
    x = np.asarray(rec.p_signal[:, ch], dtype=np.float64)

    if np.isnan(x).any():
        idx = np.arange(x.size)
        good = ~np.isnan(x)
        if good.sum() < 2:
            raise ValueError(f"{record}: PPG channel is almost entirely NaN")
        x = np.interp(idx, idx[good], x[good])

    np.save(cp, x)
    return x


def load_all(records=None, allow_download: bool = True, verbose: bool = True):
    """
    Yield (record_id, raw_ppg) for every record that loads.

    Failures are reported and skipped rather than aborting the run; a missing
    record is a smaller problem than losing the other 52.
    """
    records = records or config.BIDMC_RECORDS
    ok, failed = 0, []
    for r in records:
        try:
            x = load_record(r, allow_download=allow_download)
        except Exception as e:
            failed.append((r, str(e)))
            if verbose:
                print(f"  [skip] {r}: {e}")
            continue
        ok += 1
        if verbose:
            print(f"  [ok]   {r}: {x.size} samples "
                  f"({x.size / config.BIDMC_FS_NATIVE / 60:.1f} min)")
        yield r, x
    if verbose:
        print(f"\n  loaded {ok} records, {len(failed)} failed")
