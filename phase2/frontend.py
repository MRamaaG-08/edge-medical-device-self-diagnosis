"""The streaming front end: exactly what the ESP32 does to every sample.

Phase 1 cleaned a whole 8-minute record at once (zero-phase filtering, then one
z-score over the record). A live device cannot look into the future, so here
every sample is filtered the moment it arrives (a "causal" filter), and the
scale is fixed once from a 2-minute clean baseline (Calibration). Bhumika ports
this file to C line by line (phase2/edge/c/edge_frontend.c), Vandana checks it
gives Phase 1 answers (phase2/drift/v1_frontend_check.py), and Ramaa's hub uses
it. It lives in the shared base, so nobody edits it without a group decision.

    raw sample -> band-pass (causal, 0.5 to 4 Hz) -> calibration -> 5 s windows
"""
import numpy as np
from scipy.signal import butter, sosfilt

import config
from phase2.contracts import CALIB_S, FS, SHIFT, WARMUP_S, WIN
from selfdiag.preprocess import Calibration


def filter_sections(fs=FS, double=True):
    """The same Butterworth filters Phase 1 uses (order 2 high-pass + order 2 low-pass).

    Phase 1 ran them forward AND backward (sosfiltfilt), which squares the
    magnitude response. double=True runs each filter twice forward, which gives
    the same magnitude response without needing the future. Only the timing
    (phase) differs. phase2/drift/v1_frontend_check.py measures what that costs.
    """
    nyq = fs / 2.0
    hp = butter(config.FILTER_ORDER, config.HP_CUTOFF_HZ / nyq, btype="highpass", output="sos")
    lp = butter(config.FILTER_ORDER, min(config.LP_CUTOFF_HZ / nyq, 0.99), btype="lowpass", output="sos")
    sos = np.vstack([hp, lp])
    return np.vstack([sos, sos]) if double else sos


class StreamingFrontEnd:
    """Feed raw samples in any chunk size; get finished windows out.

    Each finished window comes with its telemetry, measured on the RAW light level
    before any normalisation (this is what lets slow sensor problems show up):
        dc_level         average raw light level over the window
        perfusion_index  100 x (pulse size / dc level), the usual pulse-oximeter measure
        signal_amplitude spread of the calibrated window (about 1.0 when healthy)
    """

    def __init__(self, fs=FS, double=True, calib_s=CALIB_S, warmup_s=WARMUP_S):
        self.fs = fs
        self.sos = filter_sections(fs, double)
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.n_warm = int(warmup_s * fs)
        self.n_cal = int(calib_s * fs)
        self.cal = Calibration()
        self.calibrated = False
        self.seen = 0
        self.cal_buf = []
        self.filt = []          # filtered (not yet calibrated) samples, post calibration
        self.raw = []           # matching raw samples
        self.since_last = 0

    def push(self, raw_chunk):
        raw_chunk = np.atleast_1d(np.asarray(raw_chunk, dtype=float))
        y, self.zi = sosfilt(self.sos, raw_chunk, zi=self.zi)
        out = []
        for r, v in zip(raw_chunk, y):
            self.seen += 1
            if self.seen <= self.n_warm:
                continue                                   # filter still settling
            if not self.calibrated:
                self.cal_buf.append(v)
                if len(self.cal_buf) == self.n_cal:
                    self.cal.fit(np.asarray(self.cal_buf))
                    self.calibrated = True
                    self.cal_buf = []
                continue
            self.filt.append(v)
            self.raw.append(r)
            self.since_last += 1
            if len(self.filt) >= WIN and self.since_last >= SHIFT:
                self.since_last = 0
                f = np.asarray(self.filt[-WIN:])
                rw = np.asarray(self.raw[-WIN:])
                w = self.cal.apply(f)
                dc = float(rw.mean())
                tele = {"dc_level": dc,
                        "perfusion_index": float(100.0 * f.std() / (abs(dc) + 1e-9)),
                        "signal_amplitude": float(w.std())}
                out.append((w, tele))
                self.filt = self.filt[-WIN:]
                self.raw = self.raw[-WIN:]
        return out

    @property
    def calibration_samples(self):
        """How many samples go by before the first window: warm-up + calibration + one window."""
        return self.n_warm + self.n_cal + WIN


def device_like(raw, dc_counts=50000.0, ac_fraction=0.01):
    """Turn a dataset PPG trace into what a MAX30102 would report.

    A real optical sensor reports a large steady light level (tens of thousands of
    counts) with a pulse riding on top that is only about 1% of it. Datasets like
    BIDMC store the pulse alone, so the steady part is put back here. This only
    matters for the telemetry (dc_level, perfusion_index); the windows are the same.
    """
    raw = np.asarray(raw, dtype=float)
    z = (raw - raw.mean()) / (raw.std() + 1e-12)
    return dc_counts * (1.0 + ac_fraction * z)
