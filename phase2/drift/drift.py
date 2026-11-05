"""The drift monitor: notices a sensor going slowly wrong, minutes before a single
window looks bad.

How it works, in plain words. During the 2-minute calibration it learns what each
telemetry number normally looks like (its average and its usual wobble). After that,
each new value is turned into "how many usual-wobbles away from normal" (a z value).
A cumulative sum (Page's CUSUM test, from the Page-Hinkley family) adds those up,
minus a small allowance. Random wobble cancels out; a steady push in one direction
keeps growing, and when it passes the threshold the alarm fires.

It is a dozen lines of arithmetic per stream, so it runs on the Pi and could run on
the ESP32 as well.
"""
import numpy as np

STREAMS = ["dc_level", "perfusion_index", "signal_amplitude"]
BASELINE_WINDOWS = 24     # 24 windows x 2.5 s = 60 s of normal behaviour, right after calibration


class Cusum:
    """Two-sided Page CUSUM on z values. allowance in z units, threshold in z units."""

    def __init__(self, threshold=20.0, allowance=0.5):
        self.threshold = threshold
        self.allowance = allowance
        self.up = 0.0
        self.down = 0.0

    def update(self, z):
        self.up = max(0.0, self.up + z - self.allowance)
        self.down = max(0.0, self.down - z - self.allowance)
        return self.up > self.threshold or self.down > self.threshold


class DriftMonitor:
    def __init__(self, streams=STREAMS, threshold=20.0, allowance=0.5,
                 baseline=BASELINE_WINDOWS):
        self.streams = list(streams)
        self.threshold = threshold
        self.allowance = allowance
        self.baseline = baseline
        self.reset()

    def reset(self):
        self.n = 0
        self.buf = {s: [] for s in self.streams}
        self.mu, self.sd = {}, {}
        self.tests = {s: Cusum(self.threshold, self.allowance) for s in self.streams}

    @property
    def ready(self):
        return self.n > self.baseline

    def update(self, values):
        """values: dict with one number per stream. Returns the streams in alarm (list)."""
        self.n += 1
        if self.n <= self.baseline:
            for s in self.streams:
                self.buf[s].append(float(values[s]))
            if self.n == self.baseline:
                for s in self.streams:
                    a = np.asarray(self.buf[s])
                    self.mu[s] = float(a.mean())
                    # floor the wobble at 1% of the level so a perfectly steady
                    # baseline does not make the test over-sensitive
                    self.sd[s] = float(max(a.std(), 0.01 * abs(a.mean()), 1e-9))
            return []
        fired = []
        for s in self.streams:
            z = (float(values[s]) - self.mu[s]) / self.sd[s]
            if self.tests[s].update(float(np.clip(z, -10, 10))):
                fired.append(s)
        return fired
