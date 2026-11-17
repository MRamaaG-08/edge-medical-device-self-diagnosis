"""
Transparent rule-based classifier, the non-learned baseline.

Phase 1 reported a "threshold rules" baseline at 33.40% accuracy without
specifying the rules, which reviewer 3 objected to. This is an explicit
implementation: one discriminative feature per fault class, with the decision
boundary placed midway between the clean median and that class's median on the
TRAINING data only.

Rules, in priority order:
    SPIKED       kurtosis above threshold        impulses fatten the tails
    ATTENUATED   peak-to-peak below threshold    amplitude loss
    NOISY        spectral entropy above thr.     energy spread across the band
    DRIFTED      power ratio below threshold     energy leaves the pulsatile band
    CLEAN        otherwise
"""
from __future__ import annotations
import numpy as np

import config

RULES = [
    ("SPIKED", "kurtosis", "greater"),
    ("ATTENUATED", "peak_to_peak", "less"),
    ("NOISY", "spectral_entropy", "greater"),
    ("DRIFTED", "power_ratio", "less"),
]


class ThresholdClassifier:
    """sklearn-shaped so it drops into the same evaluation loop."""

    def __init__(self) -> None:
        self.thresholds_: dict[str, float] = {}
        self.feature_idx_ = {n: i for i, n in enumerate(config.FEATURE_NAMES)}

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        clean = X[y == config.CLASS_TO_IDX["CLEAN"]]
        for cls, feat, _direction in RULES:
            fi = self.feature_idx_[feat]
            target = X[y == config.CLASS_TO_IDX[cls]]
            if clean.size == 0 or target.size == 0:
                self.thresholds_[cls] = float("inf")
                continue
            self.thresholds_[cls] = float(
                0.5 * (np.median(clean[:, fi]) + np.median(target[:, fi])))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], config.CLASS_TO_IDX["CLEAN"], dtype=np.int64)
        assigned = np.zeros(X.shape[0], dtype=bool)
        for cls, feat, direction in RULES:
            fi = self.feature_idx_[feat]
            thr = self.thresholds_.get(cls, float("inf"))
            hit = (X[:, fi] > thr) if direction == "greater" else (X[:, fi] < thr)
            take = hit & ~assigned
            out[take] = config.CLASS_TO_IDX[cls]
            assigned |= take
        return out

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kw):
        return self
