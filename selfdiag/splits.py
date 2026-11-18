"""
Split protocols.

The whole point of the rebuild. Two protocols are implemented so the paper can
report both and quantify the gap between them.

window_level  Windows are shuffled and split at random, ignoring which record
              they came from. This is what Phase 1 did. It leaks three ways:
                1. adjacent windows overlap (50% at spec A, 75% at spec B), so
                   near-duplicate windows land on both sides of the split
                2. every record appears in both train and test, so the model
                   can memorise subject-specific morphology
                3. the degraded variants are derived from the same clean
                   windows, so a clean window in train has its noisy, drifted,
                   attenuated and spiked twins in test
              Reported as a documented baseline, never as the headline result.

record_level  GroupKFold over record id. No record contributes windows to both
              sides, so none of the three leak paths exist. This is what four
              of the seven reviewers asked for and it is the headline protocol.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

import config


def record_level_folds(groups: np.ndarray, n_splits: int = config.N_FOLDS):
    """
    Yield (train_idx, test_idx) with no record on both sides.

    n_splits is capped at the number of distinct records so the call cannot
    fail silently on a small subset run.
    """
    groups = np.asarray(groups)
    n_groups = np.unique(groups).size
    k = int(min(n_splits, n_groups))
    if k < 2:
        raise ValueError(
            f"record-level CV needs at least 2 records, found {n_groups}. "
            "Fetch more BIDMC records before running the headline protocol."
        )
    gkf = GroupKFold(n_splits=k)
    dummy = np.zeros(groups.shape[0])
    for tr, te in gkf.split(dummy, dummy, groups=groups):
        yield tr, te


def window_level_folds(y: np.ndarray, n_splits: int = config.N_FOLDS,
                       seed: int = config.SEED):
    """Stratified folds over windows, ignoring record id. Deliberately leaky."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        yield tr, te


def window_level_holdout(y: np.ndarray, seed: int = config.SEED):
    """
    The exact 70/15/15 stratified split Phase 1 used, reproduced so the paper
    can state the inflated number it produced alongside the clean one.
    """
    tr_frac, va_frac, te_frac = config.WINDOW_LEVEL_SPLIT
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=tr_frac, stratify=y,
                                random_state=seed)
    rel = va_frac / (va_frac + te_frac)
    va, te = train_test_split(rest, train_size=rel, stratify=y[rest],
                              random_state=seed)
    return tr, va, te


def record_level_holdout(groups: np.ndarray, seed: int = config.SEED,
                         test_frac: float = 0.30):
    """
    Single record-wise holdout, for the figures that need one fixed split
    (confusion matrix, feature-space scatter) rather than a fold average.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(uniq)
    n_test = max(1, int(round(test_frac * uniq.size)))
    test_records = set(shuffled[:n_test].tolist())
    mask = np.array([g in test_records for g in groups])
    return np.where(~mask)[0], np.where(mask)[0], sorted(test_records)
