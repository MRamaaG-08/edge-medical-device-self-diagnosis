"""Scoring, aggregation across folds, and paired significance tests."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, balanced_accuracy_score)
from scipy.stats import ttest_rel, wilcoxon

import config


def score(y_true, y_pred) -> dict:
    """Headline metrics for one fold."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(config.CLASSES)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_precision": [float(v) for v in p],
        "per_class_recall": [float(v) for v in r],
        "per_class_f1": [float(v) for v in f],
        "per_class_support": [int(v) for v in s],
    }


def binary_score(y_true_bin, y_pred_bin) -> dict:
    """Clean vs degraded. Used for the SQI baselines and the BUT PPG transfer."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
        "macro_f1": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "precision_degraded": float(p[1]),
        "recall_degraded": float(r[1]),
        "f1_degraded": float(f[1]),
    }


def aggregate(fold_scores: list[dict], keys=("accuracy", "balanced_accuracy", "macro_f1")) -> dict:
    """Mean and sample std across folds."""
    out = {}
    for k in keys:
        vals = np.array([fs[k] for fs in fold_scores], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_folds": int(vals.size),
            "values": [float(v) for v in vals],
        }
    return out


def cohens_d_paired(a, b) -> float:
    """Effect size for paired samples. Mean difference over std of differences."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") if d.mean() != 0 else 0.0
    return float(d.mean() / sd)


def paired_tests(a, b, label_a="A", label_b="B") -> dict:
    """
    Paired t-test and Wilcoxon signed-rank across folds, with effect size.

    At 10 record-level folds Wilcoxon has enough resolution to reach p < 0.01,
    which the 5-fold version in the original paper did not (it bottomed out at
    p = 0.0625, the smallest value attainable at n = 5).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = {"label_a": label_a, "label_b": label_b,
           "n": int(a.size),
           "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float((a - b).mean()),
           "cohens_d": cohens_d_paired(a, b)}
    try:
        t, p = ttest_rel(a, b)
        out["t_stat"], out["t_p"] = float(t), float(p)
    except Exception as e:
        out["t_stat"], out["t_p"], out["t_error"] = None, None, str(e)
    try:
        if np.allclose(a, b):
            out["wilcoxon_stat"], out["wilcoxon_p"] = None, 1.0
            out["wilcoxon_note"] = "identical samples"
        else:
            w, p = wilcoxon(a, b)
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(w), float(p)
            out["wilcoxon_min_attainable_p"] = float(2.0 ** (-a.size + 1))
    except Exception as e:
        out["wilcoxon_stat"], out["wilcoxon_p"], out["wilcoxon_error"] = None, None, str(e)
    return out


def confusion(y_true, y_pred) -> list:
    return confusion_matrix(
        y_true, y_pred, labels=range(len(config.CLASSES))).tolist()
