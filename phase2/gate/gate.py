"""The certified release gate (Learn-then-Test with fixed-sequence testing).

Plain words: we have a trust score for every window, and we know (on data with a
reference heart rate) whether that window would have shown a WRONG heart rate.
We look for the most generous threshold for which we can still say, with 95%
confidence, that at most 5% of the heart rates we show are wrong.
"""
import numpy as np
from scipy.stats import binom


def ltt_threshold(scores, wrong, alpha=0.05, delta=0.05, n_grid=200):
    """Return the chosen threshold, or None if no threshold can be certified.

    scores : higher = more trustworthy (shape n)
    wrong  : 1 if the shown heart rate would have been misleading (shape n)
    """
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=int)
    # candidate thresholds from strictest (show almost nothing) to loosest (show everything)
    grid = np.unique(np.quantile(scores, np.linspace(1.0, 0.0, n_grid)))[::-1]
    # With fewer than n_min shown windows no threshold could ever be certified
    # (even zero mistakes is not enough evidence), so the test sequence starts
    # at the first threshold that shows at least n_min windows.
    n_min = int(np.ceil(np.log(delta) / np.log(1.0 - alpha)))
    chosen = None
    for th in grid:
        shown = scores >= th
        n = int(shown.sum())
        if n < n_min:
            continue
        k = int((wrong & shown).sum())
        # p-value for "true wrong-rate among shown > alpha"; small p = safe
        p = binom.cdf(k, n, alpha)
        if p <= delta:
            chosen = float(th)      # certified; try a looser one next
        else:
            break                   # fixed-sequence: stop at the first failure
    return chosen


def evaluate(scores, wrong, threshold):
    """Coverage (share of readings shown) and wrong-rate among the shown readings."""
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=int)
    if threshold is None:
        return {"coverage": 0.0, "wrong_among_shown": 0.0, "n_shown": 0}
    shown = scores >= threshold
    n = int(shown.sum())
    return {"coverage": float(shown.mean()),
            "wrong_among_shown": float((wrong & shown).sum() / n) if n else 0.0,
            "n_shown": n}


def risk_coverage_curve(scores, wrong, n_points=100):
    """Points for the risk-coverage plot: (coverage, wrong-rate among shown)."""
    scores = np.asarray(scores, dtype=float)
    wrong = np.asarray(wrong, dtype=int)
    out = []
    for th in np.quantile(scores, np.linspace(0.0, 0.99, n_points)):
        e = evaluate(scores, wrong, th)
        out.append((e["coverage"], e["wrong_among_shown"]))
    return np.array(out)
