#!/usr/bin/env python3
"""
Step 06 — published SQI baselines.

Reviewers 3 and 8: "include appropriate signal-quality assessment baselines and
simple threshold-based methods, in addition to SVM and Random Forest. This
would help establish whether the proposed feature-based framework offers a
meaningful improvement."

Three published single-number SQIs are evaluated on exactly the same
record-level folds as the learned models. Because an SQI produces a scalar and
not a five-way label, the comparison is binary: clean against degraded. The
Random Forest is collapsed to the same binary decision so the numbers are
comparable.

The decision threshold for each SQI is swept on the TRAINING folds only and
applied unchanged to the held-out records, which is the same discipline the
learned models get.

    python steps/step06_sqi_baselines.py
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import splits, metrics, sqi
from steps.step04_build_dataset import dataset_path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

CLEAN = config.CLASS_TO_IDX["CLEAN"]


def best_threshold(values, y_bin, n_grid: int = 200):
    """
    Sweep a scalar threshold for best balanced accuracy on training data.

    Both polarities are tried because the SQIs do not agree on direction:
    S_SQI and T_SQI are higher-is-cleaner, and a raw skewness can go either way
    depending on which fault dominates.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, "greater", 0.0
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return float(np.median(finite)), "greater", 0.0

    best = (float(lo), "greater", -1.0)
    for thr in np.linspace(lo, hi, n_grid):
        for direction in ("greater", "less"):
            pred = (values > thr) if direction == "greater" else (values < thr)
            ba = balanced_accuracy_score(y_bin, pred.astype(int))
            if ba > best[2]:
                best = (float(thr), direction, float(ba))
    return best


def apply_threshold(values, thr, direction):
    pred = (values > thr) if direction == "greater" else (values < thr)
    return pred.astype(int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, choices=list(config.WINDOW_SPECS))
    args = ap.parse_args()

    print("=" * 68)
    print("STEP 06  published SQI baselines (binary clean vs degraded)")
    print("=" * 68)

    specs = [args.spec] if args.spec else list(config.WINDOW_SPECS)
    out_all = {}

    for spec in specs:
        p = dataset_path(spec)
        if not os.path.exists(p):
            print(f"  [skip] {spec}: {p} missing")
            continue
        d = np.load(p, allow_pickle=True)
        X, S, y = d["X"], d["S"], d["y"].astype(int)
        groups = d["groups"]
        y_bin = (y != CLEAN).astype(int)

        if not bool(np.any(S)):
            print(f"  [skip] {spec}: SQIs are all zero. "
                  "Rebuild step04 without --skip-sqi.")
            continue

        print(f"\n### spec {spec}: {X.shape[0]:,} samples, "
              f"{int(y_bin.sum()):,} degraded / {int((1 - y_bin).sum()):,} clean")

        folds = list(splits.record_level_folds(groups))
        per_method = {n: [] for n in sqi.SQI_NAMES}
        per_method["RandomForest-13feat"] = []
        aucs = {n: [] for n in sqi.SQI_NAMES}
        chosen = {n: [] for n in sqi.SQI_NAMES}

        for k, (tr, te) in enumerate(folds, start=1):
            line = f"    fold {k:>2}/{len(folds)}  "

            for si, name in enumerate(sqi.SQI_NAMES):
                v_tr, v_te = S[tr, si].astype(np.float64), S[te, si].astype(np.float64)
                v_tr = np.nan_to_num(v_tr, nan=0.0, posinf=0.0, neginf=0.0)
                v_te = np.nan_to_num(v_te, nan=0.0, posinf=0.0, neginf=0.0)
                thr, direction, _ = best_threshold(v_tr, y_bin[tr])
                pred = apply_threshold(v_te, thr, direction)
                sc = metrics.binary_score(y_bin[te], pred)
                per_method[name].append(sc)
                chosen[name].append({"threshold": thr, "direction": direction})
                try:
                    signed = v_te if direction == "greater" else -v_te
                    aucs[name].append(float(roc_auc_score(y_bin[te], signed)))
                except ValueError:
                    pass
                line += f"{name} {sc['balanced_accuracy'] * 100:5.2f}%  "

            scaler = StandardScaler().fit(X[tr])
            rf = RandomForestClassifier(**config.RF_PARAMS)
            rf.fit(scaler.transform(X[tr]), y_bin[tr])
            pred = rf.predict(scaler.transform(X[te]))
            sc = metrics.binary_score(y_bin[te], pred)
            per_method["RandomForest-13feat"].append(sc)
            line += f"RF {sc['balanced_accuracy'] * 100:5.2f}%"
            print(line)

        res = {}
        for name, fs in per_method.items():
            agg = metrics.aggregate(
                fs, keys=("accuracy", "balanced_accuracy", "macro_f1"))
            agg["recall_degraded_mean"] = float(np.mean([f["recall_degraded"] for f in fs]))
            agg["precision_degraded_mean"] = float(np.mean([f["precision_degraded"] for f in fs]))
            if name in aucs and aucs[name]:
                agg["roc_auc_mean"] = float(np.mean(aucs[name]))
                agg["roc_auc_std"] = float(np.std(aucs[name], ddof=1)) if len(aucs[name]) > 1 else 0.0
            if name in chosen:
                agg["thresholds_per_fold"] = chosen[name]
            res[name] = agg

        rf_ba = [f["balanced_accuracy"] for f in per_method["RandomForest-13feat"]]
        res["significance_vs_rf"] = {}
        for name in sqi.SQI_NAMES:
            ba = [f["balanced_accuracy"] for f in per_method[name]]
            res["significance_vs_rf"][name] = metrics.paired_tests(
                rf_ba, ba, "RandomForest-13feat", name)

        print(f"\n    {'method':<22}{'bal.acc':>10}{'macroF1':>10}{'AUC':>8}")
        print("    " + "-" * 50)
        for name in list(sqi.SQI_NAMES) + ["RandomForest-13feat"]:
            a = res[name]
            auc = a.get("roc_auc_mean")
            print(f"    {name:<22}"
                  f"{a['balanced_accuracy']['mean'] * 100:>9.2f}%"
                  f"{a['macro_f1']['mean']:>10.4f}"
                  f"{(f'{auc:.3f}' if auc else '-'):>8}")

        out_all[spec] = res

    path = os.path.join(config.RESULTS, "step06_sqi_baselines.json")
    with open(path, "w") as fh:
        json.dump(out_all, fh, indent=2)
    print(f"\n  written: {path}")
    print("\nnext: python steps/step07_butppg_transfer.py --inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
