#!/usr/bin/env python3
"""
Step 05 — the main experiment. Both split protocols, three classifiers.

This is the step that answers four of the seven reviewers. It runs the same
models twice:

  window_level   random stratified folds over windows, which is what Phase 1
                 did. Leaky by construction. Reported as a documented baseline.
  record_level   GroupKFold over record id. No subject on both sides. Headline.

The difference between the two is itself a result: it quantifies how much a
random split over overlapping windows inflates reported accuracy. The script
also measures the leakage directly, by counting how many test windows have a
derivative of the same clean source window sitting in the training set.

    python steps/step05_train_eval.py
    python steps/step05_train_eval.py --spec B_64hz_8s
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import splits, metrics
from selfdiag.threshold_model import ThresholdClassifier
from steps.step04_build_dataset import dataset_path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def load(spec: str):
    p = dataset_path(spec)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing. Run step04 first.")
    d = np.load(p, allow_pickle=True)
    return d


def make_models():
    return {
        "Threshold": lambda: ThresholdClassifier(),
        "SVM-RBF": lambda: SVC(**config.SVM_PARAMS),
        "RandomForest": lambda: RandomForestClassifier(**config.RF_PARAMS),
    }


def fit_predict(name, factory, Xtr, ytr, Xte, rng):
    """
    Scaler is fitted on the training fold only, every fold, no exceptions.

    SVM training is capped because an RBF kernel is O(n^2) and the full corpus
    runs to six figures. The cap is stratified and reported in the results file
    so it cannot be mistaken for a free choice.
    """
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    n_used = Xtr.shape[0]
    if name == "SVM-RBF" and Xtr.shape[0] > config.SVM_MAX_TRAIN:
        idx = rng.choice(Xtr.shape[0], config.SVM_MAX_TRAIN, replace=False)
        Xtr_s, ytr = Xtr_s[idx], ytr[idx]
        n_used = config.SVM_MAX_TRAIN

    model = factory()
    model.fit(Xtr_s, ytr)
    return model.predict(Xte_s), n_used, model, scaler


def measure_leakage(source_window, tr_idx, te_idx) -> dict:
    """
    How many held-out windows have a sibling in training?

    Two windows are siblings when they derive from the same clean source window,
    which means one is a degraded version of the other. Under a record-level
    split this is zero by construction. Under a window-level split it is close
    to total, and that is the point.
    """
    tr_src = set(source_window[tr_idx].tolist())
    te_src = source_window[te_idx]
    shared = np.isin(te_src, list(tr_src))
    return {
        "n_test": int(te_idx.size),
        "n_test_with_sibling_in_train": int(shared.sum()),
        "pct_test_with_sibling_in_train": round(100.0 * float(shared.mean()), 2),
    }


def run_protocol(protocol, X, y, groups, source_window, rng) -> dict:
    models = make_models()
    fold_scores = {m: [] for m in models}
    leak = []
    n_folds_used = 0

    if protocol == "record_level":
        folds = list(splits.record_level_folds(groups))
    elif protocol == "window_level":
        folds = list(splits.window_level_folds(y))
    else:
        raise ValueError(protocol)

    for k, (tr, te) in enumerate(folds, start=1):
        n_folds_used += 1
        leak.append(measure_leakage(source_window, tr, te))
        line = f"    fold {k:>2}/{len(folds)}  train {tr.size:>7,}  test {te.size:>7,}  "
        for name, factory in models.items():
            t0 = time.time()
            pred, n_used, _, _ = fit_predict(name, factory, X[tr], y[tr], X[te], rng)
            s = metrics.score(y[te], pred)
            s["train_samples_used"] = int(n_used)
            s["fit_seconds"] = round(time.time() - t0, 2)
            fold_scores[name].append(s)
            line += f"{name} {s['accuracy'] * 100:5.2f}%  "
        print(line)

    out = {"protocol": protocol, "n_folds": n_folds_used, "models": {}}
    for name, fs in fold_scores.items():
        agg = metrics.aggregate(fs)
        agg["per_class_f1_mean"] = [
            float(np.mean([f["per_class_f1"][i] for f in fs]))
            for i in range(len(config.CLASSES))]
        agg["per_class_recall_mean"] = [
            float(np.mean([f["per_class_recall"][i] for f in fs]))
            for i in range(len(config.CLASSES))]
        agg["mean_fit_seconds"] = float(np.mean([f["fit_seconds"] for f in fs]))
        agg["train_samples_used"] = int(fs[0]["train_samples_used"])
        out["models"][name] = agg

    out["leakage"] = {
        "mean_pct_test_with_sibling_in_train":
            round(float(np.mean([l["pct_test_with_sibling_in_train"] for l in leak])), 2),
        "per_fold": leak,
    }

    rf = [s["accuracy"] for s in fold_scores["RandomForest"]]
    sv = [s["accuracy"] for s in fold_scores["SVM-RBF"]]
    th = [s["accuracy"] for s in fold_scores["Threshold"]]
    rf_f = [s["macro_f1"] for s in fold_scores["RandomForest"]]
    sv_f = [s["macro_f1"] for s in fold_scores["SVM-RBF"]]
    out["significance"] = {
        "rf_vs_svm_accuracy": metrics.paired_tests(rf, sv, "RandomForest", "SVM-RBF"),
        "rf_vs_svm_macro_f1": metrics.paired_tests(rf_f, sv_f, "RandomForest", "SVM-RBF"),
        "rf_vs_threshold_accuracy": metrics.paired_tests(rf, th, "RandomForest", "Threshold"),
    }
    return out


def headline_holdout(X, y, groups, source_window, rng) -> dict:
    """
    One fixed record-level holdout, used for the confusion matrix and the
    feature-space figures, plus the reproduced Phase 1 70/15/15 window split so
    the paper can state both numbers side by side.
    """
    res = {}

    tr, te, test_records = splits.record_level_holdout(groups)
    pred, _, model, scaler = fit_predict(
        "RandomForest", lambda: RandomForestClassifier(**config.RF_PARAMS),
        X[tr], y[tr], X[te], rng)
    s = metrics.score(y[te], pred)
    s["confusion"] = metrics.confusion(y[te], pred)
    s["test_record_indices"] = [int(i) for i in test_records]
    s["leakage"] = measure_leakage(source_window, tr, te)
    res["record_level_holdout"] = s

    tr2, va2, te2 = splits.window_level_holdout(y)
    pred2, _, _, _ = fit_predict(
        "RandomForest", lambda: RandomForestClassifier(**config.RF_PARAMS),
        X[tr2], y[tr2], X[te2], rng)
    s2 = metrics.score(y[te2], pred2)
    s2["confusion"] = metrics.confusion(y[te2], pred2)
    s2["leakage"] = measure_leakage(source_window, tr2, te2)
    res["window_level_holdout_phase1_reproduction"] = s2

    res["inflation"] = {
        "accuracy_points": round(100.0 * (s2["accuracy"] - s["accuracy"]), 2),
        "macro_f1_points": round(100.0 * (s2["macro_f1"] - s["macro_f1"]), 2),
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, choices=list(config.WINDOW_SPECS))
    args = ap.parse_args()

    print("=" * 68)
    print("STEP 05  train and evaluate")
    print("=" * 68)

    rng = np.random.default_rng(config.SEED)
    specs = [args.spec] if args.spec else list(config.WINDOW_SPECS)
    all_results = {}

    for spec in specs:
        d = load(spec)
        X, y = d["X"], d["y"].astype(int)
        groups, src = d["groups"], d["source_window"]
        print(f"\n### spec {spec}: {X.shape[0]:,} samples, "
              f"{np.unique(groups).size} records")

        spec_res = {"n_samples": int(X.shape[0]),
                    "n_records": int(np.unique(groups).size)}

        for protocol in ("record_level", "window_level"):
            print(f"\n  protocol: {protocol}")
            spec_res[protocol] = run_protocol(protocol, X, y, groups, src, rng)
            lk = spec_res[protocol]["leakage"]["mean_pct_test_with_sibling_in_train"]
            print(f"    leakage: {lk:.1f}% of test windows have a sibling in train")
            for m, a in spec_res[protocol]["models"].items():
                print(f"      {m:<14} acc {a['accuracy']['mean'] * 100:6.2f}% "
                      f"+/- {a['accuracy']['std'] * 100:4.2f}   "
                      f"macroF1 {a['macro_f1']['mean']:.4f}")

        print("\n  fixed holdouts")
        spec_res["holdout"] = headline_holdout(X, y, groups, src, rng)
        h = spec_res["holdout"]
        print(f"    record-level : acc {h['record_level_holdout']['accuracy'] * 100:.2f}%  "
              f"macroF1 {h['record_level_holdout']['macro_f1']:.4f}")
        print(f"    window-level : acc {h['window_level_holdout_phase1_reproduction']['accuracy'] * 100:.2f}%  "
              f"macroF1 {h['window_level_holdout_phase1_reproduction']['macro_f1']:.4f}")
        print(f"    inflation    : {h['inflation']['accuracy_points']:+.2f} accuracy points")

        all_results[spec] = spec_res

    path = os.path.join(config.RESULTS, "step05_train_eval.json")
    with open(path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n  written: {path}")
    print("\nnext: python steps/step06_sqi_baselines.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
