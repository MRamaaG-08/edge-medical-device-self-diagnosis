#!/usr/bin/env python3
"""
Step 11 — calibration, selective prediction, and severity grading.

Three analyses that need no new data and no new downloads. Everything here runs
on the corpus step04 already built, using the same record-level folds as step05,
so it adds minutes to the day rather than hours.

A. CALIBRATION. Phase 1 disclosed as a limitation that "the influence of the
   imbalance on classifier calibration specifically was not separately
   quantified". This quantifies it: expected calibration error, a reliability
   curve, and a comparison of balanced against unweighted class weights. That
   closes a self-disclosed limitation rather than restating it, which is worth
   more than most new experiments.

B. SELECTIVE PREDICTION. A device that must answer every window is a different
   product from one that can say "I am not sure about this reading". Sweeping a
   confidence threshold gives a risk-coverage curve: how much accuracy you buy
   by declining to classify the least confident windows. This is the honest
   version of a deployment claim, and it answers reviewer 8's question about
   what the predicted label would actually make the device do.

C. SEVERITY GRADING. Phase 1 folded three severity levels into one class label
   and threw the level away. step04 keeps it, so the model can be asked how bad
   the degradation is, not just which kind. Reported with the caveat it deserves:
   severity is discrete by construction here, so this measures whether the
   feature set orders severity correctly, not whether it would track a
   continuously worsening real sensor.

    python steps/step11_calibration_abstention.py
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import splits, metrics
from steps.step04_build_dataset import dataset_path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

N_BINS = 10
COVERAGE_GRID = np.arange(1.00, 0.44, -0.05)


# --------------------------------------------------------------- A
def expected_calibration_error(conf, correct, n_bins=N_BINS):
    """
    Standard binned ECE, plus the per-bin table the reliability curve needs.

    Bins are equal-width over [0, 1]. Empty bins contribute nothing and are
    reported with n = 0 rather than silently dropped, because a model that
    never emits low confidence is itself a finding.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    n = conf.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "mean_confidence": None, "accuracy": None})
            continue
        acc = float(correct[m].mean())
        mc = float(conf[m].mean())
        ece += (cnt / n) * abs(acc - mc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": cnt,
                     "mean_confidence": mc, "accuracy": acc})
    return float(ece), bins


def calibration(X, y, groups, rng) -> dict:
    out = {}
    for label, weight in (("balanced", "balanced"), ("unweighted", None)):
        confs, corrects = [], []
        for tr, te in splits.record_level_folds(groups, n_splits=min(5, config.N_FOLDS)):
            scaler = StandardScaler().fit(X[tr])
            params = dict(config.RF_PARAMS)
            params["class_weight"] = weight
            rf = RandomForestClassifier(**params)
            rf.fit(scaler.transform(X[tr]), y[tr])
            proba = rf.predict_proba(scaler.transform(X[te]))
            pred = rf.classes_[np.argmax(proba, axis=1)]
            confs.append(proba.max(axis=1))
            corrects.append((pred == y[te]).astype(float))

        conf = np.concatenate(confs)
        corr = np.concatenate(corrects)
        ece, bins = expected_calibration_error(conf, corr)
        out[label] = {
            "expected_calibration_error": round(ece, 4),
            "mean_confidence": round(float(conf.mean()), 4),
            "mean_accuracy": round(float(corr.mean()), 4),
            "overconfidence": round(float(conf.mean() - corr.mean()), 4),
            "reliability_bins": bins,
        }
        print(f"      {label:<12} ECE {ece:.4f}   "
              f"mean conf {conf.mean():.4f}   mean acc {corr.mean():.4f}   "
              f"gap {conf.mean() - corr.mean():+.4f}")

    out["imbalance_effect_on_calibration"] = {
        "ece_balanced": out["balanced"]["expected_calibration_error"],
        "ece_unweighted": out["unweighted"]["expected_calibration_error"],
        "delta": round(out["balanced"]["expected_calibration_error"]
                       - out["unweighted"]["expected_calibration_error"], 4),
        "note": ("Positive delta means balanced class weighting made calibration "
                 "worse, which is the usual trade for better minority-class "
                 "recall. This is the quantity Phase 1 listed as not separately "
                 "quantified."),
    }
    return out


# --------------------------------------------------------------- B
def selective_prediction(X, y, groups, rng) -> dict:
    """Risk-coverage curve from the max predicted probability."""
    confs, corrects, preds, trues = [], [], [], []
    for tr, te in splits.record_level_folds(groups, n_splits=min(5, config.N_FOLDS)):
        scaler = StandardScaler().fit(X[tr])
        rf = RandomForestClassifier(**config.RF_PARAMS)
        rf.fit(scaler.transform(X[tr]), y[tr])
        proba = rf.predict_proba(scaler.transform(X[te]))
        pred = rf.classes_[np.argmax(proba, axis=1)]
        confs.append(proba.max(axis=1))
        corrects.append((pred == y[te]).astype(float))
        preds.append(pred)
        trues.append(y[te])

    conf = np.concatenate(confs)
    corr = np.concatenate(corrects)
    pred = np.concatenate(preds)
    true = np.concatenate(trues)

    order = np.argsort(-conf)          # most confident first
    rows = []
    base_acc = float(corr.mean())
    for cov in COVERAGE_GRID:
        k = max(1, int(round(cov * conf.size)))
        sel = order[:k]
        acc = float(corr[sel].mean())
        rows.append({
            "coverage": round(float(cov), 3),
            "n_retained": int(k),
            "accuracy": round(acc, 4),
            "macro_f1": round(float(metrics.score(true[sel], pred[sel])["macro_f1"]), 4),
            "error_rate": round(1.0 - acc, 4),
            "confidence_threshold": round(float(conf[sel].min()), 4),
            "accuracy_gain_vs_full": round(acc - base_acc, 4),
        })
        print(f"      coverage {cov * 100:5.1f}%   acc {acc * 100:6.2f}%   "
              f"err {100 * (1 - acc):5.2f}%   thr {conf[sel].min():.3f}")

    targets = {}
    for target in (0.90, 0.95, 0.99):
        hit = [r for r in rows if r["accuracy"] >= target]
        targets[f"coverage_at_{int(target * 100)}pct_accuracy"] = (
            max(r["coverage"] for r in hit) if hit else None)

    return {"full_coverage_accuracy": round(base_acc, 4),
            "curve": rows,
            "targets": targets,
            "note": ("Confidence is the maximum class probability of the Random "
                     "Forest. No conformal or distribution-free guarantee is "
                     "claimed; this is an empirical risk-coverage curve on "
                     "held-out records.")}


# --------------------------------------------------------------- C
def severity_grading(X, y, sev, groups, rng) -> dict:
    """
    Can the feature set order severity within a fault class?

    Trained and evaluated per fault class on record-level folds, so the task is
    "given that this window is Noisy, how bad is it" rather than a joint
    class-and-severity problem.
    """
    out = {}
    for ci, cname in enumerate(config.CLASSES):
        if cname == "CLEAN":
            continue
        m = (y == ci)
        if m.sum() < 100:
            continue
        Xc, sc, gc = X[m], sev[m], groups[m]
        maes, accs = [], []
        for tr, te in splits.record_level_folds(gc, n_splits=min(5, config.N_FOLDS)):
            scaler = StandardScaler().fit(Xc[tr])
            rf = RandomForestClassifier(**config.RF_PARAMS)
            rf.fit(scaler.transform(Xc[tr]), sc[tr])
            p = rf.predict(scaler.transform(Xc[te]))
            maes.append(float(np.mean(np.abs(p - sc[te]))))
            accs.append(float(accuracy_score(sc[te], p)))
        out[cname] = {
            "mae_levels": round(float(np.mean(maes)), 4),
            "mae_std": round(float(np.std(maes, ddof=1)), 4) if len(maes) > 1 else 0.0,
            "exact_level_accuracy": round(float(np.mean(accs)), 4),
            "n_samples": int(m.sum()),
        }
        print(f"      {cname:<12} MAE {out[cname]['mae_levels']:.3f} levels   "
              f"exact {out[cname]['exact_level_accuracy'] * 100:5.1f}%")

    out["_caveat"] = (
        "Severity is discrete and generated by construction, so these figures "
        "measure whether the feature set orders the injected levels correctly. "
        "They are not evidence that the model would track a continuously "
        "degrading physical sensor, which requires Phase 2 hardware data.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=config.PRIMARY_SPEC, choices=list(config.WINDOW_SPECS))
    args = ap.parse_args()

    print("=" * 68)
    print("STEP 11  calibration, selective prediction, severity grading")
    print("=" * 68)

    p = dataset_path(args.spec)
    if not os.path.exists(p):
        print(f"ERROR: {p} missing. Run step04 first.")
        return 1
    d = np.load(p, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    sev, groups = d["severity"].astype(int), d["groups"]
    rng = np.random.default_rng(config.SEED)

    print("\n  A. probability calibration")
    cal = calibration(X, y, groups, rng)

    print("\n  B. selective prediction (risk-coverage)")
    sel = selective_prediction(X, y, groups, rng)
    for k, v in sel["targets"].items():
        print(f"      {k}: {v}")

    print("\n  C. severity grading within fault class")
    grade = severity_grading(X, y, sev, groups, rng)

    out = {"spec": args.spec, "calibration": cal,
           "selective_prediction": sel, "severity_grading": grade}
    path = os.path.join(config.RESULTS, "step11_calibration_abstention.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written: {path}")
    print("\nnext: python steps/step10_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
