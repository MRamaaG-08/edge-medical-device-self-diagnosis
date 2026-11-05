#!/usr/bin/env python3
"""
Step 09 — ablation, feature importance, and the drift detectability analysis.

Three parts.

A. ABLATION. Full 13 features against time-only (8), frequency-only (5), and
   leave-one-feature-out for all 13. Phase 1 ran ablation for Random Forest
   only and disclosed that as a limitation; this runs it for SVM too, which
   closes the limitation rather than restating it.

B. IMPORTANCE. Gini and permutation importance side by side. Gini is biased
   toward continuous high-cardinality features, so a ranking that only ever
   appears in Gini is weak evidence. Permutation importance on held-out records
   is the honest version, and disagreement between the two is worth reporting.

C. DRIFT DETECTABILITY. The quantitative explanation for the Clean/Drifted
   confusion Phase 1 disclosed but could not account for. Drift severity 1 has
   amplitude 0.02 on a z-scored unit-variance window, which puts it around
   34 dB signal-to-distortion ratio. Nothing in a 13-feature set can separate
   that from natural beat-to-beat variation. The analysis reports per-severity
   recall against measured SDR, and optionally adds an out-of-range severity
   level to show the detector works once drift is actually present.

    python steps/step09_ablation_importance.py
    python steps/step09_ablation_importance.py --extended-drift
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import splits, metrics, preprocess as P, degrade as D, features as F
from steps.step04_build_dataset import dataset_path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

TIME_FEATURES = config.FEATURE_NAMES[:8]
FREQ_FEATURES = config.FEATURE_NAMES[8:]


def _eval_subset(X, y, groups, cols, model_name, rng, n_splits=5) -> dict:
    accs, f1s = [], []
    for tr, te in splits.record_level_folds(groups, n_splits=n_splits):
        Xtr, Xte = X[np.ix_(tr, cols)], X[np.ix_(te, cols)]
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        ytr = y[tr]
        if model_name == "SVM-RBF":
            if Xtr.shape[0] > config.SVM_MAX_TRAIN:
                idx = rng.choice(Xtr.shape[0], config.SVM_MAX_TRAIN, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]
            model = SVC(**config.SVM_PARAMS)
        else:
            model = RandomForestClassifier(**config.RF_PARAMS)
        model.fit(Xtr, ytr)
        s = metrics.score(y[te], model.predict(Xte))
        accs.append(s["accuracy"])
        f1s.append(s["macro_f1"])
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "n_features": len(cols),
    }


def ablation(X, y, groups, rng, models=("RandomForest", "SVM-RBF")) -> dict:
    names = config.FEATURE_NAMES
    all_cols = list(range(len(names)))
    time_cols = [names.index(n) for n in TIME_FEATURES]
    freq_cols = [names.index(n) for n in FREQ_FEATURES]

    out = {}
    for m in models:
        print(f"\n    {m}")
        res = {"full_13": _eval_subset(X, y, groups, all_cols, m, rng)}
        print(f"      full 13        acc {res['full_13']['accuracy_mean'] * 100:5.2f}%")
        res["time_only_8"] = _eval_subset(X, y, groups, time_cols, m, rng)
        print(f"      time only  8   acc {res['time_only_8']['accuracy_mean'] * 100:5.2f}%")
        res["freq_only_5"] = _eval_subset(X, y, groups, freq_cols, m, rng)
        print(f"      freq only  5   acc {res['freq_only_5']['accuracy_mean'] * 100:5.2f}%")

        loo = {}
        base = res["full_13"]["accuracy_mean"]
        for i, n in enumerate(names):
            cols = [c for c in all_cols if c != i]
            r = _eval_subset(X, y, groups, cols, m, rng)
            r["delta_vs_full"] = float(r["accuracy_mean"] - base)
            loo[n] = r
        res["leave_one_out"] = loo
        worst = sorted(loo.items(), key=lambda kv: kv[1]["delta_vs_full"])[:3]
        print("      most costly to remove: " +
              ", ".join(f"{k} ({v['delta_vs_full'] * 100:+.2f})" for k, v in worst))
        out[m] = res
    return out


def importance(X, y, groups, rng) -> dict:
    names = config.FEATURE_NAMES
    gini, perm = [], []
    for tr, te in splits.record_level_folds(groups, n_splits=min(5, config.N_FOLDS)):
        scaler = StandardScaler().fit(X[tr])
        rf = RandomForestClassifier(**config.RF_PARAMS)
        rf.fit(scaler.transform(X[tr]), y[tr])
        gini.append(rf.feature_importances_)
        sub = min(5000, te.size)
        idx = rng.choice(te, sub, replace=False)
        pi = permutation_importance(
            rf, scaler.transform(X[idx]), y[idx],
            n_repeats=5, random_state=config.SEED, n_jobs=-1)
        perm.append(pi.importances_mean)

    G, Pm = np.vstack(gini), np.vstack(perm)
    out = {}
    for i, n in enumerate(names):
        out[n] = {
            "gini_mean": float(G[:, i].mean()),
            "gini_std": float(G[:, i].std(ddof=1)) if G.shape[0] > 1 else 0.0,
            "permutation_mean": float(Pm[:, i].mean()),
            "permutation_std": float(Pm[:, i].std(ddof=1)) if Pm.shape[0] > 1 else 0.0,
        }
    g_rank = [names[i] for i in np.argsort(-G.mean(axis=0))]
    p_rank = [names[i] for i in np.argsort(-Pm.mean(axis=0))]
    out["_gini_ranking"] = g_rank
    out["_permutation_ranking"] = p_rank
    out["_rankings_agree_top4"] = bool(set(g_rank[:4]) == set(p_rank[:4]))

    print(f"      gini top 4:        {g_rank[:4]}")
    print(f"      permutation top 4: {p_rank[:4]}")
    print(f"      top-4 sets agree:  {out['_rankings_agree_top4']}")
    return out


def drift_analysis(spec, X, y, sev, sdr, groups, rng) -> dict:
    """Per-severity recall against measured signal-to-distortion ratio."""
    tr, te, _ = splits.record_level_holdout(groups)
    scaler = StandardScaler().fit(X[tr])
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(scaler.transform(X[tr]), y[tr])
    pred = rf.predict(scaler.transform(X[te]))

    rows = {}
    for ci, cname in enumerate(config.CLASSES):
        if cname == "CLEAN":
            m = (y[te] == ci)
            if m.any():
                rows["CLEAN"] = {
                    "severity": 0, "n": int(m.sum()),
                    "recall": float((pred[m] == ci).mean()),
                    "median_sdr_db": None,
                }
            continue
        for lvl in (1, 2, 3):
            m = (y[te] == ci) & (sev[te] == lvl)
            if not m.any():
                continue
            finite = sdr[te][m]
            finite = finite[np.isfinite(finite)]
            rows[f"{cname}_L{lvl}"] = {
                "severity": lvl,
                "n": int(m.sum()),
                "recall": float((pred[m] == ci).mean()),
                "median_sdr_db": round(float(np.median(finite)), 2) if finite.size else None,
            }

    print(f"\n      {'class/severity':<20}{'n':>7}{'recall':>9}{'SDR dB':>9}")
    print("      " + "-" * 45)
    for k, v in rows.items():
        sdr_s = f"{v['median_sdr_db']:.1f}" if v["median_sdr_db"] is not None else "-"
        print(f"      {k:<20}{v['n']:>7}{v['recall'] * 100:>8.1f}%{sdr_s:>9}")
    return rows


def extended_drift(spec, rng, n_records: int = 8) -> dict:
    """
    Out-of-range drift check. Adds severity level 4 (amplitude 0.30) and tests
    whether the Clean/Drifted boundary is a detector failure or a consequence of
    the chosen severity range. Rebuilt from source because the main corpus only
    stores levels 1 to 3.
    """
    s = config.WINDOW_SPECS[spec]
    fs, win_s, shift_s = s["fs"], s["win_s"], s["shift_s"]

    manifest_path = os.path.join(config.RESULTS, "manifest.json")
    source_is_surrogate = True
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            man = json.load(fh)
        src = man.get("specs", {}).get(spec, {}).get("data_source", "")
        source_is_surrogate = "SURROGATE" in src

    if source_is_surrogate:
        from selfdiag import surrogate
        records = list(surrogate.make_corpus(n_records, 240.0, fs, seed=config.SEED))
        fs_in = fs
    else:
        from selfdiag import io_bidmc
        records = list(io_bidmc.load_all(
            config.BIDMC_RECORDS[:n_records], allow_download=False, verbose=False))
        fs_in = config.BIDMC_FS_NATIVE
        if not records:
            return {"skipped": "no cached BIDMC records"}

    feats, labels, levels, sdrs, grps = [], [], [], [], []
    for gi, (rid, raw) in enumerate(records):
        wins = P.prepare_record(raw, fs_in, fs, win_s, shift_s)[:150]
        for w in wins:
            feats.append(F.extract(w, fs)); labels.append(0); levels.append(0)
            sdrs.append(np.inf); grps.append(gi)
            for lvl in (1, 2, 3, 4):
                dw, _ = D.apply_degradation(w, "DRIFTED", lvl, fs, rng, extended=True)
                feats.append(F.extract(dw, fs)); labels.append(1); levels.append(lvl)
                sdrs.append(D.effective_snr_db(w, dw)); grps.append(gi)

    Xe = np.vstack(feats).astype(np.float32)
    ye = np.asarray(labels); lv = np.asarray(levels)
    sd = np.asarray(sdrs); ge = np.asarray(grps)

    # train on levels 1-3 only; level 4 is held out entirely as an unseen severity
    train_mask = lv <= 3
    tr, te, _ = splits.record_level_holdout(ge)
    tr = tr[train_mask[tr]]

    scaler = StandardScaler().fit(Xe[tr])
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(scaler.transform(Xe[tr]), ye[tr])
    pred = rf.predict(scaler.transform(Xe[te]))

    out = {"n_records": len(records), "trained_on_levels": [1, 2, 3], "by_level": {}}
    for lvl in (0, 1, 2, 3, 4):
        m = (lv[te] == lvl)
        if not m.any():
            continue
        finite = sd[te][m]; finite = finite[np.isfinite(finite)]
        key = "CLEAN" if lvl == 0 else f"DRIFT_L{lvl}"
        amp = (config.DEGRADATION["DRIFTED"]["levels"].get(lvl)
               or config.DEGRADATION["DRIFTED"]["extended_levels"].get(lvl))
        out["by_level"][key] = {
            "n": int(m.sum()),
            "amplitude": amp,
            "detected_as_drifted_rate": float((pred[m] == 1).mean()),
            "median_sdr_db": round(float(np.median(finite)), 2) if finite.size else None,
            "seen_in_training": bool(lvl <= 3),
        }
        print(f"      {key:<12} amp={str(amp):<6} "
              f"detected {out['by_level'][key]['detected_as_drifted_rate'] * 100:5.1f}%   "
              f"SDR {out['by_level'][key]['median_sdr_db']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=config.PRIMARY_SPEC, choices=list(config.WINDOW_SPECS))
    ap.add_argument("--extended-drift", action="store_true")
    ap.add_argument("--skip-svm-ablation", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print("STEP 09  ablation, importance, drift detectability")
    print("=" * 68)

    p = dataset_path(args.spec)
    if not os.path.exists(p):
        print(f"ERROR: {p} missing. Run step04 first.")
        return 1
    d = np.load(p, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    sev, sdr, groups = d["severity"].astype(int), d["sdr_db"], d["groups"]
    rng = np.random.default_rng(config.SEED)

    print("\n  A. feature ablation")
    models = ("RandomForest",) if args.skip_svm_ablation else ("RandomForest", "SVM-RBF")
    abl = ablation(X, y, groups, rng, models=models)

    print("\n  B. feature importance")
    imp = importance(X, y, groups, rng)

    print("\n  C. per-severity detectability")
    drift = drift_analysis(args.spec, X, y, sev, sdr, groups, rng)

    ext = None
    if args.extended_drift:
        print("\n  D. extended drift severity (out-of-range check)")
        ext = extended_drift(args.spec, rng)

    out = {"spec": args.spec, "ablation": abl, "importance": imp,
           "per_severity": drift, "extended_drift": ext}
    path = os.path.join(config.RESULTS, "step09_ablation_importance.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written: {path}")
    print("\nnext: python steps/step10_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
