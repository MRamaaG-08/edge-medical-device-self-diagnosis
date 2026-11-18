#!/usr/bin/env python3
"""
Step 07 — synthetic-to-real transfer, on BUT PPG.

Five of the seven reviewers said the same thing: the degradations are
artificial and the framework has never met real bad data. This step answers it
without any hardware, by taking the model trained entirely on BIDMC plus
synthetic faults and applying it, unchanged and untuned, to a different
database of real recordings that a human has labelled good or bad.

The test is binary, clean against degraded, because the BUT PPG annotation says
"usable or not", not "which of four faults". The three published SQIs are run on
the same records so the comparison is like for like.

Nothing here is fine-tuned on BUT PPG. The model never sees a BUT PPG sample
during training. That is the whole point: whatever number comes out is the
honest transfer number, and if it is poor that is a result worth reporting, not
a failure to hide.

Two confounds that must appear in the paper:
  - BUT PPG is recorded at a lower sampling rate and is resampled up here, so
    the spectral tail is not comparable to BIDMC.
  - "Degraded" in BUT PPG means a human found the record unusable, which is a
    broader category than the four synthetic faults.

    python steps/step07_butppg_transfer.py --inspect     # do this first
    python steps/step07_butppg_transfer.py
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import preprocess as P, features as F, sqi as Q, metrics, splits
from selfdiag import io_butppg
from steps.step04_build_dataset import dataset_path
from steps.step06_sqi_baselines import best_threshold, apply_threshold

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CLEAN = config.CLASS_TO_IDX["CLEAN"]


def do_inspect() -> int:
    info = io_butppg.inspect()
    print(json.dumps(info, indent=2))
    if not info.get("exists"):
        print(f"\nBUT PPG not found at {config.BUTPPG_DIR}")
        print("Download it from https://physionet.org/content/butppg/ and unpack")
        print(f"the contents into {config.BUTPPG_DIR}, then re-run --inspect.")
        return 1
    if not info.get("n_hea"):
        print("\nNo WFDB headers found. Check that the archive was unpacked, not just moved.")
        return 1
    print(f"\nFound {info['n_hea']} WFDB records. If the quality CSV header above does")
    print("not contain an id column and a quality column, report it and the loader")
    print("will be adapted.")
    return 0


def build_real_corpus(fs_target: int, win_s: float, shift_s: float, limit=None):
    """Load, resample, preprocess and window the real recordings."""
    quality = io_butppg.load_quality_table()
    rows = []
    n_skip_short = 0
    for rid, raw, fs_in in io_butppg.load_records(limit=limit):
        if rid not in quality:
            continue
        wins = P.prepare_record(raw, fs_in, fs_target, win_s, shift_s)
        if wins.shape[0] == 0:
            n_skip_short += 1
            continue
        for w in wins:
            rows.append((rid, quality[rid], F.extract(w, fs_target),
                         Q.extract_sqi_batch(w[None, :], fs_target)[0], fs_in))
    if n_skip_short:
        print(f"  {n_skip_short} records too short for a {win_s}s window, skipped")
    if not rows:
        raise RuntimeError("no BUT PPG windows matched the annotation table")

    ids = np.array([r[0] for r in rows])
    ylab = np.array([r[1] for r in rows], dtype=int)
    Xr = np.vstack([r[2] for r in rows]).astype(np.float32)
    Sr = np.vstack([r[3] for r in rows]).astype(np.float32)
    fs_seen = sorted({int(r[4]) for r in rows})
    return ids, ylab, Xr, Sr, fs_seen


def aggregate_per_record(ids, y_true, scores):
    """Annotation is per record, so window scores are averaged per record."""
    uniq = np.unique(ids)
    yt = np.empty(uniq.size, dtype=int)
    sc = np.empty(uniq.size, dtype=float)
    for i, u in enumerate(uniq):
        m = ids == u
        yt[i] = int(np.round(y_true[m].mean()))
        sc[i] = float(scores[m].mean())
    return uniq, yt, sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--spec", default=config.PRIMARY_SPEC, choices=list(config.WINDOW_SPECS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.inspect:
        return do_inspect()

    print("=" * 68)
    print("STEP 07  synthetic-to-real transfer (BUT PPG)")
    print("=" * 68)

    spec = config.WINDOW_SPECS[args.spec]
    fs, win_s, shift_s = spec["fs"], spec["win_s"], spec["shift_s"]

    p = dataset_path(args.spec)
    if not os.path.exists(p):
        print(f"ERROR: {p} missing. Run step04 first.")
        return 1
    d = np.load(p, allow_pickle=True)
    Xb, Sb, yb = d["X"], d["S"], d["y"].astype(int)
    groups_b = d["groups"]
    yb_bin = (yb != CLEAN).astype(int)

    print(f"\n  source domain : BIDMC + synthetic, {Xb.shape[0]:,} windows")
    print(f"  loading target domain from {config.BUTPPG_DIR}")
    try:
        ids, yr, Xr, Sr, fs_seen = build_real_corpus(fs, win_s, shift_s, args.limit)
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Run:  python steps/step07_butppg_transfer.py --inspect")
        return 1

    print(f"  target domain : {Xr.shape[0]:,} windows from "
          f"{np.unique(ids).size} records, native rates {fs_seen} Hz "
          f"-> resampled to {fs} Hz")
    print(f"  annotation    : {int(yr.sum()):,} degraded / "
          f"{int((1 - yr).sum()):,} clean windows")

    # ---- in-domain reference, so the transfer gap has something to be a gap from
    tr, te, _ = splits.record_level_holdout(groups_b)
    scaler = StandardScaler().fit(Xb[tr])
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(scaler.transform(Xb[tr]), yb_bin[tr])
    in_domain = metrics.binary_score(yb_bin[te], rf.predict(scaler.transform(Xb[te])))
    print(f"\n  in-domain (held-out BIDMC records)   "
          f"bal.acc {in_domain['balanced_accuracy'] * 100:.2f}%  "
          f"F1 {in_domain['macro_f1']:.4f}")

    # ---- transfer: retrain on ALL of BIDMC, apply to BUT PPG untouched
    scaler_full = StandardScaler().fit(Xb)
    rf_full = RandomForestClassifier(**config.RF_PARAMS)
    rf_full.fit(scaler_full.transform(Xb), yb_bin)

    Xr_s = scaler_full.transform(Xr)
    pred_w = rf_full.predict(Xr_s)
    proba_w = rf_full.predict_proba(Xr_s)[:, 1]

    win_level = metrics.binary_score(yr, pred_w)
    try:
        win_level["roc_auc"] = float(roc_auc_score(yr, proba_w))
    except ValueError:
        win_level["roc_auc"] = None

    uniq, yt_rec, sc_rec = aggregate_per_record(ids, yr, proba_w)
    rec_pred = (sc_rec >= 0.5).astype(int)
    rec_level = metrics.binary_score(yt_rec, rec_pred)
    try:
        rec_level["roc_auc"] = float(roc_auc_score(yt_rec, sc_rec))
    except ValueError:
        rec_level["roc_auc"] = None

    print(f"  transfer, window level               "
          f"bal.acc {win_level['balanced_accuracy'] * 100:.2f}%  "
          f"F1 {win_level['macro_f1']:.4f}  "
          f"AUC {win_level['roc_auc'] if win_level['roc_auc'] else float('nan'):.3f}")
    print(f"  transfer, record level               "
          f"bal.acc {rec_level['balanced_accuracy'] * 100:.2f}%  "
          f"F1 {rec_level['macro_f1']:.4f}  "
          f"AUC {rec_level['roc_auc'] if rec_level['roc_auc'] else float('nan'):.3f}")

    gap = in_domain["balanced_accuracy"] - win_level["balanced_accuracy"]
    print(f"\n  TRANSFER GAP (balanced accuracy): {gap * 100:+.2f} points")

    # ---- the SQIs, fitted on BIDMC and applied to BUT PPG the same way
    print("\n  published SQIs under the same protocol")
    sqi_res = {}
    for si, name in enumerate(Q.SQI_NAMES):
        v_src = np.nan_to_num(Sb[:, si].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        v_tgt = np.nan_to_num(Sr[:, si].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        thr, direction, src_ba = best_threshold(v_src, yb_bin)
        pred = apply_threshold(v_tgt, thr, direction)
        sc = metrics.binary_score(yr, pred)
        try:
            signed = v_tgt if direction == "greater" else -v_tgt
            sc["roc_auc"] = float(roc_auc_score(yr, signed))
        except ValueError:
            sc["roc_auc"] = None
        sc["threshold_fitted_on_bidmc"] = thr
        sc["direction"] = direction
        sc["bidmc_balanced_accuracy"] = src_ba
        sqi_res[name] = sc
        print(f"    {name:<8} BIDMC {src_ba * 100:5.2f}%  ->  "
              f"BUT PPG {sc['balanced_accuracy'] * 100:5.2f}%  "
              f"(gap {(src_ba - sc['balanced_accuracy']) * 100:+.2f})")

    out = {
        "spec": args.spec,
        "target_native_sampling_rates_hz": fs_seen,
        "resampled_to_hz": fs,
        "n_target_windows": int(Xr.shape[0]),
        "n_target_records": int(np.unique(ids).size),
        "target_class_balance": {"degraded": int(yr.sum()), "clean": int((1 - yr).sum())},
        "in_domain_reference": in_domain,
        "transfer_window_level": win_level,
        "transfer_record_level": rec_level,
        "transfer_gap_balanced_accuracy": float(gap),
        "sqi_transfer": sqi_res,
        "caveats": [
            "Model trained only on BIDMC plus synthetic degradation. No BUT PPG "
            "sample was used for training, threshold fitting or model selection.",
            "BUT PPG native sampling rate is lower than BIDMC and records were "
            "resampled up, which changes the spectral tail. The measured gap is "
            "therefore an upper bound on the true synthetic-to-real gap.",
            "The BUT PPG label is a human usability judgement, a broader category "
            "than the four synthetic fault types, so only binary evaluation is valid.",
        ],
    }
    path = os.path.join(config.RESULTS, "step07_butppg_transfer.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written: {path}")
    print("\nnext: python steps/step08_edge_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
