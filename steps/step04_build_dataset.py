#!/usr/bin/env python3
"""
Step 04 — build the labelled corpus, for both window specifications.

For every record:
    raw -> resample -> band-pass -> z-score -> window        (clean windows)
    each clean window -> inject 4 faults x 3 severities      (degraded windows)
    every window -> 13 features + 3 published SQIs

What gets stored per window, beyond the usual X and y:

  group            record id. Makes record-level CV possible.
  severity         1..3, or 0 for clean. Phase 1 threw this away.
  source_window    index of the clean window a degraded sample came from.
                   This is what lets step05 PROVE the twin leakage rather than
                   assert it: under a window-level split, a clean window and its
                   twelve degraded derivatives scatter across train and test.
  sdr_db           signal-to-distortion ratio against the pre-degradation
                   window. Turns "drift is hard" into a number.

    python steps/step04_build_dataset.py
    python steps/step04_build_dataset.py --spec B_64hz_8s
    python steps/step04_build_dataset.py --surrogate --records 6   # smoke test
    python steps/step04_build_dataset.py --max-windows-per-record 120
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import preprocess as P
from selfdiag import degrade as D
from selfdiag import features as F
from selfdiag import sqi as Q
from selfdiag import io_bidmc, surrogate


def dataset_path(spec: str) -> str:
    return os.path.join(config.DATA, f"corpus_{spec}.npz")


def build_spec(spec_name: str, use_surrogate: bool, n_surrogate: int,
               max_windows: int | None, skip_sqi: bool) -> dict:
    spec = config.WINDOW_SPECS[spec_name]
    fs, win_s, shift_s = spec["fs"], spec["win_s"], spec["shift_s"]
    win_n = int(round(win_s * fs))

    print(f"\n--- spec {spec_name}: {fs} Hz, {win_s} s ({win_n} samples), "
          f"{shift_s} s shift, {100 * (1 - shift_s / win_s):.0f}% overlap ---")

    rng = np.random.default_rng(config.SEED)

    feats, sqis, ys, sevs, groups, srcw, sdrs = [], [], [], [], [], [], []
    group_names: list[str] = []
    sample_windows = {}          # a few raw windows kept for the figures
    t0 = time.time()
    global_window_id = 0

    if use_surrogate:
        source = surrogate.make_corpus(n_surrogate, 480.0, fs, seed=config.SEED)
        fs_in = fs
    else:
        source = io_bidmc.load_all(verbose=False)
        fs_in = config.BIDMC_FS_NATIVE

    for rid, raw in source:
        gidx = len(group_names)
        group_names.append(rid)

        clean = P.prepare_record(raw, fs_in, fs, win_s, shift_s)
        if max_windows:
            clean = clean[:max_windows]
        if clean.shape[0] == 0:
            print(f"  [skip] {rid}: too short for a {win_s}s window")
            group_names.pop()
            continue

        for w in clean:
            wid = global_window_id
            global_window_id += 1

            feats.append(F.extract(w, fs))
            sqis.append(np.zeros(3, np.float32) if skip_sqi
                        else Q.extract_sqi_batch(w[None, :], fs)[0])
            ys.append(config.CLASS_TO_IDX["CLEAN"])
            sevs.append(0)
            groups.append(gidx)
            srcw.append(wid)
            sdrs.append(np.inf)

            if "CLEAN" not in sample_windows:
                sample_windows["CLEAN"] = w.copy()

            for cls in config.CLASSES[1:]:
                for lvl in (1, 2, 3):
                    dw, _ = D.apply_degradation(w, cls, lvl, fs, rng)
                    feats.append(F.extract(dw, fs))
                    sqis.append(np.zeros(3, np.float32) if skip_sqi
                                else Q.extract_sqi_batch(dw[None, :], fs)[0])
                    ys.append(config.CLASS_TO_IDX[cls])
                    sevs.append(lvl)
                    groups.append(gidx)
                    srcw.append(wid)
                    sdrs.append(D.effective_snr_db(w, dw))

                    key = f"{cls}_{lvl}"
                    if key not in sample_windows:
                        sample_windows[key] = dw.copy()

        print(f"  {rid}: {clean.shape[0]} clean windows "
              f"-> {clean.shape[0] * 13} total   [{time.time() - t0:.0f}s]")

    if not feats:
        raise RuntimeError("no windows produced. Did step02 fetch any records?")

    X = np.vstack(feats).astype(np.float32)
    S = np.vstack(sqis).astype(np.float32)
    y = np.asarray(ys, dtype=np.int8)
    sev = np.asarray(sevs, dtype=np.int8)
    g = np.asarray(groups, dtype=np.int32)
    sw = np.asarray(srcw, dtype=np.int64)
    sdr = np.asarray(sdrs, dtype=np.float32)

    path = dataset_path(spec_name)
    np.savez_compressed(
        path, X=X, S=S, y=y, severity=sev, groups=g, source_window=sw,
        sdr_db=sdr, group_names=np.array(group_names),
        feature_names=np.array(config.FEATURE_NAMES),
        sqi_names=np.array(Q.SQI_NAMES),
        classes=np.array(config.CLASSES),
        sample_window_keys=np.array(sorted(sample_windows)),
        sample_windows=np.vstack([sample_windows[k] for k in sorted(sample_windows)]),
        fs=np.int32(fs), win_s=np.float32(win_s), shift_s=np.float32(shift_s),
    )

    counts = {config.CLASSES[i]: int((y == i).sum()) for i in range(len(config.CLASSES))}
    sdr_by_class = {}
    for i, c in enumerate(config.CLASSES[1:], start=1):
        for lvl in (1, 2, 3):
            m = (y == i) & (sev == lvl)
            if m.any():
                sdr_by_class[f"{c}_L{lvl}"] = round(float(np.median(sdr[m])), 2)

    summary = {
        "spec": spec_name, "fs": fs, "win_s": win_s, "shift_s": shift_s,
        "window_samples": win_n,
        "overlap_pct": round(100 * (1 - shift_s / win_s), 1),
        "n_records": len(group_names), "records": group_names,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "class_counts": counts,
        "n_clean_source_windows": int(global_window_id),
        "median_sdr_db_by_class_severity": sdr_by_class,
        "data_source": "SURROGATE (not reportable)" if use_surrogate else "PhysioNet BIDMC",
        "sqi_computed": not skip_sqi,
        "build_seconds": round(time.time() - t0, 1),
        "path": path,
    }
    print(f"\n  {X.shape[0]:,} samples x {X.shape[1]} features over "
          f"{len(group_names)} records   [{summary['build_seconds']}s]")
    print(f"  class counts: {counts}")
    print(f"  saved: {path}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, choices=list(config.WINDOW_SPECS))
    ap.add_argument("--surrogate", action="store_true",
                    help="synthetic source, for smoke testing only")
    ap.add_argument("--records", type=int, default=8,
                    help="number of surrogate records (--surrogate only)")
    ap.add_argument("--max-windows-per-record", type=int, default=None)
    ap.add_argument("--skip-sqi", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print("STEP 04  build labelled corpus")
    print("=" * 68)
    if args.surrogate:
        print("\n  *** SURROGATE DATA. Results are NOT reportable. ***")

    specs = [args.spec] if args.spec else list(config.WINDOW_SPECS)
    manifest = {"seed": config.SEED,
                "degradation": config.DEGRADATION,
                "filter": {"hp_hz": config.HP_CUTOFF_HZ, "lp_hz": config.LP_CUTOFF_HZ,
                           "order": config.FILTER_ORDER},
                "feature_names": config.FEATURE_NAMES,
                "specs": {}}

    for s in specs:
        manifest["specs"][s] = build_spec(
            s, args.surrogate, args.records,
            args.max_windows_per_record, args.skip_sqi)

    path = os.path.join(config.RESULTS, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\n  manifest: {path}")
    print("\nnext: python steps/step05_train_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
