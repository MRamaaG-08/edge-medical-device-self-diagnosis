"""V1. Does the live (streaming) front end give the same answers as Phase 1?

Phase 1 cleaned each recording all at once. The ESP32 must clean it sample by
sample. This script takes held-out BIDMC records, makes windows BOTH ways, injects
the same faults into both, and asks the Phase 1 model to classify them.

    python -m phase2.drift.v1_frontend_check
    python -m phase2.drift.v1_frontend_check --records 6      # quicker

Writes results/phase2/v1_frontend_check.json
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config
from phase2.common import ensure_dir, load_corpus, load_raw_record
from phase2.contracts import FS, SHIFT, WIN
from phase2.frontend import StreamingFrontEnd
from selfdiag import degrade as D
from selfdiag import features as F
from selfdiag import preprocess as P
from selfdiag import splits

ap = argparse.ArgumentParser()
ap.add_argument("--records", type=int, default=10, help="how many held-out records to test")
a = ap.parse_args()

d = load_corpus()
X, y, groups, names = d["X"], d["y"], d["groups"], d["group_names"]
tr, te, test_ids = splits.record_level_holdout(groups)
test_names = [str(names[g]) for g in test_ids][: a.records]
print(f"training the Phase 1 model on {len(np.unique(groups[tr]))} records ...")
rf = RandomForestClassifier(**config.RF_PARAMS).fit(X[tr], y[tr])


def classify(windows, rng):
    """Inject every fault at every level (exactly like step04) and classify."""
    feats, labels = [], []
    for w in windows:
        feats.append(F.extract(w, FS)); labels.append(0)
        for ci, cls in enumerate(config.CLASSES[1:], start=1):
            for lvl in (1, 2, 3):
                dw, _ = D.apply_degradation(w, cls, lvl, FS, rng)
                feats.append(F.extract(dw, FS)); labels.append(ci)
    feats, labels = np.asarray(feats), np.asarray(labels)
    return feats, labels, rf.predict(feats)


res = {"records": test_names, "per_record": []}
acc = {"offline": [], "stream_single": [], "stream_double": []}
fdiff = {"stream_single": [], "stream_double": []}
for rid in test_names:
    raw, fs_in = load_raw_record(rid, n_surrogate=len(names))
    off = P.prepare_record(raw, fs_in, FS, config.WINDOW_SPECS["A_100hz_5s"]["win_s"],
                           config.WINDOW_SPECS["A_100hz_5s"]["shift_s"])
    x100 = P.resample_to(raw, fs_in, FS)
    row = {"record": rid}
    streams = {}
    for name, double in (("stream_single", False), ("stream_double", True)):
        fe = StreamingFrontEnd(double=double)
        streams[name] = np.asarray([w for w, _ in fe.push(x100)])
    first = (fe.n_warm + fe.n_cal) // SHIFT          # offline window that lines up with stream window 0
    n = min(len(off) - first, *[len(s) for s in streams.values()])
    if n <= 0:
        print(f"  [skip] {rid}: too short"); continue
    sets = {"offline": off[first:first + n], **{k: v[:n] for k, v in streams.items()}}
    clean_feats = {}
    for name, wins in sets.items():
        f, lab, pred = classify(wins, np.random.default_rng(config.SEED))
        row[name] = float((pred == lab).mean())
        acc[name].append(row[name])
        clean_feats[name] = f[lab == 0]
    for name in fdiff:
        diff = np.abs(clean_feats[name] - clean_feats["offline"]).mean(0)
        scale = np.abs(clean_feats["offline"]).mean(0) + 1e-9
        fdiff[name].append(diff / scale)
    res["per_record"].append(row)
    print(f"  {rid}: {n} windows   offline {row['offline']:.3f}   "
          f"single {row['stream_single']:.3f}   double {row['stream_double']:.3f}")

res["accuracy_mean"] = {k: float(np.mean(v)) for k, v in acc.items()}
res["accuracy_drop_points"] = {k: round(100 * (res["accuracy_mean"]["offline"] - res["accuracy_mean"][k]), 2)
                               for k in ("stream_single", "stream_double")}
res["relative_feature_change"] = {k: dict(zip(config.FEATURE_NAMES, np.mean(v, 0).round(4).tolist()))
                                  for k, v in fdiff.items()}
best = min(res["accuracy_drop_points"], key=res["accuracy_drop_points"].get)
res["recommended_frontend"] = best
print("\naccuracy (same faults, same model):")
for k, v in res["accuracy_mean"].items():
    print(f"   {k:<14} {100 * v:6.2f}%")
print(f"drop vs Phase 1: single {res['accuracy_drop_points']['stream_single']} points, "
      f"double {res['accuracy_drop_points']['stream_double']} points")
print(f"recommended front end for the ESP32: {best}")
out = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "v1_frontend_check.json")
json.dump(res, open(out, "w"), indent=2)
print(f"saved {out}")
