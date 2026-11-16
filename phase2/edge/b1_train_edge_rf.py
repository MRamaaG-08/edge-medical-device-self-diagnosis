"""B1. Pick the smallest forest that is still as good as the Phase 1 model.

The Phase 1 forest (50 trees, depth 10) is about 385 KB as a C table. The ESP32
can spare roughly 150 KB for it. Each candidate below is scored the fair way
(record-level, 5 folds) and its size is counted in nodes x 12 bytes plus the leaf
tables, which is what the C file will really hold.

Note: Phase 1 put a StandardScaler in front of the forest. A forest only compares
each feature with a threshold, so scaling changes nothing about its decisions; the
edge model is trained on raw feature values so the C code needs no scaler at all.

    python -m phase2.edge.b1_train_edge_rf

Writes models/edge_rf.joblib and results/phase2/b1_edge_rf.json
"""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config
from phase2.common import ensure_dir, load_corpus
from phase2.contracts import EDGE_MODEL
from selfdiag import splits

CANDIDATES = [(50, 10), (30, 8), (20, 10), (20, 8), (10, 10), (10, 8)]
TOLERANCE = 0.005           # accept up to half an accuracy point below the full model
FLASH_BUDGET_KB = 150


def model_kb(rf):
    nodes = sum(t.tree_.node_count for t in rf.estimators_)
    leaves = sum(int((t.tree_.children_left == -1).sum()) for t in rf.estimators_)
    return nodes, (nodes * config.BYTES_PER_NODE + leaves * len(config.CLASSES) * 4) / 1024.0


d = load_corpus()
X, y, g = d["X"], d["y"], d["groups"]
folds = list(splits.record_level_folds(g, n_splits=5))
rows = []
for n_trees, depth in CANDIDATES:
    accs, kbs, nodes = [], [], []
    for tr, te in folds:
        p = dict(config.RF_PARAMS, n_estimators=n_trees, max_depth=depth)
        rf = RandomForestClassifier(**p).fit(X[tr], y[tr])
        accs.append(float((rf.predict(X[te]) == y[te]).mean()))
        n, kb = model_kb(rf); nodes.append(n); kbs.append(kb)
    rows.append({"trees": n_trees, "depth": depth, "accuracy": round(float(np.mean(accs)), 4),
                 "accuracy_std": round(float(np.std(accs, ddof=1)), 4),
                 "nodes": int(np.mean(nodes)), "kb": round(float(np.mean(kbs)), 1)})
    print(f"  {n_trees:>2} trees, depth {depth:>2}: accuracy {100 * rows[-1]['accuracy']:5.2f}%   "
          f"{rows[-1]['nodes']:>6} nodes   {rows[-1]['kb']:6.1f} KB")

full = rows[0]["accuracy"]
ok = [r for r in rows if r["kb"] <= FLASH_BUDGET_KB and r["accuracy"] >= full - TOLERANCE]
pick = min(ok, key=lambda r: r["kb"]) if ok else max((r for r in rows if r["kb"] <= FLASH_BUDGET_KB),
                                                     key=lambda r: r["accuracy"])
print(f"\nchosen: {pick['trees']} trees, depth {pick['depth']} "
      f"({pick['kb']} KB, {100 * (full - pick['accuracy']):.2f} points below the full model)")
final = RandomForestClassifier(**dict(config.RF_PARAMS, n_estimators=pick["trees"],
                                      max_depth=pick["depth"])).fit(X, y)
ensure_dir("models")
joblib.dump(final, EDGE_MODEL)
out = {"candidates": rows, "chosen": pick, "full_model_accuracy": full,
       "rule": f"smallest model within {TOLERANCE * 100:.1f} points of the full model and under {FLASH_BUDGET_KB} KB"}
p = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "b1_edge_rf.json")
json.dump(out, open(p, "w"), indent=2)
print(f"saved {EDGE_MODEL} and {p}")
