"""T4. Freeze the trust model and the certified threshold for the device.

Uses the trust score T3 recommended. Three quarters of the records train the
model; the remaining quarter (never used for training) certifies the threshold.

    python -m phase2.gate.t4_export_gate
    python -m phase2.gate.t4_export_gate --method M2_TRUSTRF      # force a choice

Writes models/trust_rf.joblib (not in git, too big) and models/gate.json (in git)
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config
from phase2.common import ensure_dir
from phase2.contracts import GATE_ALPHA, GATE_DELTA, GATE_FILE, TRUST_MODEL
from phase2.gate.gate import evaluate, ltt_threshold

ap = argparse.ArgumentParser()
ap.add_argument("--method", default=None, choices=["M0_TSQI", "M1_PCLEAN", "M2_TRUSTRF"])
a = ap.parse_args()
method = a.method or json.load(open(os.path.join(config.RESULTS, "phase2", "t3_gate.json")))["recommended"]

d = np.load(os.path.join(config.DATA, "phase2", "trust_corpus.npz"))
X, S, wrong, cls, g = d["X"], d["S"], d["wrong"], d["cls"], d["group"]
rng = np.random.default_rng(config.SEED)
groups = np.unique(g); rng.shuffle(groups)
cal_groups = set(groups[: max(1, len(groups) // 4)].tolist())
cal = np.array([gi in cal_groups for gi in g])

model = None
if method == "M0_TSQI":
    s_cal = S[cal, 2]
else:
    target = cls if method == "M1_PCLEAN" else 1 - wrong
    model = RandomForestClassifier(**config.RF_PARAMS).fit(X[~cal], target[~cal])
    positive = 0 if method == "M1_PCLEAN" else 1
    s_cal = model.predict_proba(X[cal])[:, list(model.classes_).index(positive)]
th = ltt_threshold(s_cal, wrong[cal], GATE_ALPHA, GATE_DELTA)
e = evaluate(s_cal, wrong[cal], th)
ensure_dir("models")
if model is not None:
    joblib.dump(model, TRUST_MODEL)
gate = {"method": method, "threshold": th, "alpha": GATE_ALPHA, "delta": GATE_DELTA,
        "n_calibration_windows": int(cal.sum()), "calibration_records": int(len(cal_groups)),
        "calibration_coverage": round(e["coverage"], 4),
        "calibration_wrong_among_shown": round(e["wrong_among_shown"], 4),
        "label": "HR within ANSI/AAMI EC13 tolerance of the ECG reference",
        "reference": [str(x) for x in d["reference"]]}
json.dump(gate, open(GATE_FILE, "w"), indent=2)
print(json.dumps(gate, indent=2))
print(f"saved {GATE_FILE}" + (f" and {TRUST_MODEL}" if model is not None else ""))
