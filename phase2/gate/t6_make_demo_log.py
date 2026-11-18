"""T6. Make a practice hub log so the dashboard can be built before any hardware exists.

It writes exactly the same JSON Lines file the Pi hub writes (see LOG_FIELDS in
phase2/contracts.py), from a BIDMC record with faults switched on and off on a
script:

     0 to  60 s  clean
    60 to  90 s  noisy (heavy)
    90 to 120 s  clean
   120 to 150 s  attenuated (clip loose)
   150 to 180 s  spikes (cable knocked)
   180 s on      slow drift; the drift alarm is raised after a minute

    python -m phase2.gate.t6_make_demo_log
    python -m phase2.gate.t6_make_demo_log --record bidmc05

Writes data/phase2/demo_log.jsonl
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config
from phase2.common import ensure_dir, hr_estimate, load_corpus, load_raw_record
from phase2.contracts import FS, SHIFT
from phase2.gate.score import TrustScorer
from selfdiag import degrade as D
from selfdiag import features as F
from selfdiag import preprocess as P

ap = argparse.ArgumentParser()
ap.add_argument("--record", default=None)
a = ap.parse_args()

c = load_corpus()
names = [str(n) for n in c["group_names"]]
rid = a.record or names[0]
node_model = RandomForestClassifier(**config.RF_PARAMS).fit(c["X"], c["y"])   # stands in for the ESP32 model
scorer = TrustScorer.load()
raw, fs_in = load_raw_record(rid, n_surrogate=len(names))
spec = config.WINDOW_SPECS[config.PRIMARY_SPEC]
wins = P.prepare_record(raw, fs_in, FS, spec["win_s"], spec["shift_s"])
rng = np.random.default_rng(config.SEED)
SCRIPT = [(60, None, 0), (90, "NOISY", 3), (120, None, 0), (150, "ATTENUATED", 3),
          (180, "SPIKED", 3), (10 ** 9, "DRIFTED", 2)]
t0 = time.time()
path = os.path.join(ensure_dir(os.path.join(config.DATA, "phase2")), "demo_log.jsonl")
with open(path, "w") as fh:
    for i, w in enumerate(wins):
        ts = i * SHIFT / FS
        cls, lvl = next((c_, l_) for end, c_, l_ in SCRIPT if ts < end)
        if cls:
            w = D.apply_degradation(w, cls, lvl, FS, rng)[0]
        p = node_model.predict_proba(F.extract(w, FS)[None, :])[0]
        trust = scorer.score(w)
        rel = scorer.released(trust)
        drift = ["dc_level"] if ts >= 240 else []
        rec = {"t": round(t0 + ts, 2), "seq": i, "t_node_s": round(ts + 5.0, 2),
               "node_class": config.CLASSES[int(np.argmax(p))],
               "node_proba": [round(float(v), 3) for v in p],
               "hub_class": config.CLASSES[int(np.argmax(p))],
               "trust": round(trust, 3), "released": bool(rel),
               "hr": round(hr_estimate(w), 1) if rel else None,
               "drift_alarms": drift,
               "telemetry": {"dc_level": round(50000 * (1 - 0.002 * max(0, ts - 180)), 1),
                             "perfusion_index": 1.0, "die_temp_c": 31.5, "led_current": 6.4},
               "window": [round(float(v), 3) for v in w]}
        fh.write(json.dumps(rec) + "\n")
print(f"wrote {len(wins)} windows from {rid} to {path}")
print("now:  streamlit run phase2/gate/dashboard/app.py")
