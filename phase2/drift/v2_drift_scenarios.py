"""V2. Make realistic "sensor slowly going wrong" recordings from BIDMC.

Each record is turned into what a MAX30102 would report (a big steady light level
with a small pulse on top), then one of four things happens to it after the
2-minute calibration:

    none            nothing; used to count false alarms
    contact_loss    the clip loosens: the whole light level fades to 50% over 2.5 minutes
    led_aging       a slow, mild fade to 85% over 4 minutes (too mild for one window to show)
    pressure_creep  the finger presses harder: the steady light level creeps up by 3% over 3 minutes

Timeline of every session (BIDMC records are 8 minutes long):
    0 to 125 s    warm-up + calibration (no windows yet)
    125 to 200 s  the drift monitor learns "normal" (24 windows)
    200 s         the fault starts (nothing changes in the "none" session)

For every 5 s window we keep the telemetry and also what the Phase 1 model says
about that single window, so V3 can compare "watching over time" with "judging one
window at a time".

    python -m phase2.drift.v2_drift_scenarios
    python -m phase2.drift.v2_drift_scenarios --records 8

Writes data/phase2/drift_streams.npz (data never goes in git)
"""
import argparse
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config
from phase2.common import ensure_dir, load_corpus, load_raw_record
from phase2.contracts import FS, SHIFT
from phase2.frontend import StreamingFrontEnd, device_like
from selfdiag import features as F
from selfdiag import preprocess as P
from selfdiag import splits

ONSET_S = 200            # fault starts after calibration AND the monitor's baseline
SCENARIOS = {
    "none":           lambda t: (np.ones_like(t), np.zeros_like(t)),
    "contact_loss":   lambda t: (1 - 0.5 * np.clip((t - ONSET_S) / 150, 0, 1), np.zeros_like(t)),
    "led_aging":      lambda t: (1 - 0.15 * np.clip((t - ONSET_S) / 240, 0, 1), np.zeros_like(t)),
    "pressure_creep": lambda t: (np.ones_like(t), 0.03 * np.clip((t - ONSET_S) / 180, 0, 1)),
}

ap = argparse.ArgumentParser()
ap.add_argument("--records", type=int, default=12)
a = ap.parse_args()

d = load_corpus()
X, y, groups, names = d["X"], d["y"], d["groups"], d["group_names"]
tr, te, test_ids = splits.record_level_holdout(groups)
rf = RandomForestClassifier(**config.RF_PARAMS).fit(X[tr], y[tr])
test_names = [str(names[g]) for g in test_ids][: a.records]

out = {"scenarios": np.array(list(SCENARIOS)), "records": np.array(test_names),
       "onset_s": ONSET_S, "shift_s": SHIFT / FS}
for rid in test_names:
    raw, fs_in = load_raw_record(rid, n_surrogate=len(names))
    base = device_like(P.resample_to(raw, fs_in, FS))
    t = np.arange(base.size) / FS
    for sc, fn in SCENARIOS.items():
        gain, creep = fn(t)
        sig = base * gain + base.mean() * creep
        fe = StreamingFrontEnd(double=True)
        wins = fe.push(sig)
        t0 = fe.calibration_samples / FS                 # time of the first window's end
        tele = {k: np.array([tl[k] for _, tl in wins]) for k in wins[0][1]}
        pred = rf.predict(np.asarray([F.extract(w, FS) for w, _ in wins]))
        key = f"{rid}__{sc}"
        out[key + "__t"] = t0 + np.arange(len(wins)) * SHIFT / FS
        out[key + "__pred"] = pred
        for k, v in tele.items():
            out[key + "__" + k] = v
    print(f"  {rid}: {len(wins)} windows per scenario")

path = os.path.join(ensure_dir(os.path.join(config.DATA, "phase2")), "drift_streams.npz")
np.savez_compressed(path, **out)
print(f"saved {path}")
