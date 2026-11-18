"""T5. Does the gate keep its promise on a completely different sensor?

BUT PPG holds 10-second smartphone-camera PPG clips, each with a reference heart
rate from an ECG chest strap. Phase 1 showed the fault model does not transfer to
this kind of sensor (below chance). Here we ask the Phase 2 question instead: when
the gate DOES release a heart rate on this unfamiliar sensor, is it still right?

    python -m phase2.gate.t5_butppg_check

Needs the BUT PPG download in data/butppg/ (same as Phase 1 step07).
Writes results/phase2/t5_butppg_check.json
"""
import csv
import glob
import json
import os

import numpy as np

import config
from phase2.common import ensure_dir, hr_estimate
from phase2.contracts import FS, WIN, ec13_tolerance
from phase2.gate.score import TrustScorer
from selfdiag import io_butppg
from selfdiag import preprocess as P


def load_reference_hr():
    """Find the annotation CSV that has a heart-rate column. Returns {record_id: hr}."""
    for path in sorted(glob.glob(os.path.join(config.BUTPPG_DIR, "**", "*.csv"), recursive=True)):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        if len(rows) < 2:
            continue
        low = [h.strip().lower() for h in rows[0]]
        hr_col = next((i for i, h in enumerate(low) if h in ("hr", "heart rate", "heartrate") or h.startswith("hr")), None)
        id_col = next((i for i, h in enumerate(low) if h in ("id", "record", "record id", "name")), 0)
        if hr_col is None:
            continue
        out = {}
        for r in rows[1:]:
            try:
                out[r[id_col].strip()] = float(r[hr_col])
            except (ValueError, IndexError):
                continue
        if out:
            print(f"  reference HR from {os.path.relpath(path, config.BUTPPG_DIR)} ({len(out)} records)")
            return out
    raise FileNotFoundError("no CSV with an HR column under data/butppg/")


ref = load_reference_hr()
scorer = TrustScorer.load()
rows = []
for rid, x, fs in io_butppg.load_records():
    if rid not in ref or not (30 <= ref[rid] <= 220):
        continue
    y = P.zscore(P.bandpass(P.resample_to(x, fs, FS), FS))
    if y.size < WIN:
        continue
    w = y[(y.size - WIN) // 2:(y.size - WIN) // 2 + WIN]
    est = hr_estimate(w)
    t = scorer.score(w)
    rows.append((t, int(abs(est - ref[rid]) > ec13_tolerance(ref[rid]))))
t, wrong = np.array(rows).T
shown = t >= scorer.threshold if scorer.threshold is not None else np.zeros_like(t, bool)
res = {"n_clips": int(len(t)), "ungated_wrong": float(wrong.mean()),
       "coverage": float(shown.mean()),
       "wrong_among_shown": float(wrong[shown].mean()) if shown.any() else None,
       "promise_alpha": scorer.gate["alpha"], "method": scorer.method}
res["promise_held"] = res["wrong_among_shown"] is not None and res["wrong_among_shown"] <= res["promise_alpha"]
print(json.dumps(res, indent=2))
p = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "t5_butppg_check.json")
json.dump(res, open(p, "w"), indent=2)
print(f"saved {p}\nWhatever this says goes in the paper as it is. A broken promise on a phone camera is the"
      "\nevidence that the gate must be re-certified for each new sensor, which is exactly what Ramaa's"
      "\nreal MAX30102 recordings are for.")
