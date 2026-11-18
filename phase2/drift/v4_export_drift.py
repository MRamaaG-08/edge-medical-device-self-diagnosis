"""V4. Freeze the drift monitor settings the hub will use.

Rule: the smallest threshold whose false alarms stay at or below 1 per hour.
(Smaller threshold = earlier alarms, so we take the most sensitive one we can afford.)

    python -m phase2.drift.v4_export_drift

Writes models/drift.json (small, goes in git)
"""
import json
import os

import config
from phase2.common import ensure_dir
from phase2.contracts import DRIFT_FILE
from phase2.drift.drift import BASELINE_WINDOWS, STREAMS

r = json.load(open(os.path.join(config.RESULTS, "phase2", "v3_drift_eval.json")))
ok = [row for row in r["thresholds"] if row["false_alarms_h"] <= 1.0]
pick = min(ok, key=lambda row: row["threshold"]) if ok else max(r["thresholds"], key=lambda row: row["threshold"])
cfg = {"streams": STREAMS, "threshold": pick["threshold"], "allowance": 0.5,
       "baseline_windows": BASELINE_WINDOWS, "evidence": pick}
ensure_dir("models")
json.dump(cfg, open(DRIFT_FILE, "w"), indent=2)
print(json.dumps(cfg, indent=2))
print(f"saved {DRIFT_FILE}")
