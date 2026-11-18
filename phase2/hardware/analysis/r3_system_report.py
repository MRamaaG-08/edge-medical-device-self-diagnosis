"""R3. The end-to-end system numbers for one session (or several).

    python -m phase2.hardware.analysis.r3_system_report --sessions S01 S02

For each session: how often the chip and the Pi agree (live parity), what share of
heart rates the gate released in each kind of block, when the drift monitor fired
during the slow LED fade (and anywhere else = false alarm), and how long the chip
and the Pi spend per window.

Writes results/phase2/r3_system_report.json
"""
import argparse
import json
import os

import numpy as np

import config
from phase2.common import ensure_dir
from phase2.contracts import FS, WIN
from phase2.hardware.hub.protocol import FADE_START, block_at

ap = argparse.ArgumentParser(); ap.add_argument("--sessions", nargs="+", required=True)
a = ap.parse_args()
out = {}
for s in a.sessions:
    rows = [json.loads(x) for x in open(f"data/sessions/{s}.jsonl") if x.strip()]
    agree = [r["node_class"] == r["hub_class"] for r in rows if r.get("hub_class")]
    by_class = {}
    for r in rows:
        b = block_at(r["t_node_s"] - WIN / FS / 2)
        if b:
            by_class.setdefault(b[3] if b[2] != "slow_fade" else "SLOW_FADE", []).append(r["released"])
    alarms = [(r["t_node_s"], r["drift_alarms"]) for r in rows if r["drift_alarms"]]
    fade_hits = [t for t, _ in alarms if t >= FADE_START]
    first_att = next((r["t_node_s"] for r in rows if r["t_node_s"] >= FADE_START and r["node_class"] == "ATTENUATED"), None)
    out[s] = {"windows": len(rows),
              "chip_pi_agreement": round(float(np.mean(agree)), 4) if agree else None,
              "released_share_by_block_class": {k: round(float(np.mean(v)), 3) for k, v in by_class.items()},
              "drift_alarm_delay_s": round(fade_hits[0] - FADE_START, 1) if fade_hits else None,
              # an alarm during a clean/rest block before the fade is a false alarm; alarms during
              # the deliberate fault blocks (dimmed LED, tapping ...) are correct detections
              "drift_false_alarms": len([t for t, _ in alarms if t < FADE_START
                                         and (block_at(t) or (0, 0, "", "CLEAN"))[3] == "CLEAN"
                                         and (block_at(t - 10) or (0, 0, "", "CLEAN"))[3] == "CLEAN"]),
              "drift_alarms_during_fault_blocks": len([t for t, _ in alarms if t < FADE_START
                                                       and (block_at(t) or (0, 0, "", "CLEAN"))[3] != "CLEAN"]),
              "per_window_model_first_attenuated_delay_s": round(first_att - FADE_START, 1) if first_att else None,
              "node_ms_median": round(float(np.median([r.get("node_ms", 0) for r in rows])), 2),
              "hub_ms_median": round(float(np.median([r.get("hub_ms", 0) for r in rows])), 2)}
    print(json.dumps({s: out[s]}, indent=2))
p = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "r3_system_report.json")
json.dump(out, open(p, "w"), indent=2)
print(f"saved {p}")
