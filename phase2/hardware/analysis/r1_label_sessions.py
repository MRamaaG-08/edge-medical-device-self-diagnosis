"""R1. Turn recorded sessions into a labelled dataset, using the protocol clock only.

Labels come from WHEN a window was recorded (which protocol block it sits in), never
from looking at the waveform. A window is labelled only if all 5 seconds of it sit
inside one block, at least 3 s away from either edge; everything else is dropped.
The reference heart rate comes from the numbers typed into note_hr.py.

    python -m phase2.hardware.analysis.r1_label_sessions                 # every session except SIM
    python -m phase2.hardware.analysis.r1_label_sessions --sessions SIM  # practice run on the simulator

Writes data/sessions/labelled.npz
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

import config
from phase2.common import hr_estimate
from phase2.contracts import FS, WIN
from phase2.hardware.hub.protocol import BLOCKS, GUARD_S
from selfdiag import features as F

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", nargs="*", default=None)
a = ap.parse_args()
paths = sorted(glob.glob("data/sessions/*.jsonl"))
names = [os.path.basename(p)[:-6] for p in paths]
keep = a.sessions or [n for n in names if n != "SIM"]


def ref_lookup(session):
    p = f"data/sessions/{session}_refhr.csv"
    if not os.path.exists(p):
        return None
    rows = [(float(r["wall_time"]), float(r["hr_bpm"])) for r in csv.DictReader(open(p))]
    return np.array(rows) if rows else None


cols = {k: [] for k in ("X", "y", "sev", "block", "subject", "hr_est", "hr_ref", "t_node", "W",
                        "dc_level", "perfusion_index", "signal_amplitude")}
for path, name in zip(paths, names):
    if name not in keep:
        continue
    ref = ref_lookup(name)
    n0 = len(cols["y"])
    for line in open(path):
        r = json.loads(line)
        if "window" not in r:
            continue
        end = r["t_node_s"]; start = end - WIN / FS
        b = next((b for b in BLOCKS if b[0] + GUARD_S <= start and end <= b[1] - GUARD_S), None)
        if b is None or b[2] in ("calibrate", "slow_fade"):
            continue
        w = np.asarray(r["window"], dtype=float)
        href = np.nan
        if ref is not None:
            d = np.abs(ref[:, 0] - r["t"])
            if d.min() <= 30:
                href = float(ref[np.argmin(d), 1])
        cols["X"].append(F.extract(w, FS)); cols["y"].append(config.CLASSES.index(b[3]))
        cols["sev"].append(b[4]); cols["block"].append(b[2]); cols["subject"].append(name)
        cols["hr_est"].append(hr_estimate(w)); cols["hr_ref"].append(href); cols["t_node"].append(end)
        cols["W"].append(w)
        for k in ("dc_level", "perfusion_index", "signal_amplitude"):
            cols[k].append(r["telemetry"][k])
    got = len(cols["y"]) - n0
    counts = {c: int(np.sum(np.asarray(cols["y"][n0:]) == i)) for i, c in enumerate(config.CLASSES)}
    print(f"  {name}: {got} labelled windows {counts}  reference HR: {'yes' if ref is not None else 'NO'}")
if not cols["y"]:
    raise SystemExit("no labelled windows. Record a session first (or pass --sessions SIM to practise).")
out = {k: np.asarray(v) for k, v in cols.items()}
np.savez_compressed("data/sessions/labelled.npz", **out)
print(f"saved data/sessions/labelled.npz ({len(out['y'])} windows, {len(set(out['subject']))} subjects)")
