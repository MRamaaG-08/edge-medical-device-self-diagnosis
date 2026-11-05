"""V3. How early, and how reliably, does the drift monitor catch a failing sensor?

For each alarm threshold it reports:
    detected         share of faulty sessions where the alarm went off after the fault began
    delay_s          median time from fault start to alarm
    false_alarms_h   alarms per hour on sessions where nothing was wrong (lower is better)
and, for comparison, how the Phase 1 per-window model does on the same sessions
(it "detects" when 3 windows in a row are called ATTENUATED).

    python -m phase2.drift.v3_drift_eval

Writes results/phase2/v3_drift_eval.json and results/figures/phase2_drift_tradeoff.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from phase2.common import ensure_dir
from phase2.drift.drift import STREAMS, DriftMonitor

THRESHOLDS = [5, 10, 20, 40, 80]
ATT = config.CLASSES.index("ATTENUATED")

d = np.load(os.path.join(config.DATA, "phase2", "drift_streams.npz"), allow_pickle=True)
records, scenarios, onset = list(d["records"]), list(d["scenarios"]), float(d["onset_s"])


def run_monitor(key, th):
    mon = DriftMonitor(threshold=th)
    times = d[key + "__t"]
    alarms = []
    for i, t in enumerate(times):
        fired = mon.update({s: d[f"{key}__{s}"][i] for s in STREAMS})
        if fired:
            alarms.append((float(t), fired))
            mon.reset()            # after an alarm the device recalibrates
    return alarms


def classifier_alarm(key, run=3):
    pred, times = d[key + "__pred"], d[key + "__t"]
    hit = np.convolve((pred == ATT).astype(int), np.ones(run, int), "valid") >= run
    idx = np.where(hit)[0]
    return float(times[idx[0] + run - 1]) if idx.size else None


rows = []
for th in THRESHOLDS:
    row = {"threshold": th}
    fa, hours = 0, 0.0
    for r in records:
        key = f"{r}__none"
        fa += len(run_monitor(key, th))
        tt = d[key + "__t"]
        hours += (tt[-1] - tt[0]) / 3600
    row["false_alarms_h"] = round(float(fa / hours), 2)
    for sc in scenarios:
        if sc == "none":
            continue
        delays = []
        for r in records:
            hits = [t for t, _ in run_monitor(f"{r}__{sc}", th) if t >= onset]
            if hits:
                delays.append(hits[0] - onset)
        row[f"{sc}_detected"] = round(len(delays) / len(records), 3)
        row[f"{sc}_delay_s"] = round(float(np.median(delays)), 1) if delays else None
    rows.append(row)
    print(row)

clf = {}
for sc in scenarios:
    if sc == "none":
        continue
    delays = [t - onset for t in (classifier_alarm(f"{r}__{sc}") for r in records)
              if t is not None and t >= onset]
    clf[str(sc)] = {"detected": round(len(delays) / len(records), 3),
               "delay_s": round(float(np.median(delays)), 1) if delays else None}
fa = sum(1 for r in records if classifier_alarm(f"{r}__none") is not None)
clf["none_sessions_with_alarm"] = fa
print("Phase 1 per-window model on the same sessions:", clf)

res = {"thresholds": rows, "per_window_classifier": clf, "streams": STREAMS,
       "n_records": len(records), "onset_s": onset}
out = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "v3_drift_eval.json")
json.dump(res, open(out, "w"), indent=2)

fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
for sc, mk in (("contact_loss", "o"), ("led_aging", "s"), ("pressure_creep", "^")):
    ax.plot([r["false_alarms_h"] for r in rows], [r[f"{sc}_detected"] for r in rows],
            marker=mk, label=sc.replace("_", " "))
ax.set_xlabel("false alarms per hour (healthy sessions)")
ax.set_ylabel("share of faulty sessions caught")
ax.set_title("Drift monitor: catch rate vs false alarms")
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ensure_dir(config.FIGS), "phase2_drift_tradeoff.png"))
print(f"saved {out} and the figure")
