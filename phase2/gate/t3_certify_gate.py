"""T3. Certify the gate, and check the promise holds on records it never saw.

For every rotation and every trust score:
    1. on the calibration records, Learn-then-Test picks the threshold
    2. on the test records, we measure what really happens:
         coverage            share of windows whose heart rate is shown
         wrong_among_shown   share of SHOWN heart rates that are wrong (promise: <= 5%)
Also shown: the "naive" way (pick the threshold where the calibration wrong-rate is
just under 5%, no confidence margin), to show why the margin matters.

    python -m phase2.gate.t3_certify_gate

Writes results/phase2/t3_gate.json and results/figures/phase2_risk_coverage.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from phase2.common import ensure_dir
from phase2.contracts import GATE_ALPHA, GATE_DELTA
from phase2.gate.gate import evaluate, ltt_threshold, risk_coverage_curve

d = np.load(os.path.join(config.DATA, "phase2", "trust_rotations.npz"))
METHODS = ["M0_TSQI", "M1_PCLEAN", "M2_TRUSTRF"]


def naive_threshold(s, w, alpha=GATE_ALPHA):
    best = None
    for th in np.unique(np.quantile(s, np.linspace(1.0, 0.0, 200)))[::-1]:
        shown = s >= th
        if shown.any() and w[shown].mean() <= alpha:
            best = float(th)
    return best


res = {"alpha": GATE_ALPHA, "delta": GATE_DELTA, "methods": {}}
for m in METHODS:
    rows = []
    for r in range(int(d["n_rotations"])):
        sc, wc = d[f"r{r}_cal_{m}"], d[f"r{r}_cal_wrong"]
        st, wt = d[f"r{r}_test_{m}"], d[f"r{r}_test_wrong"]
        th = ltt_threshold(sc, wc, GATE_ALPHA, GATE_DELTA)
        e = evaluate(st, wt, th)
        nth = naive_threshold(sc, wc)
        ne = evaluate(st, wt, nth)
        rows.append({"rotation": r, "threshold": th, "ungated_wrong": float(wt.mean()),
                     **e, "promise_held": e["wrong_among_shown"] <= GATE_ALPHA,
                     "naive_coverage": ne["coverage"], "naive_wrong_among_shown": ne["wrong_among_shown"]})
    res["methods"][m] = {
        "rotations": rows,
        "mean_coverage": round(float(np.mean([x["coverage"] for x in rows])), 4),
        "mean_wrong_among_shown": round(float(np.mean([x["wrong_among_shown"] for x in rows])), 4),
        "promise_held": f"{sum(x['promise_held'] for x in rows)} of {len(rows)}",
        "mean_ungated_wrong": round(float(np.mean([x["ungated_wrong"] for x in rows])), 4),
    }
    s = res["methods"][m]
    print(f"{m:<11} shows {s['mean_coverage']:6.1%} of readings, wrong among shown "
          f"{s['mean_wrong_among_shown']:5.1%} (ungated {s['mean_ungated_wrong']:5.1%}), "
          f"promise held in {s['promise_held']} rotations")
best = max(METHODS, key=lambda m: (res["methods"][m]["mean_coverage"]
                                   if res["methods"][m]["promise_held"].split()[0] == res["methods"][m]["promise_held"].split()[-1]
                                   else -1))
res["recommended"] = best
print(f"recommended trust score: {best}")
p = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "t3_gate.json")
json.dump(res, open(p, "w"), indent=2)

fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
for m in METHODS:
    s = np.concatenate([d[f"r{r}_test_{m}"] for r in range(int(d["n_rotations"]))])
    w = np.concatenate([d[f"r{r}_test_wrong"] for r in range(int(d["n_rotations"]))])
    c = risk_coverage_curve(s, w)
    ax.plot(c[:, 0], c[:, 1], label=m.split("_", 1)[1])
ax.axhline(GATE_ALPHA, ls="--", c="k", lw=1)
ax.text(0.02, GATE_ALPHA + 0.005, "promise: at most 5% wrong", fontsize=7)
ax.set_xlabel("share of heart rates shown (coverage)")
ax.set_ylabel("share of shown heart rates that are wrong")
ax.set_title("Trust gate: how much can we show safely?")
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(ensure_dir(config.FIGS), "phase2_risk_coverage.png"))
print(f"saved {p} and the figure")
