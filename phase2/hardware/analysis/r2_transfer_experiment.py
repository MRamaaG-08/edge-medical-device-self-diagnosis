"""R2. THE central question of Phase 2.

Phase 1 trained on hospital recordings with simulated faults, and it failed on a
different real sensor (BUT PPG, below chance). Does a small amount of OUR OWN
real-sensor data fix that?

Three set-ups, always tested on a person the model has never seen:
    zero_shot   trained on BIDMC only (exactly the Phase 1 recipe)
    adapted     BIDMC + the other subjects' real MAX30102 windows
    real_only   only the other subjects' real windows
And the same question for the trust gate (if reference heart rates were typed in):
    gate_zero_shot   threshold certified on BIDMC
    gate_recertified threshold certified on the other subjects' real windows

    python -m phase2.hardware.analysis.r2_transfer_experiment

Writes results/phase2/r2_transfer.json
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

import config
from phase2.common import ensure_dir, load_corpus
from phase2.contracts import GATE_ALPHA, GATE_DELTA, ec13_tolerance

R = np.load("data/sessions/labelled.npz", allow_pickle=True)
Xr, yr, subj = R["X"], R["y"], R["subject"]
subjects = sorted(set(subj.tolist()))
B = load_corpus()
Xb, yb = B["X"], B["y"]
if len(subjects) < 2:
    print("[note] only one subject: leave-one-subject-out needs at least 2. Zero-shot only.")


def scores(y, p):
    return {"accuracy": round(float((y == p).mean()), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
            "clean_vs_degraded_bal_acc": round(float(balanced_accuracy_score(y == 0, p == 0)), 4),
            "n": int(len(y))}


zero = RandomForestClassifier(**config.RF_PARAMS).fit(Xb, yb)
res = {"subjects": subjects, "zero_shot": {}, "adapted": {}, "real_only": {}}
pred_all = {k: np.empty_like(yr) for k in res if k != "subjects"}
for s in subjects:
    te = subj == s
    tr = ~te
    pred_all["zero_shot"][te] = zero.predict(Xr[te])
    if tr.any() and len(np.unique(yr[tr])) > 1:
        w = np.concatenate([np.ones(len(yb)), np.full(tr.sum(), 0.5 * len(yb) / tr.sum())])
        ad = RandomForestClassifier(**config.RF_PARAMS).fit(np.vstack([Xb, Xr[tr]]), np.concatenate([yb, yr[tr]]),
                                                            sample_weight=w)
        ro = RandomForestClassifier(**config.RF_PARAMS).fit(Xr[tr], yr[tr])
        pred_all["adapted"][te] = ad.predict(Xr[te]); pred_all["real_only"][te] = ro.predict(Xr[te])
    else:
        pred_all["adapted"][te] = -1; pred_all["real_only"][te] = -1
    for k in ("zero_shot", "adapted", "real_only"):
        if (pred_all[k][te] >= 0).all():
            res[k][s] = scores(yr[te], pred_all[k][te])
for k in ("zero_shot", "adapted", "real_only"):
    ok = pred_all[k] >= 0
    if ok.any():
        res[k]["pooled"] = scores(yr[ok], pred_all[k][ok])
        print(f"{k:<10} balanced accuracy {res[k]['pooled']['balanced_accuracy']:.3f}   "
              f"macro F1 {res[k]['pooled']['macro_f1']:.3f}   clean-vs-degraded {res[k]['pooled']['clean_vs_degraded_bal_acc']:.3f}")

href = R["hr_ref"].astype(float)
have = np.isfinite(href)
if have.sum() >= 50:
    from phase2.gate.gate import evaluate, ltt_threshold
    from phase2.gate.score import TrustScorer
    sc = TrustScorer.load()
    trust = np.array([sc.score(w) for w in R["W"]])
    wrong = (np.abs(R["hr_est"] - href) > ec13_tolerance(np.nan_to_num(href, nan=60))).astype(int)
    res["gate_zero_shot"] = evaluate(trust[have], wrong[have], sc.threshold)
    rows = []
    for s in subjects:
        te = (subj == s) & have; cal = (subj != s) & have
        if cal.sum() >= 30 and te.any():
            th = ltt_threshold(trust[cal], wrong[cal], GATE_ALPHA, GATE_DELTA)
            rows.append({"subject": s, "threshold": th, **evaluate(trust[te], wrong[te], th)})
    res["gate_recertified"] = rows
    res["ungated_wrong"] = float(wrong[have].mean())
    print(f"gate on real data, BIDMC threshold: shows {res['gate_zero_shot']['coverage']:.1%}, "
          f"wrong among shown {res['gate_zero_shot']['wrong_among_shown']:.1%} (ungated {res['ungated_wrong']:.1%})")
else:
    print("[note] fewer than 50 windows with a reference heart rate: gate part skipped")
p = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "r2_transfer.json")
json.dump(res, open(p, "w"), indent=2, default=float)
print(f"saved {p}")
