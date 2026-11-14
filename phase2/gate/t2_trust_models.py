"""T2. Three ways to score trust, compared fairly.

    M0  T_SQI      Orphanidou template-matching quality index (best published SQI in Phase 1)
    M1  P(CLEAN)   Phase 1 five-class model: probability that the window is clean
    M2  trust RF   a Random Forest trained directly on the outcome label (WRONG or not)

Fair = record-level. The records are split 5 ways. In each round one fifth is the
test set; of the rest, three quarters train the models and one quarter is kept
aside to set the gate threshold later (T3). No record ever sits in two roles.

    python -m phase2.gate.t2_trust_models

Writes data/phase2/trust_rotations.npz and results/phase2/t2_trust_models.json
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import config
from phase2.common import ensure_dir

T_SQI = 2      # column of T_SQI in S (S_SQI, R_SQI, T_SQI)
d = np.load(os.path.join(config.DATA, "phase2", "trust_corpus.npz"))
X, S, wrong, cls, g = d["X"], d["S"], d["wrong"], d["cls"], d["group"]
rng = np.random.default_rng(config.SEED)
out, res = {}, {"rotations": []}
k = min(5, len(np.unique(g)))
for r, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, groups=g)):
    tg = np.unique(g[tr]); rng.shuffle(tg)
    cal_groups = set(tg[: max(1, len(tg) // 4)].tolist())
    cal = np.array([i for i in tr if g[i] in cal_groups])
    fit = np.array([i for i in tr if g[i] not in cal_groups])
    m1 = RandomForestClassifier(**config.RF_PARAMS).fit(X[fit], cls[fit])
    m2 = RandomForestClassifier(**config.RF_PARAMS).fit(X[fit], 1 - wrong[fit])
    row = {"rotation": r, "test_records": sorted(int(x) for x in np.unique(g[te]))}
    for part, idx in (("cal", cal), ("test", te)):
        scores = {"M0_TSQI": S[idx, T_SQI],
                  "M1_PCLEAN": m1.predict_proba(X[idx])[:, list(m1.classes_).index(0)],
                  "M2_TRUSTRF": m2.predict_proba(X[idx])[:, list(m2.classes_).index(1)]}
        for name, s in scores.items():
            out[f"r{r}_{part}_{name}"] = s
        out[f"r{r}_{part}_wrong"] = wrong[idx]
        if part == "test":
            for name, s in scores.items():
                row[f"auc_{name}"] = round(float(roc_auc_score(1 - wrong[idx], s)), 4)
    res["rotations"].append(row)
    print(f"rotation {r}: AUC  M0 {row['auc_M0_TSQI']:.3f}   M1 {row['auc_M1_PCLEAN']:.3f}   "
          f"M2 {row['auc_M2_TRUSTRF']:.3f}")
out["n_rotations"] = k
for name in ("M0_TSQI", "M1_PCLEAN", "M2_TRUSTRF"):
    v = [row[f"auc_{name}"] for row in res["rotations"]]
    res[f"auc_{name}_mean"] = round(float(np.mean(v)), 4)
np.savez_compressed(os.path.join(config.DATA, "phase2", "trust_rotations.npz"), **out)
p = os.path.join(ensure_dir(os.path.join(config.RESULTS, "phase2")), "t2_trust_models.json")
json.dump(res, open(p, "w"), indent=2)
print(f"saved {p}")
