"""The one function the hub calls: window in, trust score out.

Whatever trust score T3 recommended (T_SQI, P(CLEAN) or the trust Random Forest),
the hub only ever does:

    scorer = TrustScorer.load()
    trust = scorer.score(window)          # 0..1, higher = more trustworthy
    released = trust >= scorer.threshold
"""
import json

import joblib
import numpy as np

from phase2.contracts import FS, GATE_FILE, TRUST_MODEL
from selfdiag import features as F
from selfdiag import sqi as Q


class TrustScorer:
    def __init__(self, gate, model=None):
        self.gate = gate
        self.method = gate["method"]
        self.threshold = gate["threshold"]
        self.model = model

    @classmethod
    def load(cls, gate_file=GATE_FILE, model_file=TRUST_MODEL):
        gate = json.load(open(gate_file))
        model = None if gate["method"] == "M0_TSQI" else joblib.load(model_file)
        return cls(gate, model)

    def score(self, w):
        w = np.asarray(w, dtype=float)
        if self.method == "M0_TSQI":
            return float(Q.extract_sqi_batch(w[None, :], FS)[0][2])
        p = self.model.predict_proba(F.extract(w, FS)[None, :])[0]
        positive = 0 if self.method == "M1_PCLEAN" else 1      # CLEAN class / "not wrong"
        return float(p[list(self.model.classes_).index(positive)])

    def released(self, trust):
        return self.threshold is not None and trust >= self.threshold
