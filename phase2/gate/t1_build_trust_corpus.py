"""T1. Build the trust corpus: every window gets an OUTCOME label.

Phase 1 asked "which fault does this window have?". Phase 2 asks the question a
doctor cares about: "if we show the heart rate from this window, is it right?"

For every clean 5 s window of every BIDMC record, and for one copy of it with each
fault (at a random severity), we:
    1. estimate the heart rate from the window itself (strongest pulse frequency)
    2. look up the monitor's ECG heart rate for the same seconds (the truth)
    3. label the window WRONG if the two differ by more than the ANSI/AAMI EC13
       tolerance: 5 bpm or 10%, whichever is larger.

    python -m phase2.gate.t1_build_trust_corpus
    python -m phase2.gate.t1_build_trust_corpus --records 10     # quick practice run

Writes data/phase2/trust_corpus.npz
"""
import argparse
import os

import numpy as np

import config
from phase2.common import ensure_dir, hr_estimate, load_corpus, load_raw_record
from phase2.contracts import FS, SHIFT, WIN, ec13_tolerance
from phase2.gate.bidmc_hr import reference_hr, window_reference
from selfdiag import degrade as D
from selfdiag import features as F
from selfdiag import preprocess as P
from selfdiag import sqi as Q

ap = argparse.ArgumentParser()
ap.add_argument("--records", type=int, default=None)
a = ap.parse_args()

names = [str(n) for n in load_corpus()["group_names"]]
if a.records:
    names = names[: a.records]
rng = np.random.default_rng(config.SEED)
spec = config.WINDOW_SPECS[config.PRIMARY_SPEC]
cols = {k: [] for k in ("X", "S", "hr_est", "hr_ref", "wrong", "cls", "sev", "group")}
sources = set()
for gi, rid in enumerate(names):
    try:
        raw, fs_in = load_raw_record(rid, n_surrogate=len(load_corpus()["group_names"]))
        hr_s, src = reference_hr(rid, n_records=len(load_corpus()["group_names"]))
    except Exception as e:                       # noqa: BLE001
        print(f"  [skip] {rid}: {e}")
        continue
    sources.add(src)
    wins = P.prepare_record(raw, fs_in, FS, spec["win_s"], spec["shift_s"])
    for i, w in enumerate(wins):
        ref = window_reference(hr_s, i * SHIFT, WIN)
        if not np.isfinite(ref):
            continue
        versions = [(w, 0, 0)]
        for ci, cls in enumerate(config.CLASSES[1:], start=1):
            lvl = int(rng.integers(1, 4))
            versions.append((D.apply_degradation(w, cls, lvl, FS, rng)[0], ci, lvl))
        for v, ci, lvl in versions:
            est = hr_estimate(v)
            cols["X"].append(F.extract(v, FS))
            cols["S"].append(Q.extract_sqi_batch(v[None, :], FS)[0])
            cols["hr_est"].append(est); cols["hr_ref"].append(ref)
            cols["wrong"].append(int(abs(est - ref) > ec13_tolerance(ref)))
            cols["cls"].append(ci); cols["sev"].append(lvl); cols["group"].append(gi)
    print(f"  {rid}: {len(wins)} windows, reference = {src}")

out = {k: np.asarray(v) for k, v in cols.items()}
path = os.path.join(ensure_dir(os.path.join(config.DATA, "phase2")), "trust_corpus.npz")
np.savez_compressed(path, **out, group_names=np.array(names), reference=np.array(sorted(sources)))
wrong = out["wrong"]
print(f"\n{len(wrong):,} windows, {wrong.mean():.1%} would show a WRONG heart rate if released")
for ci, c in enumerate(config.CLASSES):
    m = out["cls"] == ci
    print(f"   {c:<11} wrong {wrong[m].mean():6.1%}")
print(f"saved {path}")
