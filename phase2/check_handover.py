"""Ramaa runs this after merging a handover pull request. It answers one question:
is everything I need here, and does it work?

    python -m phase2.check_handover              # everyone
    python -m phase2.check_handover vandana      # one person

Exit code 0 only when every check for the chosen people passes.
"""
import json
import os
import shutil
import subprocess
import sys

CHECKS = {
    "vandana": {
        "files": ["phase2/drift/drift.py", "models/drift.json", "results/phase2/v1_frontend_check.json",
                  "results/phase2/v3_drift_eval.json", "results/figures/phase2_drift_tradeoff.png",
                  "handover/vandana.md"],
        "json": [("models/drift.json", ["streams", "threshold", "baseline_windows"]),
                 ("results/phase2/v1_frontend_check.json", ["accuracy_mean", "recommended_frontend"])],
    },
    "tejaswini": {
        "files": ["phase2/gate/gate.py", "phase2/gate/score.py", "phase2/gate/dashboard/app.py",
                  "models/gate.json", "results/phase2/t2_trust_models.json", "results/phase2/t3_gate.json",
                  "results/figures/phase2_risk_coverage.png", "handover/tejaswini.md"],
        "json": [("models/gate.json", ["method", "threshold", "alpha", "delta"])],
    },
    "bhumika": {
        "files": ["models/edge_rf.joblib", "phase2/edge/c/edge_rf.h", "phase2/edge/c/edge_filter.h",
                  "phase2/edge/c/edge_features.c", "phase2/edge/c/edge_frontend.c", "phase2/edge/c/edge_model.c",
                  "tests/golden_edge/raw.csv", "results/phase2/b1_edge_rf.json", "results/phase2/b3_parity.txt",
                  "handover/bhumika.md"],
        "json": [("results/phase2/b1_edge_rf.json", ["chosen", "candidates"])],
        "parity": True,
    },
}


def run(who):
    c = CHECKS[who]
    ok = True
    print(f"\n=== {who} ===")
    for f in c["files"]:
        there = os.path.exists(f)
        ok &= there
        print(f"  {'ok     ' if there else 'MISSING'}  {f}")
    for f, keys in c.get("json", []):
        if os.path.exists(f):
            d = json.load(open(f))
            miss = [k for k in keys if k not in d]
            ok &= not miss
            print(f"  {'ok     ' if not miss else 'BAD    '}  {f} has {', '.join(keys)}" + (f" (missing {miss})" if miss else ""))
    if c.get("parity"):
        if shutil.which("gcc"):
            r = subprocess.run([sys.executable, "-m", "phase2.edge.b3_parity"], capture_output=True, text=True)
            passed = "PARITY PASSED" in r.stdout
            ok &= passed
            print(f"  {'ok     ' if passed else 'FAILED '}  C parity re-run on this laptop")
        else:
            print("  skip     C parity (gcc not installed here)")
    print(f"  -> {'READY TO INTEGRATE' if ok else 'NOT READY: ask for the missing pieces'}")
    return ok


who = sys.argv[1:] or list(CHECKS)
results = [run(w) for w in who]
sys.exit(0 if all(results) else 1)
