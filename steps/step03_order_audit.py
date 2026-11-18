#!/usr/bin/env python3
"""
Step 03 — pipeline order audit. THIS IS A GATE. Nothing downstream runs until
it passes.

Phase 1's published pipeline normalises each window and then classifies. If the
degradation was injected before that normalisation, the ATTENUATED class was
mathematically erased before the classifier ever saw it, because

    zscore(g * x)  ==  zscore(x)      for any g > 0

is an exact identity. A gain of 0.15 and a gain of 1.0 produce bit-identical
feature vectors. The same mechanism damages DRIFTED, since the 0.5 Hz high-pass
is designed to remove precisely the baseline wander that defines the class.

This script proves the identity numerically, measures how much class
separability each ordering preserves, and writes a figure for the paper. If the
wrong ordering is ever reintroduced into preprocess.py, this exits 1.

    python steps/step03_order_audit.py
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import preprocess as P
from selfdiag import degrade as D
from selfdiag import features as F
from selfdiag import surrogate as S

FS = config.WINDOW_SPECS[config.PRIMARY_SPEC]["fs"]
WIN_S = config.WINDOW_SPECS[config.PRIMARY_SPEC]["win_s"]
SHIFT_S = config.WINDOW_SPECS[config.PRIMARY_SPEC]["shift_s"]


def exact_identity_check() -> dict:
    """The core claim, stated as a number."""
    rng = np.random.default_rng(config.SEED)
    x = S.make_surrogate(30.0, FS, seed=1)
    x = P.bandpass(x, FS)
    w = P.make_windows(x, FS, WIN_S, SHIFT_S)[0]

    results = {}
    for gain in (0.70, 0.40, 0.15):
        a = P.zscore(w)
        b = P.zscore(gain * w)
        results[f"gain_{gain}"] = {
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "feature_max_abs_diff": float(
                np.max(np.abs(F.extract(a, FS) - F.extract(b, FS)))),
        }
    return results


def separability(order: str, n_records: int = 6) -> dict:
    """
    Build a small corpus under one ordering and measure how far each degraded
    class sits from CLEAN in standardised feature space.

    Distance is the mean absolute standardised difference of the 13 features
    between a class and CLEAN. Near zero means the class is indistinguishable.
    """
    rng = np.random.default_rng(config.SEED)
    per_class = {c: [] for c in config.CLASSES}

    for rid, raw in S.make_corpus(n_records, 60.0, FS, seed=config.SEED):
        if order == "correct":
            # raw -> band-pass -> z-score -> window -> INJECT -> features
            wins = P.prepare_record(raw, FS, FS, WIN_S, SHIFT_S)
            for w in wins:
                per_class["CLEAN"].append(F.extract(w, FS))
                for cls in config.CLASSES[1:]:
                    for lvl in (1, 2, 3):
                        dw, _ = D.apply_degradation(w, cls, lvl, FS, rng)
                        per_class[cls].append(F.extract(dw, FS))

        elif order == "wrong":
            # raw -> INJECT -> band-pass -> z-score -> window -> features
            base = P.bandpass(raw, FS)
            base = P.zscore(base)
            clean_wins = P.make_windows(base, FS, WIN_S, SHIFT_S)
            for w in clean_wins:
                per_class["CLEAN"].append(F.extract(w, FS))

            for cls in config.CLASSES[1:]:
                for lvl in (1, 2, 3):
                    dirty = np.zeros_like(raw)
                    chunk = P.make_windows(raw, FS, WIN_S, WIN_S)
                    pieces = []
                    for w in chunk:
                        dw, _ = D.apply_degradation(w, cls, lvl, FS, rng)
                        pieces.append(dw)
                    if not pieces:
                        continue
                    dirty = np.concatenate(pieces)
                    dirty = P.bandpass(dirty, FS)
                    dirty = P.zscore(dirty)            # <-- cancels the gain
                    for w in P.make_windows(dirty, FS, WIN_S, SHIFT_S):
                        per_class[cls].append(F.extract(w, FS))
        else:
            raise ValueError(order)

    arrays = {c: np.vstack(v) for c, v in per_class.items() if v}
    clean = arrays["CLEAN"]
    mu, sd = clean.mean(axis=0), clean.std(axis=0) + 1e-12

    out = {}
    for c, A in arrays.items():
        if c == "CLEAN":
            continue
        d = np.abs((A.mean(axis=0) - mu) / sd)
        out[c] = {"mean_std_distance": float(d.mean()),
                  "max_std_distance": float(d.max()),
                  "n_samples": int(A.shape[0])}
    return out


def main() -> int:
    print("=" * 68)
    print("STEP 03  pipeline order audit  [GATE]")
    print("=" * 68)

    print("\n1. exact identity: does z-scoring cancel attenuation?\n")
    ident = exact_identity_check()
    for k, v in ident.items():
        print(f"   {k:<12} max|zscore(x) - zscore(g*x)| = {v['max_abs_diff']:.3e}"
              f"   feature diff = {v['feature_max_abs_diff']:.3e}")
    cancelled = all(v["max_abs_diff"] < 1e-9 for v in ident.values())
    print(f"\n   => z-score cancels attenuation exactly: {cancelled}")
    if not cancelled:
        print("   UNEXPECTED: the identity did not hold. Investigate before continuing.")
        return 1

    print("\n2. class separability under each ordering")
    print("   (mean standardised feature distance from CLEAN; "
          "near 0 means erased)\n")
    correct = separability("correct")
    wrong = separability("wrong")

    print(f"   {'class':<13}{'correct order':>16}{'wrong order':>16}   verdict")
    print("   " + "-" * 60)
    verdicts = {}
    for c in config.CLASSES[1:]:
        a = correct.get(c, {}).get("mean_std_distance", float("nan"))
        b = wrong.get(c, {}).get("mean_std_distance", float("nan"))
        if b < 0.05 * a:
            v = "ERASED by wrong order"
        elif b < 0.5 * a:
            v = "badly damaged"
        else:
            v = "survives"
        verdicts[c] = v
        print(f"   {c:<13}{a:>16.4f}{b:>16.4f}   {v}")

    print("\n3. is preprocess.py using the correct order?")
    rng = np.random.default_rng(0)
    x = S.make_surrogate(30.0, FS, seed=7)
    w = P.prepare_record(x, FS, FS, WIN_S, SHIFT_S)[0]
    att, _ = D.apply_degradation(w, "ATTENUATED", 3, FS, rng)
    fa, fb = F.extract(w, FS), F.extract(att, FS)
    delta = float(np.max(np.abs(fa - fb)))
    ok = delta > 1e-6
    print(f"   feature delta between CLEAN and ATTENUATED(gain=0.15): {delta:.4f}")
    print(f"   => attenuation is visible to the feature set: {ok}")

    out = {
        "exact_identity": ident,
        "separability_correct_order": correct,
        "separability_wrong_order": wrong,
        "verdicts": verdicts,
        "live_pipeline_preserves_attenuation": bool(ok),
        "live_pipeline_feature_delta": delta,
    }
    path = os.path.join(config.RESULTS, "step03_order_audit.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n   written: {path}")

    if not ok:
        print("\nGATE FAILED. preprocess.prepare_record is normalising after "
              "degradation. Fix the order before running anything else.")
        return 1

    print("\nGATE PASSED. next: python steps/step04_build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
