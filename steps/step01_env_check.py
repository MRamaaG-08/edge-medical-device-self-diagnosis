#!/usr/bin/env python3
"""
Step 01 — environment check.

Run this first. It fails loudly rather than letting a version problem surface
six hours later as a number that is subtly wrong.

Python 3.12 is the target. 3.14 is deliberately rejected: SciPy, scikit-learn,
onnxruntime and wfdb wheel availability on 3.14 is still incomplete, and a
source build of SciPy will silently give you different Butterworth coefficients
from the pinned version.
"""
import sys
import os
import platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED = ["numpy", "scipy", "sklearn", "matplotlib", "wfdb", "joblib"]
OPTIONAL = ["skl2onnx", "onnxruntime"]


def main() -> int:
    print("=" * 68)
    print("STEP 01  environment check")
    print("=" * 68)
    print(f"python       {platform.python_version()}  ({sys.executable})")
    print(f"platform     {platform.platform()}")

    problems, warnings = [], []

    major, minor = sys.version_info[:2]
    if (major, minor) == (3, 12):
        print("python ver   ok (3.12, the target)")
    elif (major, minor) == (3, 13):
        print("python ver   ok (3.13, acceptable)")
    elif (major, minor) >= (3, 14):
        problems.append(
            f"Python {major}.{minor} is too new. Create a 3.12 venv:\n"
            "    py -3.12 -m venv .venv           (Windows)\n"
            "    python3.12 -m venv .venv         (Linux/macOS)\n"
            "    .venv\\Scripts\\activate          /  source .venv/bin/activate\n"
            "    pip install -r requirements.txt")
    else:
        problems.append(f"Python {major}.{minor} is too old, need 3.12")

    print("\npackages")
    for mod in REQUIRED:
        try:
            m = __import__(mod)
            print(f"  {mod:<14} {getattr(m, '__version__', '?')}")
        except ImportError:
            problems.append(f"missing required package: {mod}")
            print(f"  {mod:<14} MISSING")

    for mod in OPTIONAL:
        try:
            m = __import__(mod)
            print(f"  {mod:<14} {getattr(m, '__version__', '?')}  (optional)")
        except ImportError:
            warnings.append(f"{mod} missing — step08 edge benchmark will be skipped")
            print(f"  {mod:<14} missing (optional)")

    try:
        import scipy
        from packaging.version import Version
        if Version(scipy.__version__) < Version("1.11"):
            warnings.append(
                f"SciPy {scipy.__version__} is old; filter coefficients may differ "
                "from the pinned version")
    except Exception:
        pass

    import config
    print("\npaths")
    for name in ("DATA", "BIDMC_DIR", "BUTPPG_DIR", "RESULTS", "FIGS"):
        p = getattr(config, name)
        print(f"  {name:<12} {p}  {'ok' if os.path.isdir(p) else 'MISSING'}")

    print("\nconfig")
    print(f"  seed         {config.SEED}")
    print(f"  specs        {list(config.WINDOW_SPECS)}  (primary: {config.PRIMARY_SPEC})")
    print(f"  classes      {config.CLASSES}")
    print(f"  features     {config.N_FEATURES}")
    print(f"  folds        {config.N_FOLDS}")

    print("\n" + "-" * 68)
    for w in warnings:
        print(f"WARN   {w}")
    for p in problems:
        print(f"ERROR  {p}")

    if problems:
        print("\nFIX THE ERRORS ABOVE BEFORE CONTINUING.")
        return 1
    print("\nenvironment ok. next: python steps/step02_fetch_bidmc.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
