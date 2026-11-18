#!/usr/bin/env python3
"""
Step 02 — fetch and cache all 53 BIDMC records.

Phase 1 used five. Five records cannot support record-level cross-validation in
any meaningful way, which is part of why the window-level split happened in the
first place. With 53 the headline protocol becomes 10-fold grouped CV over
distinct subjects.

Downloads stream from PhysioNet through wfdb and are cached as .npy, so this is
a one-time cost. Re-running is free.

    python steps/step02_fetch_bidmc.py
    python steps/step02_fetch_bidmc.py --records bidmc01 bidmc02    # subset
    python steps/step02_fetch_bidmc.py --offline                    # cache only
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import io_bidmc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="*", default=None)
    ap.add_argument("--offline", action="store_true",
                    help="use cached .npy only, never contact PhysioNet")
    args = ap.parse_args()

    records = args.records or config.BIDMC_RECORDS
    print("=" * 68)
    print(f"STEP 02  fetching {len(records)} BIDMC records")
    print("=" * 68)
    print(f"cache: {config.BIDMC_DIR}\n")

    summary, total_s = {}, 0.0
    for rid, x in io_bidmc.load_all(records, allow_download=not args.offline):
        dur = x.size / config.BIDMC_FS_NATIVE
        total_s += dur
        summary[rid] = {
            "n_samples": int(x.size),
            "duration_s": round(dur, 2),
            "fs_native": config.BIDMC_FS_NATIVE,
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    out = os.path.join(config.RESULTS, "bidmc_inventory.json")
    with open(out, "w") as fh:
        json.dump({"n_records": len(summary),
                   "total_duration_min": round(total_s / 60.0, 1),
                   "records": summary}, fh, indent=2)

    print(f"\n  records cached : {len(summary)}")
    print(f"  total duration : {total_s / 60.0:.1f} min")
    print(f"  inventory      : {out}")

    if len(summary) < 2:
        print("\nERROR: fewer than 2 records. Record-level CV is impossible.")
        print("Check network access to physionet.org, or download the zip from")
        print("  https://physionet.org/content/bidmc/1.0.0/")
        print(f"and unpack the .hea/.dat files into {config.BIDMC_DIR}")
        return 1

    if len(summary) < 10:
        print(f"\nWARN: only {len(summary)} records. config.N_FOLDS will be "
              f"capped to {len(summary)}.")

    print("\nnext: python steps/step03_order_audit.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
