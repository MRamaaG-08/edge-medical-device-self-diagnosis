#!/usr/bin/env python3
"""
Run the whole Phase 1 rebuild in order.

    python run_all.py                  # full run on real BIDMC data
    python run_all.py --surrogate      # smoke test, results NOT reportable
    python run_all.py --from 5         # resume from step 5
    python run_all.py --skip-transfer  # if BUT PPG is not downloaded yet

Step 03 is a gate. If it fails the run stops, because every number downstream
of a broken pipeline order is wrong in a way that looks fine.
"""
import sys
import os
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

STEPS = [
    (1, "step01_env_check.py", "environment check", []),
    (2, "step02_fetch_bidmc.py", "fetch BIDMC", []),
    (3, "step03_order_audit.py", "pipeline order audit [GATE]", []),
    (4, "step04_build_dataset.py", "build corpus", []),
    (5, "step05_train_eval.py", "train and evaluate", []),
    (6, "step06_sqi_baselines.py", "published SQI baselines", []),
    (7, "step07_butppg_transfer.py", "synthetic-to-real transfer", []),
    (8, "step08_edge_benchmark.py", "edge benchmark", []),
    (9, "step09_ablation_importance.py", "ablation and importance", ["--extended-drift"]),
    (11, "step11_calibration_abstention.py", "calibration and selective prediction", []),
    (10, "step10_figures.py", "figures", []),
]

GATES = {3}
SURROGATE_ONLY = {2: "skip", 7: "skip"}      # no real data to fetch or transfer to


def run(step_no, script, title, extra, surrogate, skip_transfer):
    if surrogate and SURROGATE_ONLY.get(step_no) == "skip":
        print(f"\n### step {step_no:>2}  {title}  [skipped in surrogate mode]")
        return 0
    if skip_transfer and step_no == 7:
        print(f"\n### step {step_no:>2}  {title}  [skipped by --skip-transfer]")
        return 0

    args = [sys.executable, os.path.join(ROOT, "steps", script)] + list(extra)
    if surrogate and step_no == 4:
        args += ["--surrogate", "--records", "10"]

    print(f"\n{'=' * 72}\n### step {step_no:>2}  {title}\n{'=' * 72}")
    t0 = time.time()
    rc = subprocess.call(args, cwd=ROOT)
    print(f"--- step {step_no} finished rc={rc} in {time.time() - t0:.0f}s")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="stage position (1-11), not the step number in the filename")
    ap.add_argument("--to", dest="end", type=int, default=len(STEPS))
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    print("Phase 1 rebuild")
    print("Edge-Based Self-Diagnosis of PPG Sensors Using")
    print("Simulation-Augmented Signal Quality Classification")
    if args.surrogate:
        print("\n*** SURROGATE MODE. Nothing produced here may go in the paper. ***")

    t0 = time.time()
    failures = []
    # --from / --to refer to STAGE position in the list above, not to the step
    # number in the filename. Figures run last even though they are step10.
    for stage, (no, script, title, extra) in enumerate(STEPS, start=1):
        if stage < args.start or stage > args.end:
            continue
        rc = run(no, script, title, extra, args.surrogate, args.skip_transfer)
        if rc != 0:
            failures.append((no, title))
            if no in GATES:
                print(f"\nGATE {no} FAILED. Stopping. Fix this before continuing.")
                return 1
            if not args.continue_on_error:
                print(f"\nStep {no} failed. Re-run with --continue-on-error to push past it.")
                return 1

    print(f"\n{'=' * 72}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    if failures:
        print("failed steps: " + ", ".join(f"{n} ({t})" for n, t in failures))
    print(f"results : {os.path.join(ROOT, 'results')}")
    print(f"figures : {os.path.join(ROOT, 'results', 'figures')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
