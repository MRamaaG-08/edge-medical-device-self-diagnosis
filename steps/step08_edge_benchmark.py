#!/usr/bin/env python3
"""
Step 08 — edge feasibility, benchmarked honestly.

Phase 1 reported 0.108 ms measured on a laptop and 0.4 ms "estimated" for a
Raspberry Pi 4 via a slowdown factor. Reviewer 12 called out the gap between
the benchmark numbers and the low-latency claims. No hardware measurement is
possible in Phase 1 by scope, so this step does two things instead:

  1. Reports the host benchmark with the host fully specified, and drops the
     extrapolated Pi figure entirely. An unmeasured number should not be in a
     results table, even labelled. It belongs in future work.

  2. Adds a model-size sweep, which IS a real result and needs no hardware.
     Accuracy against serialised model size across a grid of forest sizes and
     depths gives the deployment argument something quantitative to stand on,
     and it directly feeds the Phase 2 compression work: the ESP32 budget is
     roughly 150 KB and the Phase 1 forest is about 1 MB, so the sweep answers
     what that budget actually costs in accuracy.

    python steps/step08_edge_benchmark.py
"""
import sys
import os
import json
import time
import platform
import argparse
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import splits, metrics
from steps.step04_build_dataset import dataset_path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

SWEEP_TREES = [5, 10, 20, 30, 50, 100]
SWEEP_DEPTH = [4, 6, 8, 10, None]


def host_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return info


def onnx_benchmark(model, scaler, X_sample, n_runs: int = 2000) -> dict:
    """Export to ONNX and time single-window inference, one window at a time."""
    try:
        from skl2onnx import to_onnx
        import onnxruntime as ort
    except ImportError as e:
        return {"skipped": f"{e}. pip install skl2onnx onnxruntime"}

    Xs = scaler.transform(X_sample).astype(np.float32)
    onx = to_onnx(model, Xs[:1], target_opset=15)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fh:
        fh.write(onx.SerializeToString())
        path = fh.name
    size_kb = os.path.getsize(path) / 1024.0

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1          # single-thread, the edge-relevant case
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    one = Xs[:1]
    for _ in range(200):                 # warm up
        sess.run(None, {inp: one})

    times = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        w = Xs[i % Xs.shape[0]: i % Xs.shape[0] + 1]
        t0 = time.perf_counter()
        sess.run(None, {inp: w})
        times[i] = (time.perf_counter() - t0) * 1000.0

    os.unlink(path)
    return {
        "model_size_kb": round(size_kb, 1),
        "n_runs": n_runs,
        "threads": 1,
        "mean_latency_ms": round(float(times.mean()), 4),
        "median_latency_ms": round(float(np.median(times)), 4),
        "p95_latency_ms": round(float(np.percentile(times, 95)), 4),
        "p99_latency_ms": round(float(np.percentile(times, 99)), 4),
        "max_latency_ms": round(float(times.max()), 4),
        "std_latency_ms": round(float(times.std(ddof=1)), 4),
        "note": ("Measured on the host named in host_info. No Raspberry Pi, "
                 "ESP32 or other target measurement is claimed; on-target "
                 "measurement is Phase 2 work."),
    }


def size_sweep(X, y, groups, rng) -> list:
    """Accuracy against serialised size, on record-level folds."""
    try:
        import joblib
    except ImportError:
        return [{"skipped": "joblib missing"}]

    folds = list(splits.record_level_folds(groups, n_splits=min(5, config.N_FOLDS)))
    rows = []
    for n_trees in SWEEP_TREES:
        for depth in SWEEP_DEPTH:
            accs, f1s, sizes, nodes = [], [], [], []
            for tr, te in folds:
                scaler = StandardScaler().fit(X[tr])
                params = dict(config.RF_PARAMS)
                params.update(n_estimators=n_trees, max_depth=depth)
                rf = RandomForestClassifier(**params)
                rf.fit(scaler.transform(X[tr]), y[tr])
                pred = rf.predict(scaler.transform(X[te]))
                s = metrics.score(y[te], pred)
                accs.append(s["accuracy"])
                f1s.append(s["macro_f1"])
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as fh:
                    joblib.dump(rf, fh.name, compress=0)
                    sizes.append(os.path.getsize(fh.name) / 1024.0)
                os.unlink(fh.name)
                nodes.append(sum(t.tree_.node_count for t in rf.estimators_))
            rows.append({
                "n_estimators": n_trees,
                "max_depth": depth,
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "accuracy_std": round(float(np.std(accs, ddof=1)), 4),
                "macro_f1_mean": round(float(np.mean(f1s)), 4),
                "pickle_kb_mean": round(float(np.mean(sizes)), 1),
                "total_nodes": int(round(np.mean(nodes))),
                "embedded_kb_est": round(float(np.mean(nodes)) * config.BYTES_PER_NODE / 1024.0, 1),
            })
            print(f"    trees={n_trees:>3}  depth={str(depth):>4}  "
                  f"acc {rows[-1]['accuracy_mean'] * 100:5.2f}%  "
                  f"{rows[-1]['total_nodes']:>8,} nodes  ~{rows[-1]['embedded_kb_est']:>7.1f} KB embedded")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=config.PRIMARY_SPEC, choices=list(config.WINDOW_SPECS))
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--runs", type=int, default=2000)
    args = ap.parse_args()

    print("=" * 68)
    print("STEP 08  edge feasibility benchmark")
    print("=" * 68)

    p = dataset_path(args.spec)
    if not os.path.exists(p):
        print(f"ERROR: {p} missing. Run step04 first.")
        return 1
    d = np.load(p, allow_pickle=True)
    X, y, groups = d["X"], d["y"].astype(int), d["groups"]
    rng = np.random.default_rng(config.SEED)

    hi = host_info()
    print("\n  host")
    for k, v in hi.items():
        print(f"    {k:<12} {v}")

    tr, te, _ = splits.record_level_holdout(groups)
    scaler = StandardScaler().fit(X[tr])
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(scaler.transform(X[tr]), y[tr])
    acc = metrics.score(y[te], rf.predict(scaler.transform(X[te])))

    print("\n  reference model (record-level holdout)")
    print(f"    accuracy {acc['accuracy'] * 100:.2f}%   macroF1 {acc['macro_f1']:.4f}")
    print(f"    total tree nodes: {sum(t.tree_.node_count for t in rf.estimators_):,}")

    print("\n  ONNX single-window inference")
    bench = onnx_benchmark(rf, scaler, X[te][:256], n_runs=args.runs)
    for k, v in bench.items():
        print(f"    {k:<20} {v}")

    sweep = []
    if not args.skip_sweep:
        print("\n  accuracy against model size")
        sweep = size_sweep(X, y, groups, rng)

    out = {
        "spec": args.spec,
        "host": hi,
        "reference_model": {
            "params": {k: str(v) for k, v in config.RF_PARAMS.items()},
            "accuracy": acc["accuracy"],
            "macro_f1": acc["macro_f1"],
            "total_nodes": int(sum(t.tree_.node_count for t in rf.estimators_)),
            "embedded_kb_est": round(sum(t.tree_.node_count for t in rf.estimators_)
                                     * config.BYTES_PER_NODE / 1024.0, 1),
        },
        "size_method": (f"embedded_kb_est = total tree nodes x {config.BYTES_PER_NODE} bytes "
                        "(compact C array). pickle_kb_mean is the Python joblib file and "
                        "is not an embedded footprint."),
        "onnx_benchmark": bench,
        "size_sweep": sweep,
        "disclaimer": (
            "All timings are host measurements. Phase 1 makes no claim about "
            "latency on Raspberry Pi, ESP32 or any other embedded target. The "
            "extrapolated figure reported in the earlier draft has been removed."),
    }
    path = os.path.join(config.RESULTS, "step08_edge_benchmark.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written: {path}")
    print("\nnext: python steps/step09_ablation_importance.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
