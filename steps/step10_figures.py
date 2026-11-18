#!/usr/bin/env python3
"""
Step 10 — figures.

Reads whatever result files exist and draws what it can, skipping the rest, so
it is safe to run at any point. Everything is sized for an IEEE two-column
page: 3.4 in single column, 7.0 in double, serif type, no colour needed to read
the result.

    python steps/step10_figures.py
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
})

COL1, COL2 = 3.4, 7.0
GREY = "#4a4a4a"
ACCENT = "#1f4e79"
WARN = "#9c2b2b"


def _load(name):
    p = os.path.join(config.RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _save(fig, name):
    p = os.path.join(config.FIGS, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"    {name}")
    return p


# --------------------------------------------------------------- fig 1
def fig_pipeline():
    """Data flow with the ordering constraint made explicit on the diagram."""
    fig, ax = plt.subplots(figsize=(COL1, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 25); ax.axis("off"); ax.grid(False)

    boxes = [
        ("BIDMC PPG\n53 records, 125 Hz", "resampled"),
        ("Resample\nto working rate", "continuous signal"),
        ("Butterworth band-pass\n0.5-4.0 Hz, order 2", "band-limited"),
        ("Z-score normalisation", "unit variance"),
        ("Windowing\noverlapping frames", "clean windows"),
        ("Degradation injection\n4 faults x 3 severities", "labelled windows"),
        ("Feature extraction\n8 time + 5 frequency", "13-D vectors"),
        ("Record-level split\nGroupKFold", "no subject shared"),
        ("Threshold / SVM / RF\n+ 3 published SQIs", None),
    ]
    h, gap = 1.70, 0.72
    y = 24.2
    inject_y = None
    for i, (label, edge) in enumerate(boxes):
        inject = i == 5
        if inject:
            inject_y = y - h / 2
        ax.add_patch(Rectangle((0.6, y - h), 8.8, h,
                               facecolor="#eef2f7" if not inject else "#fdeeee",
                               edgecolor=ACCENT if not inject else WARN,
                               linewidth=1.0, zorder=2))
        ax.text(5.0, y - h / 2, label, ha="center", va="center", fontsize=7,
                zorder=3, color="#111111")
        if edge is not None:
            ax.add_patch(FancyArrowPatch((5.0, y - h), (5.0, y - h - gap),
                                         arrowstyle="-|>", mutation_scale=9,
                                         color=GREY, linewidth=0.8, zorder=1))
            ax.text(5.25, y - h - gap / 2, edge, fontsize=6, color=GREY,
                    ha="left", va="center")
        y -= (h + gap)

    ax.text(5.0, 1.0,
            "highlighted step: degradation is injected after normalisation.\n"
            "The reverse order cancels attenuation exactly.",
            fontsize=6, color=WARN, ha="center", va="bottom", linespacing=1.4)
    return _save(fig, "fig01_pipeline.png")


# --------------------------------------------------------------- fig 2 and 3
def fig_degradation_examples(spec):
    p = os.path.join(config.DATA, f"corpus_{spec}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    keys = list(d["sample_window_keys"])
    wins = d["sample_windows"]
    fs = int(d["fs"])
    want = ["CLEAN"] + [f"{c}_2" for c in config.CLASSES[1:]]
    avail = [(w, keys.index(w)) for w in want if w in keys]
    if not avail:
        return None

    fig, axes = plt.subplots(len(avail), 1, figsize=(COL1, 0.85 * len(avail) + 0.6),
                             sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(wins.shape[1]) / fs
    for ax, (label, idx) in zip(axes, avail):
        ax.plot(t, wins[idx], linewidth=0.6, color=ACCENT)
        nice = label.replace("_2", " (severity 2)").replace("_", " ")
        ax.set_ylabel(nice, fontsize=6, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Synthesized degradation conditions", fontsize=8, y=0.995)
    return _save(fig, "fig02_degradation_examples.png")


def fig_order_audit():
    r = _load("step03_order_audit.json")
    if not r:
        return None
    classes = config.CLASSES[1:]
    correct = [r["separability_correct_order"].get(c, {}).get("mean_std_distance", 0)
               for c in classes]
    wrong = [r["separability_wrong_order"].get(c, {}).get("mean_std_distance", 0)
             for c in classes]

    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    x = np.arange(len(classes)); w = 0.38
    ax.bar(x - w / 2, correct, w, label="inject after normalisation",
           color=ACCENT, edgecolor="none")
    ax.bar(x + w / 2, wrong, w, label="inject before normalisation",
           color=WARN, edgecolor="none")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in classes], rotation=20, ha="right")
    ax.set_ylabel("feature distance from Clean\n(standardised, log scale)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Pipeline ordering determines class separability")
    return _save(fig, "fig03_order_audit.png")


# --------------------------------------------------------------- fig 4
def fig_split_comparison():
    r = _load("step05_train_eval.json")
    if not r:
        return None
    specs = list(r)
    width = COL1 if len(specs) == 1 else COL2
    fig, axes = plt.subplots(1, len(specs), figsize=(width, 2.9), squeeze=False)
    models = ["Threshold", "SVM-RBF", "RandomForest"]

    for ax, spec in zip(axes[0], specs):
        s = r[spec]
        x = np.arange(len(models)); w = 0.36
        for off, proto, colour, lab in (
                (-w / 2, "window_level", WARN, "window-level split (leaky)"),
                (+w / 2, "record_level", ACCENT, "record-level split")):
            if proto not in s:
                continue
            means = [s[proto]["models"][m]["accuracy"]["mean"] * 100 for m in models]
            errs = [s[proto]["models"][m]["accuracy"]["std"] * 100 for m in models]
            ax.bar(x + off, means, w, yerr=errs, capsize=2.5, label=lab,
                   color=colour, edgecolor="none",
                   error_kw=dict(linewidth=0.7, ecolor="#222222"))
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(0, 100)
        lk = s.get("window_level", {}).get("leakage", {}).get(
            "mean_pct_test_with_sibling_in_train")
        sub = spec
        if lk is not None:
            sub += f"\n{lk:.0f}% of leaky test windows have a training sibling"
        ax.set_title(sub, fontsize=6.5)
    axes[0][0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.suptitle("Reported accuracy depends on how the data is split",
                 fontsize=8.5, y=1.06)
    return _save(fig, "fig04_split_protocol.png")


# --------------------------------------------------------------- fig 5
def fig_confusion():
    r = _load("step05_train_eval.json")
    if not r:
        return None
    spec = config.PRIMARY_SPEC if config.PRIMARY_SPEC in r else list(r)[0]
    h = r[spec].get("holdout")
    if not h:
        return None

    panels = [("record-level split", h["record_level_holdout"]),
              ("window-level split", h["window_level_holdout_phase1_reproduction"])]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0))
    for ax, (title, blk) in zip(axes, panels):
        cm = np.array(blk["confusion"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(config.CLASSES)))
        ax.set_yticks(range(len(config.CLASSES)))
        ax.set_xticklabels([c.title() for c in config.CLASSES], rotation=45, ha="right")
        ax.set_yticklabels([c.title() for c in config.CLASSES])
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if norm[i, j] > 0.55 else "#222222")
        ax.set_title(f"{title}\nacc {blk['accuracy'] * 100:.2f}%, "
                     f"macro F1 {blk['macro_f1']:.3f}", fontsize=7)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("true")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalised rate")
    return _save(fig, "fig05_confusion.png")


# --------------------------------------------------------------- fig 6
def fig_feature_space(spec):
    p = os.path.join(config.DATA, f"corpus_{spec}.npz")
    if not os.path.exists(p):
        return None
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    d = np.load(p, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(X.shape[0], min(6000, X.shape[0]), replace=False)
    Z = PCA(n_components=2, random_state=config.SEED).fit_transform(
        StandardScaler().fit_transform(X[idx]))

    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    markers = ["o", "s", "^", "D", "v"]
    for ci, cname in enumerate(config.CLASSES):
        m = y[idx] == ci
        if not m.any():
            continue
        ax.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.45, marker=markers[ci % 5],
                   label=cname.title(), linewidths=0)
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.legend(frameon=False, markerscale=3, loc="best")
    ax.set_title("Feature space, first two principal components")
    return _save(fig, "fig06_feature_space.png")


# --------------------------------------------------------------- fig 7
def fig_importance():
    r = _load("step09_ablation_importance.json")
    if not r or "importance" not in r:
        return None
    imp = r["importance"]
    names = [n for n in config.FEATURE_NAMES if n in imp]
    gini = np.array([imp[n]["gini_mean"] for n in names])
    perm = np.array([imp[n]["permutation_mean"] for n in names])
    order = np.argsort(gini)
    names = [names[i] for i in order]
    gini, perm = gini[order], perm[order]

    g = gini / (gini.max() or 1)
    pz = perm / (np.abs(perm).max() or 1)

    fig, ax = plt.subplots(figsize=(COL1, 3.1))
    yy = np.arange(len(names)); h = 0.38
    ax.barh(yy - h / 2, g, h, label="Gini (normalised)", color=ACCENT, edgecolor="none")
    ax.barh(yy + h / 2, pz, h, label="permutation (normalised)",
            color="#8aa8c4", edgecolor="none")
    ax.set_yticks(yy)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=6)
    ax.set_xlabel("relative importance")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Feature importance by two measures")
    return _save(fig, "fig07_importance.png")


# --------------------------------------------------------------- fig 8
def fig_size_sweep():
    r = _load("step08_edge_benchmark.json")
    if not r or not r.get("size_sweep"):
        return None
    rows = [x for x in r["size_sweep"] if "total_nodes" in x]
    if not rows:
        return None
    for x in rows:   # older JSONs only have total_nodes; derive the estimate
        x.setdefault("embedded_kb_est", x["total_nodes"] * config.BYTES_PER_NODE / 1024.0)

    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    depths = sorted({str(x["max_depth"]) for x in rows},
                    key=lambda s: (s == "None", int(s) if s != "None" else 0))
    for dep in depths:
        sel = [x for x in rows if str(x["max_depth"]) == dep]
        sel.sort(key=lambda x: x["embedded_kb_est"])
        dep = "unlimited" if dep == "None" else dep
        ax.plot([x["embedded_kb_est"] for x in sel],
                [x["accuracy_mean"] * 100 for x in sel],
                marker="o", markersize=3, linewidth=0.9, label=f"depth {dep}")
    ax.set_xscale("log")
    ax.set_xlabel(f"estimated embedded size (KB, {config.BYTES_PER_NODE} B/node, log)")
    ax.set_ylabel("accuracy (%)")
    ax.axvline(150, color=WARN, linestyle="--", linewidth=0.8)
    ax.text(160, ax.get_ylim()[1] - 0.3, "~150 KB\nESP32 budget", fontsize=6,
            color=WARN, va="top")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Accuracy against model size")
    return _save(fig, "fig08_size_sweep.png")


# --------------------------------------------------------------- fig 9
def fig_severity():
    r = _load("step09_ablation_importance.json")
    if not r or not r.get("per_severity"):
        return None
    rows = {k: v for k, v in r["per_severity"].items()
            if v.get("median_sdr_db") is not None}
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    colours = {"NOISY": ACCENT, "DRIFTED": WARN,
               "ATTENUATED": "#2e7d52", "SPIKED": "#7a5195"}
    seen = set()
    for key, v in rows.items():
        cname = key.rsplit("_L", 1)[0]
        lab = cname.title() if cname not in seen else None
        seen.add(cname)
        ax.scatter(v["median_sdr_db"], v["recall"] * 100, s=26,
                   color=colours.get(cname, GREY), label=lab, zorder=3)
        ax.annotate(f"L{v['severity']}", (v["median_sdr_db"], v["recall"] * 100),
                    textcoords="offset points", xytext=(4, 3), fontsize=5.5)
    ax.set_xlabel("median signal-to-distortion ratio (dB)")
    ax.set_ylabel("per-class recall (%)")
    ax.set_ylim(-5, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(frameon=False, fontsize=6, loc="center right")
    ax.set_title("Detectability tracks distortion, not fault type", pad=6)
    return _save(fig, "fig09_severity_vs_sdr.png")


# --------------------------------------------------------------- fig 10
def fig_sqi():
    r = _load("step06_sqi_baselines.json")
    if not r:
        return None
    spec = config.PRIMARY_SPEC if config.PRIMARY_SPEC in r else list(r)[0]
    s = r[spec]
    names = [n for n in ["S_SQI", "R_SQI", "T_SQI", "RandomForest-13feat"] if n in s]
    vals = [s[n]["balanced_accuracy"]["mean"] * 100 for n in names]
    errs = [s[n]["balanced_accuracy"]["std"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    colours = [GREY] * (len(names) - 1) + [ACCENT]
    ax.bar(range(len(names)), vals, 0.6, yerr=errs, capsize=3,
           color=colours, edgecolor="none",
           error_kw=dict(linewidth=0.7, ecolor="#222222"))
    ax.axhline(50, color=WARN, linestyle="--", linewidth=0.8)
    ax.text(len(names) - 0.4, 51.5, "chance", fontsize=6, color=WARN, ha="right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("-13feat", "\n(13 feat)") for n in names],
                       fontsize=6.5)
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Published SQIs against the learned model\n"
                 "(clean vs degraded, identical record-level folds)", fontsize=7.5)
    return _save(fig, "fig10_sqi_baselines.png")


# --------------------------------------------------------------- fig 11
def fig_transfer():
    r = _load("step07_butppg_transfer.json")
    if not r:
        return None
    labels = ["BIDMC\n(in-domain)", "BUT PPG\n(window)", "BUT PPG\n(record)"]
    vals = [r["in_domain_reference"]["balanced_accuracy"] * 100,
            r["transfer_window_level"]["balanced_accuracy"] * 100,
            r["transfer_record_level"]["balanced_accuracy"] * 100]

    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    ax.bar(range(3), vals, 0.55, color=[ACCENT, "#8aa8c4", "#8aa8c4"],
           edgecolor="none")
    ax.axhline(50, color=WARN, linestyle="--", linewidth=0.8)
    ax.text(2.4, 51.5, "chance", fontsize=6, color=WARN, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Synthetic-to-real transfer, no target-domain training",
                 fontsize=7.5)
    return _save(fig, "fig11_transfer.png")


# --------------------------------------------------------------- fig 12
def fig_reliability():
    r = _load("step11_calibration_abstention.json")
    if not r or "calibration" not in r:
        return None
    cal = r["calibration"]

    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color=GREY,
            label="perfect calibration")
    styles = {"balanced": (ACCENT, "o", "balanced weights"),
              "unweighted": ("#8aa8c4", "s", "unweighted")}
    for key, (colour, marker, lab) in styles.items():
        if key not in cal:
            continue
        all_bins = cal[key]["reliability_bins"]
        total = sum(b["n"] for b in all_bins) or 1
        # Bins holding under 0.5% of the samples swing wildly and read as
        # structure that is not there. ECE in the legend still uses every bin.
        bins = [b for b in all_bins if b["n"] / total >= 0.005]
        if not bins:
            bins = [b for b in all_bins if b["n"] > 0]
        if not bins:
            continue
        ax.plot([b["mean_confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                marker=marker, markersize=3.2, linewidth=0.9, color=colour,
                label=f"{lab} (ECE {cal[key]['expected_calibration_error']:.3f})")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("Reliability of predicted confidence")
    return _save(fig, "fig12_reliability.png")


# --------------------------------------------------------------- fig 13
def fig_risk_coverage():
    r = _load("step11_calibration_abstention.json")
    if not r or "selective_prediction" not in r:
        return None
    curve = r["selective_prediction"]["curve"]
    if not curve:
        return None
    cov = [c["coverage"] * 100 for c in curve]
    acc = [c["accuracy"] * 100 for c in curve]
    base = r["selective_prediction"]["full_coverage_accuracy"] * 100

    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    ax.plot(cov, acc, marker="o", markersize=3, linewidth=1.0, color=ACCENT)
    ax.axhline(base, color=GREY, linestyle="--", linewidth=0.8)
    ax.text(cov[0], base + 0.6, "answer every window", fontsize=6, color=GREY,
            ha="left", va="bottom")
    ax.invert_xaxis()
    ax.set_xlabel("coverage: windows the model chooses to classify (%)")
    ax.set_ylabel("accuracy on retained windows (%)")
    ax.set_title("Accuracy bought by declining low-confidence windows")
    return _save(fig, "fig13_risk_coverage.png")


# --------------------------------------------------------------- fig 14
def fig_severity_grading():
    r = _load("step11_calibration_abstention.json")
    if not r or "severity_grading" not in r:
        return None
    g = {k: v for k, v in r["severity_grading"].items() if not k.startswith("_")}
    if not g:
        return None
    names = list(g)
    mae = [g[n]["mae_levels"] for n in names]
    err = [g[n].get("mae_std", 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    ax.bar(range(len(names)), mae, 0.55, yerr=err, capsize=3,
           color=ACCENT, edgecolor="none",
           error_kw=dict(linewidth=0.7, ecolor="#222222"))
    for i, n in enumerate(names):
        ax.text(i, mae[i] + max(err[i], 0.01) + 0.02,
                f"{g[n]['exact_level_accuracy'] * 100:.0f}% exact",
                ha="center", fontsize=5.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.title() for n in names], rotation=15, ha="right")
    ax.set_ylabel("mean absolute error (severity levels)")
    ax.set_title("Severity grading within each fault class")
    return _save(fig, "fig14_severity_grading.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=config.PRIMARY_SPEC, choices=list(config.WINDOW_SPECS))
    args = ap.parse_args()

    print("=" * 68)
    print("STEP 10  figures")
    print("=" * 68)
    print(f"  -> {config.FIGS}\n")

    made, skipped = [], []
    jobs = [
        ("pipeline", lambda: fig_pipeline()),
        ("degradation examples", lambda: fig_degradation_examples(args.spec)),
        ("order audit", lambda: fig_order_audit()),
        ("split protocol", lambda: fig_split_comparison()),
        ("confusion", lambda: fig_confusion()),
        ("feature space", lambda: fig_feature_space(args.spec)),
        ("importance", lambda: fig_importance()),
        ("size sweep", lambda: fig_size_sweep()),
        ("severity vs SDR", lambda: fig_severity()),
        ("SQI baselines", lambda: fig_sqi()),
        ("transfer", lambda: fig_transfer()),
        ("reliability", lambda: fig_reliability()),
        ("risk-coverage", lambda: fig_risk_coverage()),
        ("severity grading", lambda: fig_severity_grading()),
    ]
    for label, fn in jobs:
        try:
            out = fn()
        except Exception as e:
            print(f"    [error] {label}: {type(e).__name__}: {e}")
            skipped.append(label)
            continue
        if out:
            made.append(label)
        else:
            print(f"    [skip]  {label} (inputs not available yet)")
            skipped.append(label)

    print(f"\n  {len(made)} figures written, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
