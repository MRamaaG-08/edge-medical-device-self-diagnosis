"""
BUT PPG loader, for the synthetic-to-real transfer test.

Nemcova et al., Brno University of Technology Smartphone PPG Database (BUT PPG),
PhysioNet. Short smartphone-camera PPG records with a HUMAN-ANNOTATED quality
label per record. That annotation is the reason this dataset is worth the
trouble: it lets a model trained entirely on synthetic degradation be scored
against real recordings that a human called good or bad, with no hardware.

Two caveats that must appear in the paper, not just in this docstring:

  1. SAMPLING RATE CONFOUND. BUT PPG is recorded at a much lower rate than
     BIDMC. Records are resampled up to the working rate, which adds no
     information and changes the spectral tail. Any transfer gap is therefore
     an upper bound on the true gap, not a clean measurement of it.
  2. LABEL SEMANTICS DIFFER. The annotation is "is this record usable", not
     "which of four faults is present". The transfer test is therefore binary,
     clean against degraded, and cannot evaluate the five-class problem.

The directory layout is read at runtime rather than hard coded, because the
release layout has changed between versions. Run

    python steps/step07_butppg_transfer.py --inspect

first; it prints what it found and stops.
"""
from __future__ import annotations
import os
import csv
import glob
import numpy as np

import config

QUALITY_COL_HINTS = ["quality", "qual", "annotation", "label", "good"]
ID_COL_HINTS = ["id", "record", "name", "file", "subject"]


def inspect(root: str | None = None) -> dict:
    """Describe what is actually on disk so the loader can be adapted if needed."""
    root = root or config.BUTPPG_DIR
    info = {"root": root, "exists": os.path.isdir(root)}
    if not info["exists"]:
        return info

    heas = glob.glob(os.path.join(root, "**", "*.hea"), recursive=True)
    csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    info["n_hea"] = len(heas)
    info["sample_hea"] = [os.path.relpath(h, root) for h in sorted(heas)[:8]]
    info["csv_files"] = [os.path.relpath(c, root) for c in sorted(csvs)[:20]]

    for c in sorted(csvs):
        try:
            with open(c, newline="", encoding="utf-8-sig") as fh:
                header = next(csv.reader(fh))
            info.setdefault("csv_headers", {})[os.path.relpath(c, root)] = header
        except Exception as e:
            info.setdefault("csv_headers", {})[os.path.relpath(c, root)] = f"<unreadable: {e}>"
    return info


def _pick(header, hints):
    low = [h.strip().lower() for h in header]
    for hint in hints:
        for i, h in enumerate(low):
            if hint in h:
                return i
    return None


def load_quality_table(root: str | None = None) -> dict:
    """
    Return {record_id: 1 if degraded else 0}.

    The annotation convention in the published table is 1 = good quality, so it
    is inverted here to match this project's label convention where 1 means
    degraded.
    """
    root = root or config.BUTPPG_DIR
    for path in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        try:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.reader(fh))
        except Exception:
            continue
        if len(rows) < 2:
            continue
        header = rows[0]
        qi = _pick(header, QUALITY_COL_HINTS)
        ii = _pick(header, ID_COL_HINTS)
        if qi is None or ii is None:
            continue

        table = {}
        for r in rows[1:]:
            if len(r) <= max(qi, ii):
                continue
            rid = r[ii].strip()
            raw = r[qi].strip().lower()
            if raw in ("1", "1.0", "good", "true", "yes"):
                good = 1
            elif raw in ("0", "0.0", "bad", "false", "no"):
                good = 0
            else:
                continue
            table[rid] = 1 - good          # 1 == degraded
        if table:
            print(f"  quality table: {os.path.relpath(path, root)} "
                  f"({len(table)} records, id col '{header[ii]}', quality col '{header[qi]}')")
            return table

    raise FileNotFoundError(
        "no usable quality annotation CSV found under "
        f"{root}. Run step07 with --inspect and report the layout.")


def load_records(root: str | None = None, limit: int | None = None):
    """
    Yield (record_id, raw_signal, fs) for every PPG record found.

    Record id is the .hea basename with any channel suffix stripped, so it can
    be matched against the annotation table.
    """
    import wfdb
    root = root or config.BUTPPG_DIR
    heas = sorted(glob.glob(os.path.join(root, "**", "*.hea"), recursive=True))
    if not heas:
        raise FileNotFoundError(f"no WFDB headers under {root}")

    n = 0
    for h in heas:
        base = os.path.splitext(h)[0]
        name = os.path.basename(base)
        if name.lower().endswith(("_acc", "_ecg")):
            continue
        try:
            rec = wfdb.rdrecord(base)
        except Exception:
            continue
        sig_names = [s.strip().lower() for s in (rec.sig_name or [])]
        ch = None
        for i, s in enumerate(sig_names):
            if "ppg" in s or "pleth" in s:
                ch = i
                break
        if ch is None:
            if rec.p_signal is None or rec.p_signal.shape[1] != 1:
                continue
            ch = 0

        x = np.asarray(rec.p_signal[:, ch], dtype=np.float64)
        if np.isnan(x).any():
            idx = np.arange(x.size)
            good = ~np.isnan(x)
            if good.sum() < 2:
                continue
            x = np.interp(idx, idx[good], x[good])

        rid = name
        for suffix in ("_ppg", "_PPG"):
            if rid.lower().endswith("_ppg"):
                rid = rid[: -len(suffix)]
                break

        yield rid, x, int(rec.fs)
        n += 1
        if limit and n >= limit:
            return
