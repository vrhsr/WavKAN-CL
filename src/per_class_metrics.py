#!/usr/bin/env python
"""
per_class_metrics.py -- per-class precision, recall and F1 for every arm, over
all seeds, computed from the saved per-seed prediction arrays.

Why this exists
---------------
The manuscript originally reported only per-class *recall* and Macro-F1.  That
is not enough to place a minority-class recall on the sensitivity/precision
trade-off, and it is not the reporting convention for the AAMI EC57 /
de Chazal inter-patient protocol, which pairs sensitivity with positive
predictivity.  Without precision, an S-recall of 0.22 cannot be compared with
any published SVEB sensitivity, because the published figures sit at declared
operating points (de Chazal et al. 2004: SVEB sensitivity 0.759 at a positive
predictivity of 0.385).

No retraining is involved: every arm already stores `predictions`/`true` (or
`test_predictions`/`test_true`) arrays per seed, so this is a pure recomputation
from artifacts that are already on disk.

Output
------
results/per_class_metrics.json -- per-arm, per-class mean/std over seeds, plus
the raw per-seed values so a later statistical test can be paired by seed
identity rather than by position.

Usage
-----
    python src/per_class_metrics.py
    python src/per_class_metrics.py --arm ablation_no_rr_attn
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# UTF-8 console guard (AUDIT_FINDINGS.md C19/H30/H48: this bug class recurs).
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CLASSES = ["N", "S", "V", "F", "Q"]

# Arm -> display name.  Order matches the manuscript's baseline table.
ARMS = {
    "ablation_no_rr_attn": "PC-WavKAN",
    "baseline_resnet1d": "ResNet1D",
    "baseline_transformer": "Transformer",
    "baseline_cnn_focal": "CNN+Focal",
    "baseline_bspline_kan": "B-Spline KAN",
}

# Different training scripts used different filenames for the same thing.
PRED_NAMES = ("test_predictions.npy", "predictions.npy")
TRUE_NAMES = ("test_true.npy", "true.npy")


def _find(seed_dir: str, names) -> str | None:
    for n in names:
        p = os.path.join(seed_dir, n)
        if os.path.exists(p):
            return p
    return None


def per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, recall and F1 per class, computed explicitly.

    Written out rather than delegated to sklearn so that the zero-denominator
    convention is visible: a class the model never predicts has undefined
    precision, and we record it as NaN rather than silently as 0.0.  Averaging
    a spurious 0.0 into a macro statistic would understate precision for the
    classes that are actually the point of this table.
    """
    out = {}
    for idx, c in enumerate(CLASSES):
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / support if support > 0 else float("nan")
        if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
            f1 = 0.0 if support > 0 else float("nan")
        else:
            f1 = 2 * prec * rec / (prec + rec)
        out[c] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": support, "tp": tp, "fp": fp, "fn": fn}
    return out


def collect(arm_dir: str) -> dict:
    """Per-seed per-class metrics for one arm, keyed by seed identity."""
    per_seed = {}
    for sd in sorted(glob.glob(os.path.join(arm_dir, "seed_*"))):
        pf, tf = _find(sd, PRED_NAMES), _find(sd, TRUE_NAMES)
        if pf is None or tf is None:
            continue
        y_pred, y_true = np.load(pf), np.load(tf)
        if y_pred.shape != y_true.shape:
            print(f"  !! shape mismatch in {sd}: {y_pred.shape} vs {y_true.shape}")
            continue
        per_seed[os.path.basename(sd).replace("seed_", "")] = per_class(y_true, y_pred)
    return per_seed


def aggregate(per_seed: dict) -> dict:
    """Mean/std across seeds, NaN-aware (a class never predicted by some seeds
    contributes no precision value for those seeds rather than a zero)."""
    agg = {}
    for c in CLASSES:
        for metric in ("precision", "recall", "f1"):
            vals = np.array([s[c][metric] for s in per_seed.values()], dtype=float)
            finite = vals[np.isfinite(vals)]
            agg.setdefault(c, {})[metric] = {
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "n_defined": int(finite.size),
                "n_seeds": int(vals.size),
            }
        agg[c]["support"] = int(np.median([s[c]["support"] for s in per_seed.values()]))
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, help="only this arm (directory name)")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output", default="results/per_class_metrics.json")
    a = ap.parse_args()

    arms = {a.arm: ARMS.get(a.arm, a.arm)} if a.arm else ARMS
    report = {"classes": CLASSES, "arms": {}}

    for d, name in arms.items():
        arm_dir = os.path.join(a.results_root, d)
        if not os.path.isdir(arm_dir):
            print(f"  skip {name}: {arm_dir} not found")
            continue
        ps = collect(arm_dir)
        if not ps:
            print(f"  skip {name}: no per-seed prediction arrays")
            continue
        report["arms"][name] = {"dir": d, "n_seeds": len(ps),
                                "seeds": sorted(ps, key=int),
                                "aggregate": aggregate(ps),
                                "per_seed": ps}
        print(f"\n{name}  ({len(ps)} seeds, {arm_dir})")
        print(f"  {'class':6s} {'support':>8s} {'precision':>16s} {'recall':>16s} {'F1':>16s}")
        for c in CLASSES:
            g = report["arms"][name]["aggregate"][c]
            def fmt(m):
                v = g[m]
                if not np.isfinite(v["mean"]):
                    return "undefined".rjust(16)
                note = "" if v["n_defined"] == v["n_seeds"] else f" [{v['n_defined']}/{v['n_seeds']}]"
                return f"{v['mean']:.4f}+/-{v['std']:.4f}{note}".rjust(16)
            print(f"  {c:6s} {g['support']:8d} {fmt('precision')} {fmt('recall')} {fmt('f1')}")

    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
