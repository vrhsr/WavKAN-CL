"""
sensitivity_exclude_202.py -- DS2 five-class Macro-F1 with record 202 removed, for all five models
and 20 seeds, from the predictions each run saved at test time.

Added 2026-10-01 in response to Array reviewer #4 (ARRAY-D-26-02633). Records 201 and 202 of the
MIT-BIH Arrhythmia Database come from the same subject; the de Chazal partition places 201 in DS1
(training) and 202 in DS2. This analysis removes record 202 from DS2 and repeats the primary
comparison (Table tab:baseline_comparison): seed-paired two-sided Wilcoxon, Holm across the four
baselines, t-based 95% CIs, all through the canonical src/paired_stats.py. Forward passes are not
needed; no model is retrained.

Inputs (regenerable; predictions are not git-tracked):
    data/processed_rr_history/{ids_test,y_test}.npy
    results/ablation_no_rr_attn/seed_*/test_predictions.npy, test_true.npy   (PC-WavKAN)
    results/baseline_*/seed_*/predictions.npy, true.npy                       (baselines)
Output:
    results/sensitivity_exclude_202.json  (per-seed values, read by verify_manuscript_numbers.py)

Usage:
    python src/sensitivity_exclude_202.py
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import paired_stats as ps  # noqa: E402

ARMS = {"PC-WavKAN": ("results/ablation_no_rr_attn", "test_predictions.npy", "test_true.npy"),
        "ResNet1D": ("results/baseline_resnet1d", "predictions.npy", "true.npy"),
        "Transformer": ("results/baseline_transformer", "predictions.npy", "true.npy"),
        "CNN+Focal": ("results/baseline_cnn_focal", "predictions.npy", "true.npy"),
        "B-Spline KAN": ("results/baseline_bspline_kan", "predictions.npy", "true.npy")}
RECORD = "202"
OUT = "results/sensitivity_exclude_202.json"


def macro_f1(t, p):
    return float(f1_score(t, p, labels=range(5), average="macro", zero_division=0))


def main():
    ids = np.load("data/processed_rr_history/ids_test.npy", allow_pickle=True).astype(str)
    y = np.load("data/processed_rr_history/y_test.npy")
    keep = ids != RECORD
    out = {"record_removed": RECORD,
           "beats_removed": int((~keep).sum()),
           "beats_removed_by_class": {c: int(((y == i) & ~keep).sum()) for i, c in enumerate("NSVFQ")},
           "beats_total": int(len(y)),
           "note": "Macro-F1 from the predictions saved at test time; each run's label array is checked "
                   "against data/processed_rr_history/y_test.npy before use.",
           "models": {}}
    excl = {}
    for name, (d, pf, tf) in ARMS.items():
        per_seed = {}
        for sd in sorted(glob.glob(os.path.join(d, "seed_*"))):
            t, p = np.load(os.path.join(sd, tf)), np.load(os.path.join(sd, pf))
            if len(t) != len(y) or not (t == y).all():
                raise SystemExit(f"label mismatch in {sd}")
            per_seed[os.path.basename(sd).split("_")[1]] = {"full": macro_f1(t, p),
                                                            "without": macro_f1(t[keep], p[keep])}
        if len(per_seed) != 20:
            raise SystemExit(f"{name}: expected 20 seeds, found {len(per_seed)}")
        out["models"][name] = per_seed
        excl[name] = {s: v["without"] for s, v in per_seed.items()}
    fam = {b: ps.paired_compare(excl["PC-WavKAN"], excl[b], "macro_f1") for b in ARMS if b != "PC-WavKAN"}
    ps.holm_family(fam)
    out["comparisons_without_record"] = {b: {k: r[k] for k in ("mean_diff", "ci_low", "ci_high", "cohens_d", "p_value", "holm_p")
                                             if k in r} for b, r in fam.items()}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved -> {OUT}")
    for name, per_seed in out["models"].items():
        a = np.array([v["full"] for v in per_seed.values()])
        b = np.array([v["without"] for v in per_seed.values()])
        print(f"  {name:13s} full {a.mean():.4f}  without {RECORD} {b.mean():.4f}  change {b.mean() - a.mean():+.4f}")
    for b, r in fam.items():
        print(f"  PC-WavKAN vs {b:12s} diff {r['mean_diff']:+.4f} CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"Holm p {r['holm_p']:.3g}")


if __name__ == "__main__":
    main()
