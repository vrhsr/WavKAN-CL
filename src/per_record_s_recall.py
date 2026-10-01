"""
per_record_s_recall.py -- DS2 S-class recall on record 232 versus the other DS2 records,
for all five models and 20 seeds, from the saved per-seed predictions.

Record 232 holds 1,382 of DS2's 1,837 supraventricular (S) beats. This breakdown was
decided before it was computed (final review, 2026-10-01; AUDIT_FINDINGS.md H84) and is
reported in the manuscript whatever it shows. No training and no re-inference: it reads the
predictions each run saved at test time and the record IDs written by src/process_data.py.

Inputs (all regenerable; predictions are not git-tracked):
    data/processed_rr_history/{ids_test,y_test}.npy
    results/ablation_no_rr_attn/seed_*/test_predictions.npy        (PC-WavKAN)
    results/baseline_*/seed_*/predictions.npy                      (baselines)
Output (git-tracked with -f, read by src/verify_manuscript_numbers.py):
    results/per_record_s_recall.json

Usage:
    python src/per_record_s_recall.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ARMS = {"PC-WavKAN": ("results/ablation_no_rr_attn", "test_predictions.npy", "test_true.npy"),
        "ResNet1D": ("results/baseline_resnet1d", "predictions.npy", "true.npy"),
        "Transformer": ("results/baseline_transformer", "predictions.npy", "true.npy"),
        "CNN+Focal": ("results/baseline_cnn_focal", "predictions.npy", "true.npy"),
        "B-Spline KAN": ("results/baseline_bspline_kan", "predictions.npy", "true.npy")}
S = 1
RECORD = "232"
OUT = "results/per_record_s_recall.json"


def main():
    ids = np.load("data/processed_rr_history/ids_test.npy", allow_pickle=True).astype(str)
    y = np.load("data/processed_rr_history/y_test.npy")
    in_rec = (ids == RECORD) & (y == S)
    other = (ids != RECORD) & (y == S)
    out = {"record": RECORD,
           "s_beats_record": int(in_rec.sum()),
           "s_beats_other_records": int(other.sum()),
           "other_records_with_s_beats": int(len(set(ids[other]))),
           "note": "S-recall from the predictions saved at test time; label arrays are checked "
                   "against data/processed_rr_history/y_test.npy before use.",
           "models": {}}
    for name, (d, pf, tf) in ARMS.items():
        per_seed = {}
        for sd in sorted(glob.glob(os.path.join(d, "seed_*"))):
            t = np.load(os.path.join(sd, tf))
            if not np.array_equal(t, y):
                raise SystemExit("label order mismatch in %s" % sd)
            p = np.load(os.path.join(sd, pf))
            per_seed[os.path.basename(sd)] = {"record": float((p[in_rec] == S).mean()),
                                              "other": float((p[other] == S).mean())}
        r = np.array([v["record"] for v in per_seed.values()])
        o = np.array([v["other"] for v in per_seed.values()])
        out["models"][name] = {"n_seeds": len(per_seed),
                               "record_mean": float(r.mean()), "record_std": float(r.std(ddof=1)),
                               "other_mean": float(o.mean()), "other_std": float(o.std(ddof=1)),
                               "per_seed": per_seed}
        print("%-13s record %s: %.3f +- %.3f   other records: %.3f +- %.3f"
              % (name, RECORD, r.mean(), r.std(ddof=1), o.mean(), o.std(ddof=1)))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
