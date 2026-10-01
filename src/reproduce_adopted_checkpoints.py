"""
reproduce_adopted_checkpoints.py -- forward-only reproduction check of the 20
adopted-configuration checkpoints (results/ablation_no_rr_attn/seed_*/best_model.pth).

Backs the manuscript's test-set-exposure paragraph (Submission_Array/manuscript.tex,
sec:results_selection): "all 20 adopted-configuration checkpoints reproduce their DS2
predictions beat for beat, and 19 reproduce exactly the validation Macro-F1 logged at
their saved epoch. Seed 1001's logged validation score does not reproduce."
AUDIT_FINDINGS.md H50 records that forensic check; this script makes it re-runnable and
writes its result to a file the verifier can read (added 2026-10-01, final review).

No training. Uses the trainer's own dataset class and evaluate() (as preflight 2b of
run_remote_experiment.py does) on arrays regenerated with the unmodified
src/process_data.py. CPU is enough; batch 256 bounds memory.

Usage:
    python src/reproduce_adopted_checkpoints.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from models.wavkan_v2 import WavKAN_v2  # noqa: E402
from src.train_pca import ECGDatasetRR, evaluate  # noqa: E402

ARM = "results/ablation_no_rr_attn"
DATA = "data/processed_rr_history"
OUT = "results/checkpoint_reproduction/adopted_config.json"


def main():
    dev = torch.device("cpu")
    loaders = {s: DataLoader(ECGDatasetRR(s, DATA), batch_size=256, shuffle=False)
               for s in ("test", "val")}
    rows = []
    for sd in sorted(glob.glob(os.path.join(ARM, "seed_*"))):
        m = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=False).to(dev)
        m.load_state_dict(torch.load(os.path.join(sd, "best_model.pth"), map_location=dev))
        got = {}
        for split, loader in loaders.items():
            yt, yp, _ = evaluate(m, loader, dev)
            got[split] = (float(f1_score(yt, yp, average="macro", zero_division=0)), np.asarray(yp))
        hist = json.load(open(os.path.join(sd, "training_history.json")))
        saved_pred = np.load(os.path.join(sd, "test_predictions.npy"))
        row = {"seed": os.path.basename(sd),
               "test_macro_f1_saved": json.load(open(os.path.join(sd, "test_metrics.json")))["macro_f1"],
               "test_macro_f1_reproduced": got["test"][0],
               "test_predictions_identical": bool(np.array_equal(saved_pred, got["test"][1])),
               "val_macro_f1_logged_best_epoch": max(e["val_macro_f1"] for e in hist),
               "val_macro_f1_reproduced": got["val"][0]}
        rows.append(row)
        print(row, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {"arm": ARM, "data_dir": DATA, "device": "cpu", "batch_size": 256,
           "note": "Forward pass only. Validation is reproduced exactly when "
                   "|reproduced - logged| < 1e-9.",
           "rows": rows}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    exact = [r["seed"] for r in rows
             if abs(r["val_macro_f1_reproduced"] - r["val_macro_f1_logged_best_epoch"]) < 1e-9]
    print("test predictions identical: %d/%d" % (sum(r["test_predictions_identical"] for r in rows), len(rows)))
    print("validation reproduced exactly: %d/%d; not: %s"
          % (len(exact), len(rows), sorted(set(r["seed"] for r in rows) - set(exact))))
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
