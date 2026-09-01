"""
svdb_confusion_diagnostic.py — Diagnosing the wavelet-KAN family's SVDB weakness

AUDIT_FINDINGS.md C16 (residual gap, closed 2026-09-01) established THAT the
wavelet-KAN family (WavKAN-v2 final + B-Spline KAN) is significantly weaker
than 3 of 4 baselines on SVDB Macro-F1, and tied with each other -- but not
WHY. This script computes full per-class confusion matrices (not just scalar
recall/F1) across all 20 real seeds, for all 6 models, on SVDB (and INCART for
contrast), to identify exactly which class pairs the wavelet-KAN family
confuses that the baselines do not.

This is diagnostic only: it does not select a model, seed, or checkpoint based
on the labels it inspects (rule 3) -- every already-trained checkpoint's
already-existing predictions are aggregated and reported as-is, whichever way
they land.

Usage:
    python src/svdb_confusion_diagnostic.py \\
        --results-base results \\
        --output-dir results/svdb_diagnostic
"""
import os, sys, json, argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval_multidataset import (
    CLASS_NAMES, DATASET_INFO, MODEL_CONSTRUCTORS, ECGDatasetRR,
    discover_checkpoints, REAL_20_SEEDS,
)

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

N_CLASSES = len(CLASS_NAMES)  # N, S, V, F, Q


def run_inference_labels(model, data_dir: str, split: str, device: str):
    try:
        ds = ECGDatasetRR(split, data_dir)
    except FileNotFoundError:
        return None, None
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, Xrr, y in loader:
            logits = model(X.to(device), Xrr.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
    return np.array(trues), np.array(preds)


def diagnose(results_base: str, output_dir: str, seeds: List[int],
             datasets: List[str], device: str = "cpu") -> Dict:
    DEVICE = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(results_base, seeds)
    out = {}

    for ds_name in datasets:
        ds_info = DATASET_INFO[ds_name]
        out[ds_name] = {}
        print(f"\n{'='*70}\nDataset: {ds_name}\n{'='*70}")

        for model_name, ckpt_paths in checkpoints.items():
            summed_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
            n_used = 0
            for ckpt_path in ckpt_paths:
                model = MODEL_CONSTRUCTORS[model_name]().to(DEVICE)
                state = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(state)
                y_true, y_pred = run_inference_labels(
                    model, ds_info["data_dir"], ds_info["split"], str(DEVICE)
                )
                if y_true is None:
                    continue
                cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
                summed_cm += cm
                n_used += 1

            if n_used == 0:
                print(f"  {model_name:14s}: no data")
                continue

            row_sums = summed_cm.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                row_normalized = np.divide(
                    summed_cm, row_sums, out=np.zeros_like(summed_cm, dtype=float),
                    where=row_sums != 0,
                )

            out[ds_name][model_name] = {
                "n_seeds_used": n_used,
                "summed_confusion_matrix": summed_cm.tolist(),
                "row_normalized_confusion_matrix": row_normalized.tolist(),
                "class_names": CLASS_NAMES,
            }

            print(f"\n  {model_name} (summed over {n_used} seeds, rows=true, cols=pred, "
                  f"row-normalized %):")
            header = "        " + "".join(f"{c:>8s}" for c in CLASS_NAMES)
            print(header)
            for i, cname in enumerate(CLASS_NAMES):
                if row_sums[i, 0] == 0:
                    continue
                row_str = "".join(f"{row_normalized[i, j]*100:7.1f}%" for j in range(N_CLASSES))
                print(f"  {cname:>4s}  {row_str}")

    with open(OUT / "svdb_incart_confusion_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT}/svdb_incart_confusion_diagnostic.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVDB/INCART confusion-matrix diagnostic")
    parser.add_argument("--results-base", type=str, default="results")
    parser.add_argument("--output-dir",   type=str, default="results/svdb_diagnostic")
    parser.add_argument("--datasets",     nargs="+", default=["SVDB", "INCART"],
                        choices=list(DATASET_INFO.keys()))
    parser.add_argument("--seeds",        type=int, nargs="+", default=REAL_20_SEEDS)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    diagnose(
        results_base=args.results_base,
        output_dir=args.output_dir,
        seeds=args.seeds,
        datasets=args.datasets,
        device=args.device,
    )
