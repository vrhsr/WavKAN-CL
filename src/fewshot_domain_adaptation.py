"""
fewshot_domain_adaptation.py — Honest few-shot cross-dataset adaptation

Motivation: the AIIM rejection (AIIM_REJECTION_ANALYSIS.md) flagged PTB-XL
zero-shot generalization as too weak. This script asks a different, additive
question: given a *small* amount of real target-domain data, how much of that
gap closes? It reports the full answer -- including "not much" or "none" --
never a cherry-picked favorable k.

This is a fresh design, NOT a reuse of src/eval_fewshot_ptbxl.py, which
AUDIT_FINDINGS.md C8 documents has a silent numeric-fabrication fallback and a
docstring-stated intent to swap out an honest result that "hurt the
narrative." Concretely, this script differs from that landmine in every way
that matters:
  - The adaptation/held-out split is record- or patient-disjoint, fixed by a
    constant seed BEFORE any model touches the data -- never re-drawn per
    model-seed, per k, or after looking at a result.
  - Every k value in K_SHOTS is always run and always reported, including
    k=0 (zero-shot). There is no "if this k looks bad, drop it" path anywhere.
  - The zero-shot number here is recomputed on the SAME held-out-eval-pool
    subset the few-shot numbers use (not the full test set eval_ptbxl.py/
    eval_multidataset.py report) -- this is a deliberate apples-to-apples
    choice, and it means this script's k=0 number will differ slightly from
    the "official" zero-shot number elsewhere. Both are disclosed, not
    substituted for one another.
  - No fallback ever fabricates a metric when data or a checkpoint is
    missing -- it raises.

Design (see module docstring sections below for the reasoning):
  1. Load the target dataset's full test pool + its per-beat record/patient
     IDs (added to process_incart.py / process_svdb.py / process_ptbxl.py
     this same session specifically to make this split possible).
  2. Split UNIQUE record/patient IDs into an "adaptation pool" (~30%) and a
     disjoint "held-out eval pool" (~70%), by a fixed constant seed
     (SPLIT_SEED) independent of any model's training seed. This never
     changes across k values or model seeds -- the held-out eval pool a
     reviewer could re-derive from this script is always the same set of
     beats.
  3. For each of the 20 real, independently-trained MIT-BIH final-
     configuration checkpoints (results/ablation_no_rr_attn/seed_*/), and for
     each k in K_SHOTS:
       - Deep-copy the ORIGINAL zero-shot state_dict (so k=500 never starts
         from a checkpoint that already saw the k=100 adaptation beats --
         every k is an independent fine-tune from the same starting point).
       - Sample up to k beats per class from the adaptation pool only,
         deterministically (fixed SPLIT_SEED-derived sampling, identical
         across all 20 model seeds for a given k -- the only thing that
         varies across the 20 replicates is which independently-trained
         checkpoint is being adapted, not which target beats it saw).
       - Fine-tune ALL parameters for a small fixed epoch budget (no
         early stopping against target-domain labels -- there is no
         target-domain validation split to select against, by design, since
         rule 3 forbids any label-driven selection and carving out a third
         split would shrink an already-small k-shot budget further).
       - Evaluate ONCE on the held-out eval pool. Never touched again.
  4. Aggregate mean+-std across the 20 seeds, per k. Report the full curve.

Usage:
    python src/fewshot_domain_adaptation.py \\
        --dataset PTB-XL \\
        --checkpoints-dir results/ablation_no_rr_attn \\
        --output-dir results/fewshot_adaptation/ptbxl
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.wavkan_v2 import WavKAN_v2

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
N_CLASSES = len(CLASS_NAMES)

DATASET_CONFIG = {
    "INCART": {"data_dir": "data/incart_processed", "id_file": "record_ids_test.npy"},
    "SVDB":   {"data_dir": "data/svdb_processed",   "id_file": "record_ids_test.npy"},
    "PTB-XL": {"data_dir": "data/processed_ptbxl",  "id_file": "patient_ids_test.npy"},
}

K_SHOTS = [0, 10, 50, 100, 500]     # beats PER CLASS, capped by availability
SPLIT_SEED = 1337                   # fixed, independent of any model's training seed
ADAPT_FRACTION = 0.3                # fraction of unique records/patients -> adaptation pool

REAL_20_SEEDS = [7, 11, 13, 42, 99, 101, 333, 555, 777, 888, 1001, 1234,
                 1998, 2024, 2026, 5050, 8080, 9999, 27182, 31415]


def load_dataset(data_dir: str, id_file: str):
    base = Path(data_dir)
    X   = np.load(base / "X_test.npy")
    Xrr = np.load(base / "X_rr_test.npy")
    y   = np.load(base / "y_test.npy")
    ids = np.load(base / id_file, allow_pickle=True)
    if len(ids) != len(X):
        raise ValueError(
            f"{id_file} has {len(ids)} entries but X_test.npy has {len(X)} -- "
            "was this dataset reprocessed with the record/patient-ID-saving "
            "code without regenerating the ID file, or vice versa?"
        )
    return X, Xrr, y, ids


def split_adaptation_and_eval(ids: np.ndarray, adapt_fraction: float = ADAPT_FRACTION,
                                seed: int = SPLIT_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Splits beats into an adaptation-pool mask and a disjoint held-out-eval
    mask, by holding out whole records/patients (never splitting one record's
    beats across both pools). Deterministic given `seed`, independent of any
    model's training seed."""
    unique_ids = np.unique(ids)
    rng = np.random.RandomState(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)
    n_adapt = max(1, int(len(shuffled) * adapt_fraction))
    adapt_ids = set(shuffled[:n_adapt].tolist())
    adapt_mask = np.array([i in adapt_ids for i in ids])
    eval_mask = ~adapt_mask
    return adapt_mask, eval_mask


def sample_k_shot(X: np.ndarray, Xrr: np.ndarray, y: np.ndarray, k: int,
                   seed: int = SPLIT_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """Samples up to k beats per class, deterministically. Returns the
    sampled (X, Xrr, y) plus a dict of {class: n_available} for classes where
    fewer than k beats existed (so the report can disclose the cap)."""
    rng = np.random.RandomState(seed)
    idx_all = []
    capped = {}
    for c in range(N_CLASSES):
        class_idx = np.where(y == c)[0]
        if len(class_idx) == 0:
            continue
        if len(class_idx) < k:
            capped[c] = len(class_idx)
        chosen = rng.choice(class_idx, size=min(k, len(class_idx)), replace=False)
        idx_all.append(chosen)
    if not idx_all:
        return X[:0], Xrr[:0], y[:0], capped
    idx_all = np.concatenate(idx_all)
    return X[idx_all], Xrr[idx_all], y[idx_all], capped


def evaluate_model(model, X, Xrr, y, device, batch_size: int = 256) -> Dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            xrb = torch.tensor(Xrr[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(xb, xrb)
            preds.extend(logits.argmax(1).cpu().numpy())
    preds = np.array(preds)
    return {
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "v_recall": float(recall_score(y, preds, labels=[2], average="macro", zero_division=0)),
        "s_recall": float(recall_score(y, preds, labels=[1], average="macro", zero_division=0)),
        "n_recall": float(recall_score(y, preds, labels=[0], average="macro", zero_division=0)),
        "n_eval_beats": int(len(y)),
    }


def fine_tune(model, X, Xrr, y, device, epochs: int = 10, lr: float = 1e-4,
              batch_size: int = 32):
    """Fine-tunes ALL parameters on the k-shot adaptation set. No
    target-domain validation split / early stopping (rule 3: no
    label-driven selection, and a 3-way split would shrink an already-small
    k-shot budget further) -- fixed epoch budget instead."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = len(y)
    if n == 0:
        return model
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.tensor(X[idx], dtype=torch.float32).to(device)
            xrb = torch.tensor(Xrr[idx], dtype=torch.float32).to(device)
            yb = torch.tensor(y[idx], dtype=torch.long).to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, xrb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def run_fewshot_study(dataset: str, checkpoints_dir: str, output_dir: str,
                       seeds: List[int], k_shots: List[int], device: str,
                       epochs: int = 10, lr: float = 1e-4) -> Dict:
    cfg = DATASET_CONFIG[dataset]
    DEVICE = torch.device(device)
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset} from {cfg['data_dir']}...")
    X, Xrr, y, ids = load_dataset(cfg["data_dir"], cfg["id_file"])
    print(f"  {len(X)} beats, {len(np.unique(ids))} unique records/patients")

    adapt_mask, eval_mask = split_adaptation_and_eval(ids)
    X_adapt, Xrr_adapt, y_adapt = X[adapt_mask], Xrr[adapt_mask], y[adapt_mask]
    X_eval, Xrr_eval, y_eval = X[eval_mask], Xrr[eval_mask], y[eval_mask]
    print(f"  Adaptation pool: {len(X_adapt)} beats ({len(np.unique(ids[adapt_mask]))} records/patients)")
    print(f"  Held-out eval pool: {len(X_eval)} beats ({len(np.unique(ids[eval_mask]))} records/patients) "
          f"-- NEVER used for sampling or fine-tuning")

    results = {k: [] for k in k_shots}
    capped_log = {}

    for seed in seeds:
        ckpt_path = Path(checkpoints_dir) / f"seed_{seed}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  seed {seed}: checkpoint not found at {ckpt_path}, skipping")
            continue
        base_state = torch.load(ckpt_path, map_location=DEVICE)

        for k in k_shots:
            model = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=False).to(DEVICE)
            model.load_state_dict(copy.deepcopy(base_state))  # fresh copy every k

            if k > 0:
                Xk, Xrrk, yk, capped = sample_k_shot(X_adapt, Xrr_adapt, y_adapt, k)
                if capped:
                    capped_log[f"seed_{seed}_k{k}"] = capped
                fine_tune(model, Xk, Xrrk, yk, DEVICE, epochs=epochs, lr=lr)

            metrics = evaluate_model(model, X_eval, Xrr_eval, y_eval, DEVICE)
            metrics["seed"] = seed
            metrics["k"] = k
            results[k].append(metrics)
            print(f"  seed {seed:>5d}  k={k:>4d}  Macro-F1={metrics['macro_f1']:.4f}  "
                  f"V={metrics['v_recall']:.3f}  S={metrics['s_recall']:.3f}")

    summary = {}
    for k in k_shots:
        runs = results[k]
        if not runs:
            summary[k] = None
            continue
        summary[k] = {
            m: {"mean": float(np.mean([r[m] for r in runs])),
                "std": float(np.std([r[m] for r in runs], ddof=1) if len(runs) > 1 else 0.0)}
            for m in ["macro_f1", "v_recall", "s_recall", "n_recall"]
        }
        summary[k]["n_seeds"] = len(runs)

    out = {
        "dataset": dataset,
        "adaptation_pool_beats": int(len(X_adapt)),
        "held_out_eval_pool_beats": int(len(X_eval)),
        "split_seed": SPLIT_SEED,
        "adapt_fraction": ADAPT_FRACTION,
        "k_shots": k_shots,
        "summary": summary,
        "per_seed_raw": {str(k): results[k] for k in k_shots},
        "classes_capped_below_k": capped_log,
        "note": (
            "k=0 is zero-shot, recomputed on THIS script's own held-out eval "
            "pool -- it will differ slightly from the 'official' zero-shot "
            "number in results/ptbxl_zero_shot_metrics_multiseed_final.json "
            "or results/multidataset_table_final/, which evaluate on the "
            "full test set, not this disjoint subset. Both are real; report "
            "both, do not treat one as replacing the other."
        ),
    }
    with open(OUT / f"fewshot_{dataset.lower().replace('-', '')}_report.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*70}\nFEW-SHOT ADAPTATION SUMMARY -- {dataset}\n{'='*70}")
    for k in k_shots:
        s = summary[k]
        if s is None:
            print(f"  k={k:>4d}: no data")
            continue
        print(f"  k={k:>4d}  Macro-F1={s['macro_f1']['mean']:.4f}+-{s['macro_f1']['std']:.4f}  "
              f"V={s['v_recall']['mean']:.3f}+-{s['v_recall']['std']:.3f}  "
              f"S={s['s_recall']['mean']:.3f}+-{s['s_recall']['std']:.3f}  (n={s['n_seeds']})")
    print(f"\nSaved -> {OUT}/fewshot_{dataset.lower().replace('-', '')}_report.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Honest few-shot cross-dataset adaptation study")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--checkpoints-dir", type=str, default="results/ablation_no_rr_attn")
    parser.add_argument("--output-dir", type=str, default="results/fewshot_adaptation")
    parser.add_argument("--seeds", type=int, nargs="+", default=REAL_20_SEEDS)
    parser.add_argument("--k-shots", type=int, nargs="+", default=K_SHOTS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_fewshot_study(
        dataset=args.dataset,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        k_shots=args.k_shots,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
    )
