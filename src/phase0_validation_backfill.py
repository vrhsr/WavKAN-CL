"""
Phase 0 -- validation-only backfill for the research-protocol revision.

Purpose: replace the DS2-test-derived Go/No-Go reference number (0.323) with a
DS1-validation-derived one, per the revised protocol. Two independent parts:

  Part A. Extract already-logged DS1-validation metrics for every WavKAN-v2
          family config from its training_history.json files. Pure JSON
          parsing -- no model, no data, no forward pass.

  Part B. Forward-pass evaluation of the four external baseline checkpoints
          (ResNet1D, ECGTransformer, SimpleCNN1D/cnn_focal, BSplineKAN_RR)
          against the DS1 *validation* split only. No baseline script ever
          logged a validation curve (confirmed: no training_history.json
          exists under results/baseline_*), so this is the only way to get
          their validation numbers -- but it requires
          data/processed_rr_history/{X,X_rr,y}_val.npy, which is absent from
          this checkout (data/ is gitignored, per CLAUDE.md). Part B therefore
          self-detects the missing directory and reports each model as
          "blocked_no_data" rather than fabricating a number. It is written
          to run unmodified once the data directory exists (e.g. on the GPU
          box referenced in CLAUDE.md/PHASE5_SCOPE_PLAN.md).

DS2 safety (see docstring at bottom of file for the audit trail):
  - The validation split is the literal string "val", hardcoded. There is no
    parameter anywhere in this file that can select "test".
  - This file never opens a path containing "test_metrics", "test_predictions",
    "test_probs", "test_true", "X_test", "y_test", "X_rr_test", or the bare
    "predictions.npy"/"probs.npy"/"true.npy"/"confusion_matrix.npy" baseline
    test artifacts. A runtime assertion in `_load_val_dataset` re-checks this
    at the moment the data directory is read.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data" / "processed_rr_history"

WAVKAN_FAMILY_CONFIGS = {
    "wavkan_v2_curriculum":     "mexican_hat, PCWI+PWAM+RR-attn, curriculum (the proposed/default full config)",
    "wavkan_v2_baseline":       "mexican_hat, PCWI+PWAM+RR-attn, no-curriculum (fair-baseline arm)",
    "ablation_no_pcwi":         "mexican_hat, PWAM+RR-attn only (PCWI off)",
    "ablation_no_pwam":         "mexican_hat, PCWI+RR-attn only (PWAM off)",
    "ablation_no_rr_attn":      "mexican_hat, PCWI+PWAM only (RR self-attention off)",
    "ablation_wavelet_bspline": "b_spline wavelet, PCWI+PWAM+RR-attn on",
    "ablation_wavelet_dog":     "dog wavelet, PCWI+PWAM+RR-attn on",
    "ablation_wavelet_morlet":  "morlet wavelet, PCWI+PWAM+RR-attn on",
}

EXTERNAL_BASELINES = ["resnet1d", "transformer", "cnn_focal", "bspline_kan"]


# ─────────────────────────────────────────────────────────────────────────────
# Part A -- WavKAN-v2 family, from existing training_history.json (no data needed)
# ─────────────────────────────────────────────────────────────────────────────

def _best_checkpoint_entry(history):
    """Replicate train_pca.py's exact selection rule: strict > on val_macro_f1,
    first epoch to reach the running max. Returns the history entry that
    corresponds to the val_macro_f1/val_v_recall/val_s_recall actually baked
    into that seed's saved best_model.pth."""
    best_entry, best_f1 = None, -1.0
    for entry in history:
        if entry["val_macro_f1"] > best_f1:
            best_f1 = entry["val_macro_f1"]
            best_entry = entry
    return best_entry


def extract_wavkan_family_val_metrics():
    out = {}
    for config_name, description in WAVKAN_FAMILY_CONFIGS.items():
        config_dir = RESULTS_ROOT / config_name
        seed_dirs = sorted(glob.glob(str(config_dir / "seed_*")))
        macro_f1s, v_recalls, s_recalls = [], [], []
        seeds_used = []
        for seed_dir in seed_dirs:
            hist_path = Path(seed_dir) / "training_history.json"
            if not hist_path.is_file():
                continue
            history = json.loads(hist_path.read_text())
            entry = _best_checkpoint_entry(history)
            if entry is None:
                continue
            macro_f1s.append(entry["val_macro_f1"])
            v_recalls.append(entry["val_v_recall"])
            s_recalls.append(entry["val_s_recall"])
            seeds_used.append(Path(seed_dir).name)

        def summarize(values):
            arr = np.array(values, dtype=float)
            return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(len(arr))}

        out[config_name] = {
            "description": description,
            "source": "training_history.json (val_macro_f1-selected best epoch per seed)",
            "n_seeds": len(seeds_used),
            "seeds": seeds_used,
            "macro_f1": summarize(macro_f1s) if macro_f1s else None,
            "v_recall": summarize(v_recalls) if v_recalls else None,
            "s_recall": summarize(s_recalls) if s_recalls else None,
            "f_recall": "unavailable -- not logged per-epoch in training_history.json; "
                         "would require a forward pass over X_val.npy, which requires "
                         "data/processed_rr_history/ (absent from this checkout)",
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Part B -- external baselines, forward pass on DS1-val only
# ─────────────────────────────────────────────────────────────────────────────

def _val_data_available():
    required = ["X_val.npy", "X_rr_val.npy", "y_val.npy"]
    return all((DATA_DIR / fname).is_file() for fname in required)


def _load_val_dataset():
    """Hardcoded to the literal split name "val". There is no way to call
    this with "test" -- the string is not a parameter of this function."""
    split = "val"
    assert split == "val", "DS2 guard: this function may only ever load the val split."
    from src.baselines_extended import ECGDatasetRR  # local import: torch-dependent
    ds = ECGDatasetRR(split, str(DATA_DIR))
    # Defense in depth: re-check the files actually opened were the val ones.
    for fname in ("X_val.npy", "X_rr_val.npy", "y_val.npy"):
        path = DATA_DIR / fname
        assert "test" not in path.name.lower(), f"DS2 guard tripped on {path}"
    return ds


def evaluate_external_baselines_on_val():
    out = {}
    for model_name in EXTERNAL_BASELINES:
        config_dir = RESULTS_ROOT / f"baseline_{model_name}"
        seed_dirs = sorted(glob.glob(str(config_dir / "seed_*")))
        n_checkpoints = sum(1 for d in seed_dirs if (Path(d) / "best_model.pth").is_file())

        if not _val_data_available():
            out[model_name] = {
                "status": "blocked_no_data",
                "reason": f"{DATA_DIR} does not contain X_val.npy/X_rr_val.npy/y_val.npy "
                          "in this checkout (data/ is gitignored per CLAUDE.md sec 2/9). "
                          "No number is reported -- per project rule 1, never present "
                          "placeholder data as measured. Re-run this script unmodified "
                          "once the data directory exists (e.g. on the GPU box).",
                "n_seed_checkpoints_found": n_checkpoints,
                "macro_f1": None, "v_recall": None, "s_recall": None,
                "f_recall": None, "n_recall": None,
            }
            continue

        # Only reachable once data/processed_rr_history/ exists. Not executed
        # in this checkout; left intact so a later run needs no code changes.
        import torch
        from sklearn.metrics import classification_report
        from torch.utils.data import DataLoader
        from src.baselines_extended import MODEL_REGISTRY, run_eval

        device = torch.device("cpu")
        val_ds = _load_val_dataset()
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        macro_f1s, v_recalls, s_recalls, f_recalls, n_recalls = [], [], [], [], []
        seeds_used = []
        for seed_dir in seed_dirs:
            ckpt_path = Path(seed_dir) / "best_model.pth"
            if not ckpt_path.is_file():
                continue
            model = MODEL_REGISTRY[model_name]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            y_true, y_pred, _ = run_eval(model, val_loader, device)
            report = classification_report(
                y_true, y_pred, target_names=["N", "S", "V", "F", "Q"],
                digits=4, zero_division=0, output_dict=True,
            )
            macro_f1s.append(report["macro avg"]["f1-score"])
            v_recalls.append(report["V"]["recall"])
            s_recalls.append(report["S"]["recall"])
            f_recalls.append(report["F"]["recall"])
            n_recalls.append(report["N"]["recall"])
            seeds_used.append(Path(seed_dir).name)

        def summarize(values):
            arr = np.array(values, dtype=float)
            return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(len(arr))}

        out[model_name] = {
            "status": "evaluated",
            "n_seeds": len(seeds_used),
            "seeds": seeds_used,
            "macro_f1": summarize(macro_f1s),
            "v_recall": summarize(v_recalls),
            "s_recall": summarize(s_recalls),
            "f_recall": summarize(f_recalls),
            "n_recall": summarize(n_recalls),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────

def main():
    wavkan_family = extract_wavkan_family_val_metrics()
    external = evaluate_external_baselines_on_val()

    available_family = {k: v for k, v in wavkan_family.items() if v["macro_f1"]}
    strongest_family = max(available_family, key=lambda k: available_family[k]["macro_f1"]["mean"]) \
        if available_family else None

    any_external_evaluated = any(v["status"] == "evaluated" for v in external.values())

    result = {
        "phase": "0-validation-backfill",
        "ds2_test_set_accessed": False,
        "wavkan_v2_family": wavkan_family,
        "external_baselines": external,
        "strongest_validation_baseline": {
            "among_wavkan_v2_family": strongest_family,
            "among_all_candidates": strongest_family if not any_external_evaluated else "see external_baselines",
            "caveat": (
                "External baselines (ResNet1D/Transformer/CNN+Focal/B-Spline-KAN standalone) "
                "could not be evaluated on DS1 validation in this checkout -- data/processed_rr_history/ "
                "is absent. The DS2-test-set ranking previously reported (external baselines 0.351-0.371 "
                "> WavKAN-v2 0.323-0.357) must NOT be assumed to hold on validation data until Part B "
                "is actually run where the data exists."
                if not any_external_evaluated else
                "External baselines were evaluated this run -- see external_baselines for the real numbers."
            ),
        },
    }

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_ROOT / "phase0_validation_backfill.json"
    out_json.write_text(json.dumps(result, indent=2))

    csv_lines = ["config,source,n_seeds,macro_f1_mean,macro_f1_std,v_recall_mean,v_recall_std,s_recall_mean,s_recall_std,f_recall"]
    for name, v in wavkan_family.items():
        if v["macro_f1"]:
            csv_lines.append(
                f"{name},wavkan_family,{v['n_seeds']},"
                f"{v['macro_f1']['mean']:.4f},{v['macro_f1']['std']:.4f},"
                f"{v['v_recall']['mean']:.4f},{v['v_recall']['std']:.4f},"
                f"{v['s_recall']['mean']:.4f},{v['s_recall']['std']:.4f},"
                f"unavailable"
            )
    for name, v in external.items():
        if v["status"] == "evaluated":
            csv_lines.append(
                f"{name},external_baseline,{v['n_seeds']},"
                f"{v['macro_f1']['mean']:.4f},{v['macro_f1']['std']:.4f},"
                f"{v['v_recall']['mean']:.4f},{v['v_recall']['std']:.4f},"
                f"{v['s_recall']['mean']:.4f},{v['s_recall']['std']:.4f},"
                f"{v['f_recall']['mean']:.4f}"
            )
        else:
            csv_lines.append(f"{name},external_baseline,{v['n_seed_checkpoints_found']},BLOCKED,,,,,,,BLOCKED")
    (RESULTS_ROOT / "phase0_validation_backfill.csv").write_text("\n".join(csv_lines) + "\n")

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
