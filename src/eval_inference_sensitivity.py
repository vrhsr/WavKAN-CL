"""
eval_inference_sensitivity.py -- forward-pass-only re-evaluation of the published
checkpoints on a given set of arrays. No training, no tuning, no selection.

Added 2026-09-30 (final pre-submission audit). Used for two analyses:

  * External data under the matched pipeline (AUDIT_FINDINGS.md H58): INCART and SVDB
    re-extracted with src/extract_matched.py, i.e. the same filter, window, z-scoring,
    interval formula and AAMI mapping as the MIT-BIH training data.
  * RR sensitivity (AUDIT_FINDINGS.md H59): MIT-BIH DS2 with the published interval
    features vs. intervals computed between beat annotations only.

For each of the five models (the adopted PC-WavKAN configuration and the four
baselines), all 20 published seeds are evaluated. Per-seed confusion matrices are
saved, so every derived number is recomputable. Statistics use the canonical
src/paired_stats.py (seed-keyed pairing, two-sided Wilcoxon, Holm within a declared
family, paired d = mean/sd of differences).

Usage:
  python src/eval_inference_sensitivity.py --data-dir data/incart_matched --out results/external_matched/incart
  python src/eval_inference_sensitivity.py --data-dir data/processed_rr_history \
      --rr-override data/mitdb_test_beats_only_rr/X_rr_test.npy --out results/rr_sensitivity/beats_only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.wavkan_v2 import WavKAN_v2                                   # noqa: E402
from src.baselines_extended import ResNet1D, ECGTransformer, SimpleCNN1D, BSplineKAN_RR  # noqa: E402
from src import paired_stats as ps                                       # noqa: E402

SEEDS = [42, 101, 777, 2026, 9999, 1234, 2024, 31415, 27182, 7,
         11, 13, 888, 555, 333, 99, 1001, 5050, 8080, 1998]

MODELS = {
    "PC-WavKAN":    ("results/ablation_no_rr_attn",
                     lambda: WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=False)),
    "ResNet1D":     ("results/baseline_resnet1d", ResNet1D),
    "Transformer":  ("results/baseline_transformer", ECGTransformer),
    "CNN+Focal":    ("results/baseline_cnn_focal", SimpleCNN1D),
    "B-Spline KAN": ("results/baseline_bspline_kan", BSplineKAN_RR),
}
REFERENCE = "PC-WavKAN"
CLASSES = ["N", "S", "V", "F", "Q"]


def confusion(y_true, y_pred, k=5):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    row = cm.sum(1, keepdims=True)
    rown = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    # Same label set as sklearn's f1_score(average="macro") without `labels=`, which is
    # what every published number used: classes present in y_true or y_pred. On MIT-BIH
    # DS2 all five classes have support; an external set with no Q beats and no Q
    # predictions averages over four.
    used = (cm.sum(1) > 0) | (cm.sum(0) > 0)
    out = {
        "macro_f1": float(f1[used].mean()),
        "macro_f1_nsvf": float(f1[:4].mean()),
        "s_to_v_rate": float(rown[1, 2]),
        "v_pred_share": float(cm[:, 2].sum() / cm.sum()),
        "n_beats": int(cm.sum()),
    }
    for i, c in enumerate(CLASSES):
        out[f"{c.lower()}_precision"] = float(prec[i])
        out[f"{c.lower()}_recall"] = float(rec[i])
        out[f"{c.lower()}_f1"] = float(f1[i])
    return out


DEVICE = "cpu"
BATCH = 1024


@torch.no_grad()
def predict(model, X, R, batch=None):
    batch = batch or BATCH
    model.eval().to(DEVICE)
    out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i:i + batch].to(DEVICE), R[i:i + batch].to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--rr-override", default=None,
                    help="replace X_rr with this array (same beat order) -- RR sensitivity")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-published", action="store_true",
                    help="assert each checkpoint reproduces its saved DS2 predictions (MIT-BIH only)")
    # Parallel mode (added 2026-09-30 because the Transformer baseline makes a full
    # external pass take many CPU-hours): evaluate a subset of (model, seed) pairs and
    # write one small JSON per pair into --parts-dir; a later --merge-parts run
    # assembles them and writes exactly the same per_seed.json/report.json as the
    # serial path. Parts are skipped if already present, so runs are resumable.
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names (parts mode)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="subset of seeds (parts mode)")
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    ap.add_argument("--parts-dir", default=None, help="write one JSON per (model, seed) here")
    ap.add_argument("--merge-parts", action="store_true", help="assemble --parts-dir into the report")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--batch", type=int, default=1024,
                    help="inference batch size; the Transformer baseline needs ~2 GB of attention "
                         "maps per layer at 1024, so use 64-256 when running jobs in parallel")
    a = ap.parse_args()
    global BATCH
    BATCH = a.batch
    if a.threads:
        torch.set_num_threads(a.threads)
    global DEVICE
    DEVICE = a.device
    if DEVICE.startswith("cuda"):
        # plain FP32 on GPU: TF32 would change the numerics relative to the CPU path
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    d = Path(a.data_dir)
    y = np.load(d / f"y_{a.split}.npy")
    if a.merge_parts:
        per_seed = {name: {} for name in MODELS}
        for name in MODELS:
            for s in SEEDS:
                part = Path(a.parts_dir) / f"{name.replace(' ', '_').replace('+', 'p')}__{s}.json"
                cm = json.load(open(part))["confusion_matrix"]
                per_seed[name][str(s)] = {"confusion_matrix": cm, **metrics_from_cm(cm)}
        _write_report(a, d, y, per_seed, [])
        return

    X = torch.tensor(np.load(d / f"X_{a.split}.npy"), dtype=torch.float32)
    R = np.load(a.rr_override) if a.rr_override else np.load(d / f"X_rr_{a.split}.npy")
    R = torch.tensor(R, dtype=torch.float32)
    assert len(X) == len(R) == len(y), "array lengths differ"

    if a.parts_dir:
        os.makedirs(a.parts_dir, exist_ok=True)
        for name in (a.models or list(MODELS)):
            rdir, ctor = MODELS[name]
            for s in (a.seeds or SEEDS):
                part = Path(a.parts_dir) / f"{name.replace(' ', '_').replace('+', 'p')}__{s}.json"
                if part.exists():
                    continue
                m = ctor()
                m.load_state_dict(torch.load(ROOT / rdir / f"seed_{s}" / "best_model.pth", map_location="cpu"))
                cm = confusion(y, predict(m, X, R))
                tmp = str(part) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"model": name, "seed": s, "data_dir": str(d), "rr_override": a.rr_override,
                               "confusion_matrix": cm.tolist()}, f)
                os.replace(tmp, part)
                print(f"part {name} seed {s} done", flush=True)
        return

    per_seed, mismatches = {}, []
    for name, (rdir, ctor) in MODELS.items():
        per_seed[name] = {}
        for s in SEEDS:
            ck = ROOT / rdir / f"seed_{s}" / "best_model.pth"
            m = ctor()
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            yp = predict(m, X, R)
            if a.check_published:
                ref = ROOT / rdir / f"seed_{s}" / ("test_predictions.npy" if name == REFERENCE else "predictions.npy")
                if ref.exists():
                    saved = np.load(ref)
                    saved = saved.argmax(1) if saved.ndim > 1 else saved
                    n_diff = int((saved != yp).sum()) if len(saved) == len(yp) else -1
                    if n_diff != 0:
                        mismatches.append((name, s, n_diff))
            cm = confusion(y, yp)
            per_seed[name][str(s)] = {"confusion_matrix": cm.tolist(), **metrics_from_cm(cm)}
        mf = [v["macro_f1"] for v in per_seed[name].values()]
        print(f"{name:13s} Macro-F1 {np.mean(mf):.4f}+-{np.std(mf, ddof=1):.4f}", flush=True)
    _write_report(a, d, y, per_seed, mismatches)


def _write_report(a, d, y, per_seed, mismatches):
    summary = {}
    for name in MODELS:
        summary[name] = {k: ps.describe([v[k] for v in per_seed[name].values()])
                         for k in next(iter(per_seed[name].values())) if k != "confusion_matrix"}
    fam = {}
    for name in MODELS:
        if name == REFERENCE:
            continue
        fam[name] = ps.paired_compare(
            {s: v["macro_f1"] for s, v in per_seed[REFERENCE].items()},
            {s: v["macro_f1"] for s, v in per_seed[name].items()}, "macro_f1")
    ps.holm_family(fam)

    os.makedirs(a.out, exist_ok=True)
    report = {
        "data_dir": str(d), "split": a.split, "rr_override": a.rr_override,
        "n_beats": int(len(y)), "class_counts": np.bincount(y, minlength=5).tolist(),
        "seeds": SEEDS, "checkpoint_dirs": {k: v[0] for k, v in MODELS.items()},
        "published_prediction_mismatches": mismatches if a.check_published else "not checked",
        "summary": summary,
        "macro_f1_vs_reference_holm_family": fam,
        "note": "Forward pass only. Positive cohens_d favours PC-WavKAN.",
    }
    with open(os.path.join(a.out, "per_seed.json"), "w") as f:
        json.dump(per_seed, f)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    for k, v in fam.items():
        print(f"  PC-WavKAN - {k:12s}: diff {v['mean_diff']:+.4f} d={v['cohens_d']:+.2f} Holm p={v['holm_p']:.3g}")
    if a.check_published:
        print("published-prediction mismatches:", mismatches or "none")


if __name__ == "__main__":
    main()
