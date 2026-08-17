"""Regression tests for src/baselines_extended.py's training recipe.

Found 2026-08-17: a real 20-seed x 4-model Phase 5b run showed all 4 baseline
models (ResNet1D, Transformer, CNN+Focal, B-Spline KAN) never predicted class N
(the majority class) at all -- v_recall=0.0 for 3 of 4, macro_f1 in the 0.02-0.10
range vs. WavKAN-v2's ~0.32 on the same test set. Root cause: train_baseline()
stacked a fully class-balanced WeightedRandomSampler with an inverse-frequency-
weighted Focal Loss -- with this dataset's real N:Q ratio (~6300:1), that drove
N's effective per-sample weight to near-zero while also capping its per-batch
sampling frequency to match the rarest class, leaving ~zero net gradient signal
for N. These tests reproduce that failure mode on tiny synthetic data (fast,
no GPU, no real ECG data needed) and confirm the fix (natural-distribution
sampling only, matching train_pca.py's proven fair-baseline arm) avoids it.
"""
import json

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from src.baselines_extended import train_baseline

N_CLASSES = 5
# Extreme, MIT-BIH-like imbalance (real train ratio N:Q is ~6300:1) -- small
# enough to train in a couple seconds on CPU, skewed enough to reproduce the
# collapse if the double-correction bug were still present.
TRAIN_COUNTS = [300, 15, 20, 8, 3]
EVAL_COUNTS  = [60, 8, 8, 4, 2]


def _write_synthetic_split(data_dir, split, counts, seed):
    """Classes are trivially separable (distinct means) so a model that gets ANY
    real training signal for a class should learn to predict it sometimes --
    if it still never does, that's collapse, not just an unsolvable task."""
    rng = np.random.default_rng(seed)
    X, X_rr, y = [], [], []
    for c, n in enumerate(counts):
        offset = c * 8.0
        X.append(rng.normal(loc=offset, scale=1.0, size=(n, 360)).astype(np.float32))
        X_rr.append(rng.normal(loc=offset, scale=1.0, size=(n, 5)).astype(np.float32))
        y.append(np.full(n, c, dtype=np.int64))
    np.save(data_dir / f"X_{split}.npy", np.concatenate(X))
    np.save(data_dir / f"X_rr_{split}.npy", np.concatenate(X_rr))
    np.save(data_dir / f"y_{split}.npy", np.concatenate(y))


@pytest.fixture
def imbalanced_data_dir(tmp_path):
    data_dir = tmp_path / "synthetic_imbalanced"
    data_dir.mkdir()
    _write_synthetic_split(data_dir, "train", TRAIN_COUNTS, seed=0)
    _write_synthetic_split(data_dir, "val",   EVAL_COUNTS,  seed=1)
    _write_synthetic_split(data_dir, "test",  EVAL_COUNTS,  seed=2)
    return data_dir


def test_train_baseline_does_not_use_a_class_balanced_sampler(monkeypatch, imbalanced_data_dir, tmp_path):
    """Structural check for the actual fix, independent of training-outcome noise:
    train_baseline() must not construct a WeightedRandomSampler at all -- that's
    the specific mechanism that stacked with the loss weighting to cause collapse.
    """
    calls = []
    real_init = WeightedRandomSampler.__init__

    def tracking_init(self, *a, **kw):
        calls.append((a, kw))
        return real_init(self, *a, **kw)

    monkeypatch.setattr(WeightedRandomSampler, "__init__", tracking_init)

    train_baseline(
        model_name="resnet1d", seed=0, epochs=2, batch_size=16, patience=1000,
        data_dir=str(imbalanced_data_dir), output_dir=str(tmp_path / "out"),
    )

    assert calls == [], (
        "train_baseline() constructed a WeightedRandomSampler -- this is exactly "
        "the double-correction (balanced sampler + inverse-frequency loss weight) "
        "that caused the real 2026-08-17 baseline collapse. Training must use "
        "natural-distribution sampling only."
    )


def test_train_baseline_reports_n_recall(imbalanced_data_dir, tmp_path):
    """n_recall was previously omitted from test_metrics.json entirely (every
    other class's recall was reported, N's was silently dropped)."""
    metrics = train_baseline(
        model_name="cnn_focal", seed=0, epochs=3, batch_size=16, patience=1000,
        data_dir=str(imbalanced_data_dir), output_dir=str(tmp_path / "out"),
    )
    assert "n_recall" in metrics
    assert 0.0 <= metrics["n_recall"] <= 1.0


def test_train_baseline_does_not_collapse_away_from_majority_class(imbalanced_data_dir, tmp_path):
    """The actual, end-to-end failure mode: with the old sampler+loss double
    correction, the model reliably never predicted class 0 at all despite it
    being 100x more common than class 4 and trivially separable by construction.
    With natural sampling only, it should predict class 0 for at least some of
    the 60 real class-0 test examples.
    """
    out_dir = tmp_path / "out"
    train_baseline(
        model_name="resnet1d", seed=0, epochs=6, batch_size=16, patience=1000,
        data_dir=str(imbalanced_data_dir), output_dir=str(out_dir),
    )
    preds = np.load(out_dir / "predictions.npy")
    assert 0 in set(preds.tolist()), (
        "Model never predicted class 0 (the majority class) even once on a "
        "trivially-separable synthetic task -- this is the exact collapse "
        "signature found in the real 2026-08-17 run (v_recall=0.0, no N "
        "predictions at all)."
    )
