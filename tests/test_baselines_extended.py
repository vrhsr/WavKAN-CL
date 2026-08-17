"""Regression tests for src/baselines_extended.py's training recipe.

Found 2026-08-17 in two rounds against the real Phase 5b run:

Round 1: all 4 baseline models never predicted class N (the majority class) at
all -- v_recall=0.0 for 3 of 4, macro_f1 in the 0.02-0.10 range vs. WavKAN-v2's
~0.32. Root cause: train_baseline() stacked a fully class-balanced
WeightedRandomSampler with an inverse-frequency-weighted Focal Loss -- with this
dataset's real N:Q ratio (~6300:1), that drove N's effective per-sample weight to
near-zero while also capping its per-batch sampling frequency to match the rarest
class, leaving ~zero net gradient signal for N. Fixed: natural-distribution
sampling only (no sampler).

Round 2: the sampler fix alone was NOT sufficient -- a real re-run still showed
resnet1d collapsing (v_recall=0.0) even with natural sampling, because Focal
Loss's (1-pt)^gamma term further suppresses "easy" (well-classified) N examples,
compounding with N's already-tiny per-sample weight a second time. This is
exactly why train_pca.py's proven fair-baseline arm uses plain class-weighted
CrossEntropyLoss, not Focal Loss. Fixed: only "cnn_focal" (whose entire point is
testing focal loss) keeps FocalLoss; the other 3 (plain architecture baselines)
now use CrossEntropyLoss, matching the proven recipe exactly.

The first synthetic reproduction (round 1) used a ~100:1 imbalance ratio, mild
enough that it didn't actually reproduce round 2's failure -- these tests were
strengthened to use a ~1000:1 ratio (closer to the real ~6300:1) and to cover
BOTH loss paths (CrossEntropyLoss via resnet1d, FocalLoss via cnn_focal), so a
future regression in either direction would be caught here first.
"""
import json

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from src.baselines_extended import train_baseline

N_CLASSES = 5
# Real-scale-like imbalance (real train N:Q ratio is ~6300:1; this uses ~1000:1,
# strong enough to reproduce both round-1 and round-2 collapse if either bug were
# still present, while staying small/fast enough for a CPU regression test).
TRAIN_COUNTS = [1000, 40, 70, 10, 1]
EVAL_COUNTS  = [100, 10, 12, 4, 2]


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


@pytest.mark.parametrize("model_name", ["resnet1d", "cnn_focal"])
def test_train_baseline_does_not_collapse_away_from_majority_class(model_name, imbalanced_data_dir, tmp_path):
    """The actual, end-to-end failure mode, checked on BOTH loss paths this file
    uses (resnet1d -> CrossEntropyLoss, cnn_focal -> FocalLoss): with either the
    round-1 (sampler+loss double correction) or round-2 (Focal Loss compounding
    with natural sampling) bug present, the model reliably never predicted class 0
    at all despite it being ~1000x more common than class 4 and trivially
    separable by construction. With both fixes in place, it should predict class 0
    for at least some of the 100 real class-0 test examples.
    """
    out_dir = tmp_path / "out"
    train_baseline(
        model_name=model_name, seed=0, epochs=8, batch_size=16, patience=1000,
        data_dir=str(imbalanced_data_dir), output_dir=str(out_dir),
    )
    preds = np.load(out_dir / "predictions.npy")
    assert 0 in set(preds.tolist()), (
        f"[{model_name}] Model never predicted class 0 (the majority class) even "
        "once on a trivially-separable synthetic task -- this is the exact "
        "collapse signature found in the real 2026-08-17 runs (v_recall=0.0, no N "
        "predictions at all)."
    )
