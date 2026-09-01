"""Regression tests for src/fewshot_domain_adaptation.py.

Unit tests against tiny synthetic data (no real dataset/checkpoint needed) --
verifies the mechanism that matters for scientific integrity here: the
adaptation/eval split is genuinely record-disjoint and deterministic, and the
k-shot sampler respects real per-class availability caps rather than silently
padding or fabricating beats.
"""
import numpy as np
import torch

from models.wavkan_v2 import WavKAN_v2
from src.fewshot_domain_adaptation import (
    evaluate_model, fine_tune, sample_k_shot, split_adaptation_and_eval,
)


def test_split_is_record_disjoint():
    # 10 unique records, 20 beats each, interleaved.
    ids = np.array([f"rec{i}" for i in range(10) for _ in range(20)])
    adapt_mask, eval_mask = split_adaptation_and_eval(ids, adapt_fraction=0.3, seed=1337)

    assert not np.any(adapt_mask & eval_mask)
    assert np.all(adapt_mask | eval_mask)

    adapt_records = set(ids[adapt_mask].tolist())
    eval_records = set(ids[eval_mask].tolist())
    assert adapt_records.isdisjoint(eval_records)
    # No record split across both pools.
    for rec in np.unique(ids):
        rec_mask = ids == rec
        assert np.all(adapt_mask[rec_mask]) or np.all(eval_mask[rec_mask])


def test_split_is_deterministic_given_same_seed():
    ids = np.array([f"rec{i}" for i in range(10) for _ in range(20)])
    a1, e1 = split_adaptation_and_eval(ids, seed=1337)
    a2, e2 = split_adaptation_and_eval(ids, seed=1337)
    assert np.array_equal(a1, a2)
    assert np.array_equal(e1, e2)


def test_sample_k_shot_respects_class_availability_cap():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 360).astype(np.float32)
    Xrr = rng.randn(100, 5).astype(np.float32)
    # Class 3 (F) has only 2 beats -- fewer than k=10.
    y = np.array([0]*40 + [1]*30 + [2]*25 + [3]*2 + [4]*3)

    Xk, Xrrk, yk, capped = sample_k_shot(X, Xrr, y, k=10, seed=1337)

    assert 3 in capped
    assert capped[3] == 2
    assert (yk == 3).sum() == 2      # never fabricates beats beyond what exists
    assert (yk == 0).sum() == 10     # capped at k for classes with enough beats


def test_fine_tune_and_evaluate_run_without_crashing_on_tiny_synthetic_data():
    model = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=False)
    device = torch.device("cpu")

    rng = np.random.RandomState(0)
    X = rng.randn(20, 360).astype(np.float32)
    Xrr = rng.randn(20, 5).astype(np.float32)
    y = rng.randint(0, 5, size=20).astype(np.int64)

    fine_tune(model, X, Xrr, y, device, epochs=1, batch_size=8)
    metrics = evaluate_model(model, X, Xrr, y, device)

    assert "macro_f1" in metrics
    assert metrics["n_eval_beats"] == 20
