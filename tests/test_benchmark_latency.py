"""Regression tests for src/benchmark_latency.py.

Was: hardcoded a local HybridWavKAN_RR class (the old, no-longer-canonical 95K
architecture) and defaulted --model-path to results/hybrid_rr_history_20_seeds/...,
a path AUDIT_FINDINGS.md C5 confirms was never written by anything in this repo --
the exact script meant to produce the manuscript's missing results/latency_report.json
(H9) could never have benchmarked the real, canonical WavKAN_v2 model. Fixed to
import models.wavkan_v2.WavKAN_v2 and default to a real checkpoint path. These tests
run real (tiny) inference, not mocks -- consistent with how this repo already tests
train_pca.py's resume logic.
"""
import json

import pytest
import torch

from src.benchmark_latency import (
    count_parameters, measure_batch_throughput, measure_latency, run_benchmark,
)
from models.wavkan_v2 import WavKAN_v2


def test_imports_real_wavkan_v2_not_the_old_hybrid_class():
    """The whole point of the fix: this must be the 154,325-param canonical model,
    not the old 95,189-param HybridWavKAN_RR that used to be defined inline here."""
    model = WavKAN_v2()
    assert count_parameters(model) == 154_325


def test_measure_latency_runs_real_inference_and_returns_expected_keys():
    model = WavKAN_v2().eval()
    result = measure_latency(model, n_beats=5, warmup=2, device=torch.device("cpu"))
    for key in ("mean_ms", "median_ms", "std_ms", "p95_ms", "min_ms", "max_ms"):
        assert key in result
        assert result[key] >= 0.0


def test_measure_batch_throughput_runs_real_inference():
    model = WavKAN_v2().eval()
    throughput = measure_batch_throughput(model, batch_size=4, n_batches=3, device=torch.device("cpu"))
    assert throughput > 0.0


def test_run_benchmark_with_no_checkpoint_writes_valid_report(tmp_path):
    """No checkpoint path given -- must benchmark the real architecture with random
    weights rather than crashing, and must not silently use the old broken default."""
    out_path = tmp_path / "latency_report.json"
    report = run_benchmark(n_beats=5, warmup=2, batch_size=4, n_batches=3,
                            model_path=None, out_path=str(out_path))

    assert report["n_parameters"] == 154_325
    assert out_path.exists()
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["n_parameters"] == 154_325
    assert saved["latency"]["mean_ms"] >= 0.0


def test_run_benchmark_loads_a_real_checkpoint_when_path_exists(tmp_path):
    """Regression for the exact class of bug this fix addresses: a real checkpoint at
    a real path must load into the real architecture without a state-dict mismatch."""
    ckpt_path = tmp_path / "fake_checkpoint.pth"
    torch.save(WavKAN_v2().state_dict(), ckpt_path)
    out_path = tmp_path / "latency_report.json"

    report = run_benchmark(n_beats=5, warmup=2, batch_size=4, n_batches=3,
                            model_path=str(ckpt_path), out_path=str(out_path))

    assert report["model_path"] == str(ckpt_path)
    assert report["n_parameters"] == 154_325
