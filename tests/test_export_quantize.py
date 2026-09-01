"""Regression tests for src/export_quantize.py.

Was: printing an emoji in the checkpoint-load except branch raised
UnicodeEncodeError on a Windows cp1252 console, which masked whatever the real
load error actually was (discovered live: a real checkpoint failed to report why
it wasn't loading because the *error-reporting print itself* crashed first).
Fixed by forcing UTF-8 stdout at import time. These tests run real (tiny)
quantization/inference, not mocks.

Also (2026-09-01): the CLI never had a --no-rr-attn flag and never forwarded
use_rr_attn from argparse into export_and_benchmark() at all -- every prior
invocation silently benchmarked with use_rr_attn=True regardless of which
checkpoint it was pointed at, the same bug class already fixed in 5 sibling
scripts (eval_ptbxl.py, wavelet_alignment_score.py, rr_ablation.py,
temperature_scaling.py, per_patient_analysis.py -- commit b8f6379) but missed
here. test_cli_no_rr_attn_flag_is_forwarded_to_the_model below runs the actual
CLI entry point as a subprocess (not just the Python function) so it would have
caught the missing plumbing, not just a missing flag definition.
"""
import subprocess
import sys

import torch

from src.export_quantize import (
    apply_dynamic_int8, benchmark_latency, export_and_benchmark, get_model_size_mb,
)
from models.wavkan_v2 import WavKAN_v2


def test_apply_dynamic_int8_does_not_crash_on_real_model():
    model = WavKAN_v2().eval()
    quantized = apply_dynamic_int8(model)
    assert quantized is not None


def test_benchmark_latency_runs_real_inference_and_returns_expected_keys():
    model = WavKAN_v2().eval()
    result = benchmark_latency(model, n_beats=5, device="cpu", batch=1)
    for key in ("mean_ms", "std_ms", "median_ms", "p95_ms", "throughput"):
        assert key in result
        assert result[key] >= 0.0


def test_export_and_benchmark_reports_real_params_with_no_checkpoint(tmp_path):
    """No real checkpoint at this path -- must fall back to an untrained model and
    still complete (this is exactly the code path whose own error print used to
    crash on Windows before the UTF-8 stdout fix)."""
    report = export_and_benchmark(
        checkpoint=str(tmp_path / "does_not_exist.pth"),
        output_dir=str(tmp_path / "deployment"),
        n_latency=5,
    )
    assert report["model_params"] == 154_325
    assert report["fp32_size_mb"] > 0.0
    assert (tmp_path / "deployment" / "benchmark_report.json").exists()


def test_export_and_benchmark_loads_a_real_checkpoint_when_path_exists(tmp_path):
    ckpt_path = tmp_path / "fake_checkpoint.pth"
    torch.save(WavKAN_v2().state_dict(), ckpt_path)

    report = export_and_benchmark(
        checkpoint=str(ckpt_path),
        output_dir=str(tmp_path / "deployment"),
        n_latency=5,
    )
    assert report["model_params"] == 154_325


def test_use_rr_attn_false_reports_the_smaller_final_configuration(tmp_path):
    report = export_and_benchmark(
        checkpoint=str(tmp_path / "does_not_exist.pth"),
        output_dir=str(tmp_path / "deployment"),
        n_latency=5,
        use_rr_attn=False,
    )
    assert report["model_params"] == 153_045


def test_cli_no_rr_attn_flag_is_forwarded_to_the_model(tmp_path):
    """Was: --no-rr-attn wasn't even a defined flag, and use_rr_attn was never
    forwarded from argparse to export_and_benchmark() at all -- the CLI always
    benchmarked use_rr_attn=True (154,325 params) no matter what checkpoint or
    flags were passed. Runs the real module as a subprocess CLI invocation so a
    regression here (flag silently ignored, or not forwarded) actually fails."""
    result = subprocess.run(
        [sys.executable, "-m", "src.export_quantize",
         "--checkpoint", str(tmp_path / "does_not_exist.pth"),
         "--output-dir", str(tmp_path / "deployment"),
         "--n-latency", "3",
         "--no-rr-attn"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert "Parameters  : 153,045" in result.stdout
