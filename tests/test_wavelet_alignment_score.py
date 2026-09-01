"""Regression tests for src/wavelet_alignment_score.py.

Was: the per-component results print (mu/gamma/right-arrow Unicode characters)
had no UTF-8 stdout handling, unlike its sibling export_quantize.py (already
fixed, CHANGELOG.md 2026-08-14) -- crashed with UnicodeEncodeError on a plain
Windows cp1252 console on every run that succeeded in loading a checkpoint,
not just on load failures. AUDIT_FINDINGS.md C19 (2026-08-27) already fixed
this script's checkpoint-load-*failure* print (an emoji); this is a second,
independent print statement on the success path (H30, found 2026-09-01).

Reproduced deterministically via PYTHONIOENCODING=cp1252 rather than relying
on the test-runner's actual OS/console encoding, so this fails the same way
on any platform if the guard regresses.
"""
import os
import subprocess
import sys
from pathlib import Path

import torch

from models.wavkan_v2 import WavKAN_v2

REPO_ROOT = str(Path(__file__).resolve().parents[1])


def test_cli_does_not_crash_under_a_non_utf8_stdout_encoding(tmp_path):
    ckpt_path = tmp_path / "fake_checkpoint.pth"
    torch.save(
        WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=False).state_dict(),
        ckpt_path,
    )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"  # forces the exact failure mode found on Windows

    result = subprocess.run(
        [sys.executable, "-m", "src.wavelet_alignment_score",
         "--checkpoint", str(ckpt_path),
         "--output-dir", str(tmp_path / "interp"),
         "--no-rr-attn"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=REPO_ROOT, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Macro WAS" in result.stdout
    assert (tmp_path / "interp" / "was_scores.json").exists()


def test_cli_no_rr_attn_reports_the_smaller_final_configuration(tmp_path):
    """use_rr_attn actually changes which architecture gets loaded -- a
    regression that silently ignored --no-rr-attn would fail a load against a
    real use_rr_attn=False checkpoint (mismatched state_dict shapes)."""
    ckpt_path = tmp_path / "fake_checkpoint.pth"
    torch.save(
        WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=False).state_dict(),
        ckpt_path,
    )

    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "src.wavelet_alignment_score",
         "--checkpoint", str(ckpt_path),
         "--output-dir", str(tmp_path / "interp"),
         "--no-rr-attn"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=REPO_ROOT, env=env,
    )
    assert result.returncode == 0, result.stderr
