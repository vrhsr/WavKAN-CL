"""Regression tests for src/generate_fig4_convergence.py's data-handling logic
(the plotting itself was verified by directly rendering the real output PDF, not by a
test -- see CHANGELOG.md 2026-08-13)."""
import json

import pytest

from src.generate_fig4_convergence import find_phase_transition_epoch, load_history


def test_find_phase_transition_epoch_detects_warmup_to_anneal():
    history = [
        {"epoch": 1, "phase": "WARMUP"},
        {"epoch": 2, "phase": "WARMUP"},
        {"epoch": 3, "phase": "ANNEAL(e=1.00)"},
        {"epoch": 4, "phase": "ANNEAL(e=0.90)"},
    ]
    assert find_phase_transition_epoch(history) == 3


def test_find_phase_transition_epoch_none_for_baseline_arm():
    history = [{"epoch": e, "phase": "BASELINE"} for e in range(1, 6)]
    assert find_phase_transition_epoch(history) is None


def test_load_history_raises_clear_error_on_missing_seed(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_history(str(tmp_path), seed=999999)


def test_load_history_reads_real_json(tmp_path):
    run_dir = tmp_path
    seed_dir = run_dir / "seed_1"
    seed_dir.mkdir()
    data = [{"epoch": 1, "phase": "WARMUP", "loss": 0.5,
             "val_macro_f1": 0.3, "val_v_recall": 0.8, "val_s_recall": 0.2}]
    (seed_dir / "training_history.json").write_text(json.dumps(data))
    assert load_history(str(run_dir), seed=1) == data
