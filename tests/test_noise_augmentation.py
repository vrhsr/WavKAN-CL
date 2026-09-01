"""Regression tests for src/noise_augmentation.py.

Was: train_one_strategy() hardcoded
WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=True) with no override --
the same bug class already fixed in 5+ sibling scripts (commit b8f6379,
AUDIT_FINDINGS.md H27) -- so the entire augmentation study reported in the
manuscript's Table tab:augmentation was trained on the superseded initial
(RR-self-attention) configuration, with no CLI way to point it at the final,
adopted one instead. Fixed by adding --no-rr-attn and threading use_rr_attn
through run_augmentation_study -> train_one_strategy.

These tests patch out real training (no data/processed_rr_history in this
checkout) and only verify the plumbing -- consistent with test_ablation.py's
pattern of not running live training as a side effect of test collection.
"""
import inspect
from unittest.mock import patch

from models.wavkan_v2 import WavKAN_v2
from src.noise_augmentation import run_augmentation_study, train_one_strategy


def test_train_one_strategy_has_use_rr_attn_param_defaulting_true():
    sig = inspect.signature(train_one_strategy)
    assert "use_rr_attn" in sig.parameters
    assert sig.parameters["use_rr_attn"].default is True


def test_run_augmentation_study_forwards_use_rr_attn_false(tmp_path):
    captured = []

    def fake_train_one_strategy(strategy_name, seed, epochs, data_dir, device,
                                 use_rr_attn=True, **kwargs):
        captured.append(use_rr_attn)
        return {"macro_f1": 0.0, "v_recall": 0.0, "s_recall": 0.0, "f_recall": 0.0}

    with patch("src.noise_augmentation.train_one_strategy", fake_train_one_strategy):
        run_augmentation_study(
            strategies=["none"],
            seeds=[42],
            epochs=1,
            data_dir="unused",
            output_dir=str(tmp_path),
            device="cpu",
            use_rr_attn=False,
        )
    assert captured == [False]


def test_run_augmentation_study_default_still_uses_initial_configuration(tmp_path):
    captured = []

    def fake_train_one_strategy(strategy_name, seed, epochs, data_dir, device,
                                 use_rr_attn=True, **kwargs):
        captured.append(use_rr_attn)
        return {"macro_f1": 0.0, "v_recall": 0.0, "s_recall": 0.0, "f_recall": 0.0}

    with patch("src.noise_augmentation.train_one_strategy", fake_train_one_strategy):
        run_augmentation_study(
            strategies=["none"],
            seeds=[42],
            epochs=1,
            data_dir="unused",
            output_dir=str(tmp_path),
            device="cpu",
        )
    assert captured == [True]


def test_model_construction_matches_use_rr_attn_flag():
    """The actual mechanism train_one_strategy relies on: the two
    configurations really do have different parameter counts, so a regression
    that silently ignores use_rr_attn would be architecturally detectable,
    not just a missing kwarg."""
    initial = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=True)
    final = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=False)
    assert sum(p.numel() for p in initial.parameters()) == 154_325
    assert sum(p.numel() for p in final.parameters()) == 153_045
