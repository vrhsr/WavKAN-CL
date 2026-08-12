"""Regression test for the ablation-study training path (src/run_ablation_study.py).

Fixed per AUDIT_FINDINGS.md H5: the original version of this file had zero `assert`
statements and wrapped its body in a bare `try/except` that only printed a traceback on
failure -- no `raise`, no `sys.exit(1)`. Despite the `test_*.py` name (which pytest
auto-collects), it could never fail: a broken training run just printed a traceback and
the process still exited 0. Worse, the training calls sat at module level, so merely
*collecting* this file for test discovery triggered two live 1-epoch training runs as a
side effect of import, regardless of whether anyone wanted to run this test.

This version converts it to real pytest test functions with real assertions, and lets a
genuine failure actually fail the test. It requires `data/processed_rr_history/` (via
`ECGDatasetRR`), which is not present in this checkout (data/ is gitignored -- see
CLAUDE.md section 9) -- so both tests skip with a clear reason rather than erroring
confusingly, until the data pipeline has been regenerated from PhysioNet.
"""
import pathlib

import pytest

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "processed_rr_history"
# Checking a real data file, not just the directory: several scripts in this repo
# (e.g. process_data.py) call os.makedirs(OUT_DIR, exist_ok=True) at import time, which
# creates this directory empty as a side effect of merely importing them -- a
# directory-only check would then wrongly conclude data is present and try to run.
_REQUIRED_FILE = DATA_DIR / "y_train.npy"

pytestmark = pytest.mark.skipif(
    not _REQUIRED_FILE.exists(),
    reason=(
        f"{_REQUIRED_FILE} not present -- regenerate MIT-BIH data from PhysioNet first "
        "(see CLAUDE.md section 3/9). data/ is gitignored and absent in this checkout."
    ),
)


def _assert_valid_metrics(result, model):
    expected_keys = {"f1_macro", "n_recall", "s_recall", "v_recall", "f_recall", "val_best_f1"}
    assert expected_keys.issubset(result.keys()), f"Missing expected metric keys: {expected_keys - result.keys()}"
    for key in ("f1_macro", "n_recall", "s_recall", "v_recall", "f_recall", "val_best_f1"):
        value = result[key]
        assert 0.0 <= value <= 1.0, f"{key}={value} is not a valid [0,1] score"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0, "Model has zero trainable parameters"


def test_bspline_ablation_config_trains_without_error():
    import torch
    from src.run_ablation_study import FullModel, train_one_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel("b_spline").to(device)
    result = train_one_config(model, seed=42, epochs=1, device=device)
    _assert_valid_metrics(result, model)


def test_mexican_hat_focal_loss_ablation_config_trains_without_error():
    import torch
    from src.run_ablation_study import FullModel, train_one_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel("mexican_hat").to(device)
    result = train_one_config(model, seed=42, epochs=1, device=device, use_focal_loss=True)
    _assert_valid_metrics(result, model)
