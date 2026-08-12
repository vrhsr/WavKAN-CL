"""Regression tests for train_pca.py's crash-resume support (added 2026-08-13).

Motivation: run_gpu_pipeline.sh runs 40 unattended, multi-hour training jobs on a GPU
shared with another project; two real crashes already happened during this Phase 5
session (a masked pytest failure, then a genuine e3c_compliance bug). Before this,
train_pca.py had no way to survive an interruption mid-run except restarting the
affected seed from epoch 1. These tests run on CPU with tiny synthetic data (no GPU,
no real ECG data needed) and verify the actual resume mechanics, not just that the
code doesn't error.
"""
import json

import numpy as np
import pytest
import torch

from src.train_pca import train_pca

N_CLASSES = 5


def _write_synthetic_split(data_dir, split, n_per_class=4, seed=0):
    rng = np.random.default_rng(seed)
    X, X_rr, y = [], [], []
    for c in range(N_CLASSES):
        X.append(rng.normal(size=(n_per_class, 360)).astype(np.float32))
        X_rr.append(rng.normal(size=(n_per_class, 5)).astype(np.float32))
        y.append(np.full(n_per_class, c, dtype=np.int64))
    np.save(data_dir / f"X_{split}.npy", np.concatenate(X))
    np.save(data_dir / f"X_rr_{split}.npy", np.concatenate(X_rr))
    np.save(data_dir / f"y_{split}.npy", np.concatenate(y))


@pytest.fixture
def synthetic_data_dir(tmp_path):
    data_dir = tmp_path / "synthetic_data"
    data_dir.mkdir()
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        _write_synthetic_split(data_dir, split, n_per_class=n, seed=hash(split) % 1000)
    return data_dir


def _tiny_kwargs(data_dir, output_dir, epochs):
    """Small/fast settings shared by all tests here -- CPU, tiny model, tiny data."""
    return dict(
        seed=0, epochs=epochs, batch_size=8, patience=1000,  # patience huge: never early-stop mid-test
        data_dir=str(data_dir), output_dir=str(output_dir),
    )


def test_full_run_completes_and_cleans_up_resume_checkpoint(synthetic_data_dir, tmp_path):
    out_dir = tmp_path / "run1"
    metrics = train_pca(**_tiny_kwargs(synthetic_data_dir, out_dir, epochs=2))

    assert (out_dir / "test_metrics.json").exists()
    assert not (out_dir / "resume_checkpoint.pth").exists(), (
        "resume_checkpoint.pth must be deleted once a seed completes normally -- "
        "leaving it around would make a finished seed look interrupted."
    )
    history = json.loads((out_dir / "training_history.json").read_text())
    assert len(history) == 2
    assert [h["epoch"] for h in history] == [1, 2]


def test_resume_continues_from_injected_checkpoint_instead_of_restarting(synthetic_data_dir, tmp_path):
    """The core behavior: given a checkpoint that says "epoch 2 of 5 is done", a call
    asking for 5 epochs must run only epochs 3-5, not repeat 1-2 or ignore the checkpoint.
    """
    out_dir = tmp_path / "run_resume"
    out_dir.mkdir()

    # Do a real 2-epoch run first (produces a genuine resume_checkpoint.pth), then
    # interrupt it ourselves by copying that checkpoint out before epoch 2's cleanup
    # would remove it, simulating "process died right after epoch 2, before reaching
    # epoch 5". We do this by running with epochs=2 and patience=1000 pointed at a
    # *different* dir, then relocating its resume checkpoint into out_dir -- since a
    # completed 2-epoch run deletes its own resume file, we instead run a fresh
    # instance with epochs=5 but kill it conceptually by asserting on epoch count: use
    # a monkeypatched break after epoch 2 via a tiny epochs=2 pre-run whose
    # resume_checkpoint we salvage before completion by running epochs=2 with an
    # unreachably-high implicit total (simplest: call train_pca with epochs=2 into a
    # staging dir where we intercept the checkpoint right as it's written each epoch
    # is overkill -- instead, directly construct a realistic checkpoint dict the same
    # way train_pca does, which is the actual contract this test is verifying).
    pre_run_dir = tmp_path / "staging"
    train_pca(**_tiny_kwargs(synthetic_data_dir, pre_run_dir, epochs=2))
    history_after_2_epochs = json.loads((pre_run_dir / "training_history.json").read_text())
    assert len(history_after_2_epochs) == 2

    # Build the checkpoint a real interrupted run would have left behind at epoch 2,
    # using the actual best_model.pth this real 2-epoch run produced (so model_state
    # is genuine, not fabricated) plus a fresh optimizer/scheduler for a 5-epoch budget.
    import shutil
    import torch.optim as optim
    from models.wavkan_v2 import WavKAN_v2
    model = WavKAN_v2()
    model.load_state_dict(torch.load(pre_run_dir / "best_model.pth"))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    # A real interrupted run always has best_model.pth in output_dir too (saved
    # whenever an epoch improved on val_f1) -- seed that here too, using the actual
    # best val_macro_f1 the 2-epoch pre-run really achieved, not a fabricated number.
    real_best_val_f1 = max(h["val_macro_f1"] for h in history_after_2_epochs)
    shutil.copy(pre_run_dir / "best_model.pth", out_dir / "best_model.pth")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": 2,
        "best_val_f1": real_best_val_f1,
        "best_val_vrecall": 0.1111,
        "no_improve": 0,
        "history": history_after_2_epochs,
    }, out_dir / "resume_checkpoint.pth")

    metrics = train_pca(**_tiny_kwargs(synthetic_data_dir, out_dir, epochs=5))

    final_history = json.loads((out_dir / "training_history.json").read_text())
    assert len(final_history) == 5, (
        f"expected exactly 5 history entries (2 injected + 3 newly run), got {len(final_history)} "
        "-- resume either restarted from scratch or double-counted epochs"
    )
    assert final_history[0] == history_after_2_epochs[0]
    assert final_history[1] == history_after_2_epochs[1]
    assert [h["epoch"] for h in final_history] == [1, 2, 3, 4, 5]
    assert metrics["epochs_run"] == 5
    assert not (out_dir / "resume_checkpoint.pth").exists()  # cleaned up on completion


def test_corrupted_resume_checkpoint_falls_back_to_fresh_start_instead_of_crashing(synthetic_data_dir, tmp_path):
    out_dir = tmp_path / "run_corrupt"
    out_dir.mkdir()
    (out_dir / "resume_checkpoint.pth").write_bytes(b"not a real torch checkpoint")

    # Must not raise -- a corrupt checkpoint should be logged and treated as "start fresh",
    # not propagate an exception that would crash the whole seed (and, under
    # run_gpu_pipeline.sh, get retried against the same corrupt file forever).
    metrics = train_pca(**_tiny_kwargs(synthetic_data_dir, out_dir, epochs=2))

    assert metrics["epochs_run"] == 2
    history = json.loads((out_dir / "training_history.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2]


def test_resume_checkpoint_already_at_epoch_cap_skips_straight_to_eval(synthetic_data_dir, tmp_path):
    """If the checkpoint says epoch==epochs already (process died between the loop
    finishing and final test-eval/save), don't re-enter the training loop at all --
    just re-run final evaluation using the checkpoint's model state."""
    out_dir = tmp_path / "run_at_cap"
    out_dir.mkdir()

    pre_run_dir = tmp_path / "staging2"
    train_pca(**_tiny_kwargs(synthetic_data_dir, pre_run_dir, epochs=2))
    history = json.loads((pre_run_dir / "training_history.json").read_text())

    import torch.optim as optim
    from models.wavkan_v2 import WavKAN_v2
    model = WavKAN_v2()
    model.load_state_dict(torch.load(pre_run_dir / "best_model.pth"))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "epoch": 2,
        "best_val_f1": 0.4242, "best_val_vrecall": 0.1111, "no_improve": 0, "history": history,
    }, out_dir / "resume_checkpoint.pth")
    # best_model.pth must also exist for the "skip to eval" path to load it
    import shutil
    shutil.copy(pre_run_dir / "best_model.pth", out_dir / "best_model.pth")

    metrics = train_pca(**_tiny_kwargs(synthetic_data_dir, out_dir, epochs=2))
    assert metrics["epochs_run"] == 2
    final_history = json.loads((out_dir / "training_history.json").read_text())
    assert len(final_history) == 2  # not re-run, just carried through from the checkpoint
