"""Regression test for src/eval_ptbxl.py's classification-report labeling bug.

Found live on the real GPU box (2026-08-17): PTB-XL genuinely has no Fusion (F) beats
(confirmed by the real process_ptbxl.py run's own class-distribution printout: only
N/S/V/Q are present). `evaluate()` used to call `classification_report(..., target_names=
class_names, labels=list(unique))` -- passing only the 4 *present* label values while
`target_names` still listed all 5 class names causes sklearn to match them positionally,
silently relabeling the real Q-class row as "F" in both the printed table and the saved
JSON (`report_dict`), while a genuine "Q" key never appears in that sub-object at all.
The separately-computed `per_class_recall` (via a fixed confusion_matrix(labels=range(5)))
was already correct and unaffected -- only classification_report's own breakdown was wrong.
Fixed by passing `labels=list(range(5))` (matching class_names index-for-index) instead of
`labels=list(unique)`, the same pattern confusion_matrix already used correctly.
"""
import json

import numpy as np
import torch

from src.eval_ptbxl import evaluate
from models.wavkan_v2 import WavKAN_v2


def _write_ptbxl_split(data_dir, y_values):
    n = len(y_values)
    np.save(data_dir / "X_test.npy", np.random.randn(n, 360).astype(np.float32))
    np.save(data_dir / "X_rr_test.npy", np.random.randn(n, 5).astype(np.float32))
    np.save(data_dir / "y_test.npy", np.array(y_values, dtype=np.int64))


def test_evaluate_correctly_labels_a_class_absent_from_the_target_dataset(tmp_path):
    """The exact real-world scenario: PTB-XL has classes {N=0, S=1, V=2, Q=4} present,
    F=3 entirely absent -- Q must be reported as "Q" with its real support, not silently
    relabeled "F"."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # 10 N, 6 S, 4 V, 0 F, 8 Q -- F (class 3) is entirely absent, matching real PTB-XL
    y_values = [0] * 10 + [1] * 6 + [2] * 4 + [4] * 8
    _write_ptbxl_split(data_dir, y_values)

    ckpt_path = tmp_path / "fake_checkpoint.pth"
    torch.save(WavKAN_v2().state_dict(), ckpt_path)

    out_path = tmp_path / "ptbxl_metrics.json"
    evaluate(str(ckpt_path), str(data_dir), str(out_path))

    with open(out_path) as f:
        report = json.load(f)

    # The real Q-class data (support=8) must appear under its own real name "Q",
    # never silently relabeled "F" (the H8-class labeling bug this test guards against).
    assert "Q" in report
    assert report["Q"]["support"] == 8
    # F is genuinely absent from this dataset -- it must show zero support, not Q's data.
    assert "F" in report
    assert report["F"]["support"] == 0
    # Present classes report their real support, unaffected by the bug.
    assert report["N"]["support"] == 10
    assert report["S"]["support"] == 6
    assert report["V"]["support"] == 4
    # per_class_recall (computed independently via a fixed-size confusion matrix) was
    # already correct before this fix -- confirm it still is, as a cross-check.
    assert report["per_class_recall"]["Q"] is not None
    assert report["per_class_recall"]["F"] is None  # no F beats -> no recall to report
