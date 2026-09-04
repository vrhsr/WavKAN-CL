"""Regression test for AUDIT_FINDINGS.md H38.

src/rr_ablation.py's per-arm metric dict previously returned
    "macro_f1": recall_score(y_true, y_pred, average="macro")
i.e. it labelled macro-averaged RECALL as macro-F1. On this dataset's real
class distribution the two differ substantially (the script's saved baseline
read 0.3759 where the same checkpoints' real DS2 Macro-F1 is 0.3228), so a
reader reconciling the RR-ablation artefact against test_metrics.json would
find an unexplained discrepancy.

These tests pin the distinction: they construct predictions for which
macro-recall and macro-F1 are provably unequal and assert that each key holds
the metric its name claims. The first test fails against the pre-fix code.
"""
import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

rr_ablation = pytest.importorskip(
    "src.rr_ablation",
    reason="requires torch and the src package on sys.path",
)


def _imbalanced_case():
    """5-class problem, heavy majority class, imperfect precision on the
    minority classes -- the regime where macro-recall and macro-F1 diverge."""
    rng = np.random.default_rng(0)
    y_true = np.concatenate([
        np.zeros(4000, dtype=int),   # N
        np.ones(120, dtype=int),     # S
        np.full(300, 2, dtype=int),  # V
        np.full(40, 3, dtype=int),   # F
        np.full(5, 4, dtype=int),    # Q
    ])
    y_pred = y_true.copy()
    # Over-predict the minority classes: recall stays high, precision drops,
    # so macro-F1 falls well below macro-recall.
    maj = np.flatnonzero(y_true == 0)
    y_pred[rng.choice(maj, 900, replace=False)] = 1
    y_pred[rng.choice(maj, 600, replace=False)] = 2
    return y_true, y_pred


def test_macro_recall_and_macro_f1_are_distinct_in_this_regime():
    """Guards the test itself: if these coincided, the test below would be vacuous."""
    y_true, y_pred = _imbalanced_case()
    mr = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mf = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert abs(mr - mf) > 0.05, (
        "test fixture is too easy: macro-recall %.4f vs macro-F1 %.4f" % (mr, mf))


def test_metric_keys_hold_the_metric_they_name(monkeypatch):
    """macro_f1 must be F1 and macro_recall must be recall.

    rr_ablation.inference_with_mask() runs a model over a DataLoader; we bypass
    that by monkeypatching its prediction collection so only the metric
    computation is exercised.
    """
    y_true, y_pred = _imbalanced_case()

    fn = getattr(rr_ablation, "inference_with_mask", None)
    if fn is None:
        pytest.skip("rr_ablation.inference_with_mask not present under that name")

    import torch

    class _Loader:
        """Yields one batch shaped the way rr_ablation expects."""
        def __iter__(self):
            yield (torch.zeros(len(y_true), 360),
                   torch.zeros(len(y_true), 5),
                   torch.as_tensor(y_true))

        def __len__(self):
            return 1

    class _Model:
        """Returns logits that argmax to y_pred."""
        def eval(self):
            return self

        def to(self, *a, **k):
            return self

        def __call__(self, x, xrr):
            logits = torch.full((len(y_pred), 5), -10.0)
            logits[torch.arange(len(y_pred)), torch.as_tensor(y_pred)] = 10.0
            return logits

    out = fn(_Model(), _Loader(), torch.device("cpu"), mask_pos=-1)

    assert "macro_recall" in out, (
        "macro_recall key missing -- H38 regression: macro-averaged recall is "
        "being reported under some other name")
    expected_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    expected_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    assert out["macro_recall"] == pytest.approx(expected_recall, abs=1e-9)
    if "macro_f1" in out:
        assert out["macro_f1"] == pytest.approx(expected_f1, abs=1e-9), (
            "H38 regression: 'macro_f1' is %.4f but real macro-F1 is %.4f "
            "(macro-recall is %.4f) -- the key is holding recall again"
            % (out["macro_f1"], expected_f1, expected_recall))
    assert out["s_recall"] == pytest.approx(
        recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0), abs=1e-9)
    assert out["v_recall"] == pytest.approx(
        recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0), abs=1e-9)


def test_saved_report_key_is_not_mislabelled():
    """The checked-in real report must not reintroduce the wrong key name."""
    import json
    import os
    path = os.path.join("results", "rr_ablation_real", "rr_ablation_report.json")
    if not os.path.exists(path):
        pytest.skip("real RR-ablation report not present in this checkout")
    baseline = json.load(open(path))["baseline"]
    assert "macro_recall" in baseline, "H38: baseline lost its macro_recall key"
    if "macro_f1" in baseline:
        # If a macro_f1 is ever added back it must not equal the macro-recall
        # value, which is what the mislabelling produced.
        assert baseline["macro_f1"]["mean"] != baseline["macro_recall"]["mean"], (
            "H38 regression: macro_f1 and macro_recall hold the same value, "
            "so one of them is mislabelled again")
