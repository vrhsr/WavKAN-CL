"""Regression tests for AUDIT_FINDINGS.md H49.

The manuscript's class-distribution table and the Phase 10 GPU config both
carried per-split counts from an extraction made before the H16 fix, when
process_data.py hardcoded validation records 215/220/223/230. Every published
arm was trained on split.py's split (validation 208/209/223/230) -- proven on
2026-09-28 by re-evaluating their checkpoints, which reproduce their own
recorded validation scores only on that split. The stale counts then made the
Phase 10 preflight reject correct data, and the box was pushed onto the old
split, invalidating the whole run.

These tests pin every copy of the counts to one ground truth,
configs/mitbih_split_counts.json, which src/derive_split_counts.py derives from
the PhysioNet annotation files. None of them needs network access or data.
"""
import ast
import io
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GT = json.load(open(REPO / "configs" / "mitbih_split_counts.json", encoding="utf-8"))
STALE_TRAIN = [36396, 773, 3150, 399, 8]
STALE_VAL = [9443, 170, 638, 15, 0]


def _split_module():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    import split
    return split


def test_ground_truth_records_are_split_py():
    s = _split_module()
    assert GT["records"]["train"] == sorted(s.TRAIN_RECORDS)
    assert GT["records"]["val"] == sorted(s.VAL_RECORDS)
    assert GT["records"]["test"] == sorted(s.TEST_RECORDS)
    assert sorted(s.VAL_RECORDS) == ["208", "209", "223", "230"]


def test_ground_truth_is_internally_consistent():
    per = GT["per_record"]
    for split, recs in GT["records"].items():
        summed = [sum(per[r][k] for r in recs) for k in range(5)]
        assert summed == [GT["counts"][split][str(k)] for k in range(5)], split
        assert GT["totals"][split] == sum(summed), split
    # the three partitions are disjoint and DS1+DS2 is the whole of both lists
    tr, va, te = (set(GT["records"][k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_published_split_counts():
    """The split the published arms were trained on (verified forensically)."""
    assert [GT["counts"]["train"][str(k)] for k in range(5)] == [37337, 485, 2321, 28, 6]
    assert [GT["counts"]["val"][str(k)] for k in range(5)] == [8502, 458, 1467, 386, 2]
    assert [GT["counts"]["test"][str(k)] for k in range(5)] == [44232, 1837, 3220, 388, 7]
    assert GT["totals"] == {"train": 40177, "val": 10815, "test": 49684}


def test_stale_counts_are_the_pre_h16_split_not_the_published_one():
    """Documents H49: the stale numbers are a real split, just not the one used."""
    st = GT["stale_pre_h16_for_reference_only"]
    assert st["val_records"] == ["215", "220", "223", "230"]
    assert st["train"] == STALE_TRAIN and st["val"] == STALE_VAL
    assert st["train"] != [GT["counts"]["train"][str(k)] for k in range(5)]


def test_selection_rule_matches_process_data_py():
    """derive_split_counts.py must apply exactly process_data.py's rule. Parsed
    from source rather than imported, since process_data imports neurokit2 and
    creates directories at import time."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    import derive_split_counts as D
    src = io.open(REPO / "src" / "process_data.py", encoding="utf-8").read()
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "AAMI_MAP":
                found[name] = ast.literal_eval(node.value)
    assert found.get("AAMI_MAP") == D.AAMI_MAP
    assert "PRE_SAMPLES = int(0.25 * FS)" in src and "POST_SAMPLES = int(0.75 * FS)" in src
    assert "FS = 360" in src
    assert (D.PRE_SAMPLES, D.POST_SAMPLES) == (90, 270)
    assert "r - PRE_SAMPLES < 0 or r + POST_SAMPLES > len(ecg_clean)" in src, \
        "process_data.py's edge rule changed; update derive_split_counts.py to match"
    assert "from split import" in src, "process_data.py no longer takes its split from split.py"


def test_phase10_config_counts_equal_ground_truth():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(open(REPO / "configs" / "final_component_ablation.yaml", encoding="utf-8"))
    for split in ("train", "val", "test"):
        exp = {int(k): int(v) for k, v in cfg["data"]["expect_class_counts"][split].items()}
        want = {int(k): int(v) for k, v in GT["counts"][split].items()}
        assert {k: exp.get(k, 0) for k in want} == want, split
        assert cfg["data"]["expect_beats"][split] == GT["totals"][split], split


def test_manuscript_class_table_equals_ground_truth():
    tex = io.open(REPO / "Submission_JBHI" / "ieee_manuscript_v2.tex", encoding="utf-8").read()
    tab = tex.split(r"\label{tab:class_dist}")[1].split(r"\end{tabular}")[0]
    num = lambda s: int(re.sub(r"[^0-9]", "", s))
    for i, cls in enumerate(["N", "S", "V", "F", "Q"]):
        row = [ln for ln in tab.split("\n") if ln.strip().startswith(r"\textbf{%s}" % cls)][0]
        c = [x.strip() for x in row.split("&")]
        assert (num(c[2]), num(c[3]), num(c[4])) == tuple(
            GT["counts"][s][str(i)] for s in ("train", "val", "test")), cls


def test_manuscript_prose_record_lists_equal_ground_truth():
    tex = io.open(REPO / "Submission_JBHI" / "ieee_manuscript_v2.tex", encoding="utf-8").read()
    prose = tex.split(r"\subsection{Inter-Patient Split}")[1].split(r"\subsection")[0]
    LIST = r"((?:\d{3}, )*\d{3},? and \d{3})"
    for split, pat in (("train", r"training uses the (\d+) records " + LIST),
                       ("val", r"validation the (\d+) records " + LIST),
                       ("test", r"held out, the (\d+) records " + LIST)):
        m = re.search(pat, prose)
        assert m, f"{split} list not found in the Inter-Patient Split prose"
        recs = sorted(re.findall(r"\d{3}", m.group(2)))
        assert int(m.group(1)) == len(recs), split
        assert recs == GT["records"][split], split


def test_runner_refuses_a_config_with_stale_counts(monkeypatch):
    """The preflight must reject a config that disagrees with the ground truth,
    rather than enforce it against correct data (the H49 failure mode)."""
    import sys
    sys.path.insert(0, str(REPO))
    import run_remote_experiment as R
    monkeypatch.setattr(R, "say", lambda *a, **k: None)
    cfg = {"data": {"expect_beats": {"train": 40726, "val": 10266, "test": 49684},
                    "expect_class_counts": {
                        "train": dict(zip(range(5), STALE_TRAIN)),
                        "val": dict(zip(range(5), STALE_VAL)),
                        "test": {0: 44232, 1: 1837, 2: 3220, 3: 388, 4: 7}}}}
    with pytest.raises(SystemExit) as e:
        R.check_counts_against_ground_truth(cfg)
    assert e.value.code == R.EXIT_PREFLIGHT
    good = {"data": {"expect_beats": dict(GT["totals"]),
                     "expect_class_counts": {s: {int(k): v for k, v in GT["counts"][s].items()}
                                             for s in ("train", "val", "test")}}}
    R.check_counts_against_ground_truth(good)   # must not raise
