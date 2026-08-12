"""Regression tests for the inter-patient DS1/DS2 record split.

Added per AUDIT_FINDINGS.md (Phase 2 finding, "Zero-automated-coverage note"): this is
the single most safety-critical guarantee in the codebase -- the paper's "strict
inter-patient evaluation" claim depends entirely on TRAIN/VAL/TEST never sharing a
record -- but it had zero automated coverage. src/split.py's own disjointness
assertions only ran when the file was executed directly (`python src/split.py`), never
under pytest/CI, and neither of the repo's other test-like files touches this at all.

Record lists are parsed directly out of the source files rather than imported, because
several of the modules being cross-checked here (process_data.py, process_extended_rr.py,
process_sequence.py) have import-time side effects (os.makedirs) and require optional
dependencies (neurokit2) that aren't even listed in requirements.txt -- importing them
just to read a list of record IDs would make this test needlessly fragile.
"""
import ast
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _extract_list_literal(file_path, var_name):
    """Pull a top-level `VAR_NAME = [...]` list literal out of a source file's text."""
    source = (REPO_ROOT / file_path).read_text(encoding="utf-8")
    match = re.search(rf"^{var_name}\s*=\s*(\[[^\]]*\])", source, re.MULTILINE | re.DOTALL)
    assert match, f"Could not find `{var_name} = [...]` in {file_path}"
    return ast.literal_eval(match.group(1))


# --- src/split.py: the canonical split ---
SPLIT_PY_DS1 = _extract_list_literal("src/split.py", "DS1_RECORDS")
SPLIT_PY_VAL = _extract_list_literal("src/split.py", "VAL_RECORDS")
SPLIT_PY_TEST = _extract_list_literal("src/split.py", "TEST_RECORDS")
SPLIT_PY_TRAIN = [r for r in SPLIT_PY_DS1 if r not in SPLIT_PY_VAL]


def test_split_py_train_val_test_disjoint():
    """No MIT-BIH record may appear in more than one of TRAIN/VAL/TEST.

    This is the exact guarantee the manuscript's "strict inter-patient DS1/DS2
    protocol, no patient overlap between train and test" claim rests on.
    """
    train, val, test = set(SPLIT_PY_TRAIN), set(SPLIT_PY_VAL), set(SPLIT_PY_TEST)
    assert train.isdisjoint(val), f"Leakage between Train and Val: {train & val}"
    assert train.isdisjoint(test), f"Leakage between Train and Test: {train & test}"
    assert val.isdisjoint(test), f"Leakage between Val and Test: {val & test}"


def test_split_py_train_val_partitions_ds1_exactly():
    """TRAIN_RECORDS + VAL_RECORDS must reconstruct DS1_RECORDS with no drops or dupes."""
    assert set(SPLIT_PY_TRAIN) | set(SPLIT_PY_VAL) == set(SPLIT_PY_DS1)
    assert len(SPLIT_PY_TRAIN) + len(SPLIT_PY_VAL) == len(SPLIT_PY_DS1)


def test_records_201_and_202_are_split_across_ds1_ds2():
    """Records 201 and 202 (same underlying subject, per the manuscript's own text and
    AUDIT_FINDINGS.md LOW-14) must land in different sets -- this is the specific
    same-patient pair the de Chazal protocol is known to separate."""
    assert "201" in SPLIT_PY_DS1
    assert "202" in SPLIT_PY_TEST


@pytest.mark.parametrize("file_path", [
    "src/process_data.py",
    "src/process_extended_rr.py",
    "src/process_sequence.py",
])
def test_alternate_pipelines_now_import_the_canonical_split(file_path):
    """AUDIT_FINDINGS.md H16 (FIXED 2026-08-12): process_data.py / process_extended_rr.py /
    process_sequence.py used to each independently hardcode their own VAL_RECORDS
    (215/220/223/230), diverging from src/split.py's canonical list (208/209/223/230) --
    the one the manuscript's own Table 1 actually describes. They now import from
    split.py instead of re-hardcoding, which is what this test verifies (via source text,
    not a full import, since these modules have import-time side effects and optional
    dependencies -- see module docstring). If someone reverts to a hardcoded literal,
    this test fails and forces a deliberate decision, not a silent regression.
    """
    source = (REPO_ROOT / file_path).read_text(encoding="utf-8")
    assert re.search(r"from\s+split\s+import\s+.*VAL_RECORDS", source), (
        f"{file_path} no longer imports VAL_RECORDS from split.py -- "
        "did it revert to hardcoding its own list? See AUDIT_FINDINGS.md H16."
    )
    assert not re.search(r"^VAL_RECORDS\s*=\s*\[", source, re.MULTILINE), (
        f"{file_path} has a hardcoded VAL_RECORDS literal again -- "
        "see AUDIT_FINDINGS.md H16."
    )
