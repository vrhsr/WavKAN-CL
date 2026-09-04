"""Regression test for AUDIT_FINDINGS.md H48.

src/train_pca.py prints non-ASCII characters in several places: box-drawing
rules, Greek letters in the schedule description, and status glyphs on the
lines that report resume, early stopping and completion. On a Windows console
using the cp1252 code page each of those raised UnicodeEncodeError and killed
the run.

Two of the offending prints sit inside the crash-resume handler, which is the
worst possible placement: the code path whose whole purpose is to let a long
multi-seed job survive an interruption was itself guaranteed to abort on that
console, turning a recoverable resume into a dead seed. It was found by running
the Phase 10 remote pipeline locally, where every seed failed with rc=1 before
completing an epoch.

This is the third occurrence of the same bug class in this repository
(wavelet_alignment_score.py -- C19 and H30 -- and export_quantize.py), so the
tests below cover every training/analysis entry point that prints non-ASCII,
not just the one that failed.
"""
import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Scripts that print non-ASCII and are invoked directly as __main__.
GUARDED_SCRIPTS = [
    "src/train_pca.py",
    "src/wavelet_alignment_score.py",
    "src/export_quantize.py",
]


def _source(rel):
    p = REPO / rel
    if not p.exists():
        pytest.skip(f"{rel} not present in this checkout")
    return p, io.open(p, encoding="utf-8").read()


def _non_ascii_lines(text):
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        if any(ord(c) > 127 for c in line):
            out.append(i)
    return out


@pytest.mark.parametrize("rel", GUARDED_SCRIPTS)
def test_script_with_non_ascii_output_has_an_encoding_guard(rel):
    """Any script printing non-ASCII must reconfigure stdout before doing so."""
    path, src = _source(rel)
    if not _non_ascii_lines(src):
        pytest.skip(f"{rel} contains no non-ASCII text")

    # The guard: reconfigure stdout to UTF-8 when the console reports otherwise.
    has_guard = bool(
        re.search(r"sys\.stdout\.reconfigure\s*\(\s*encoding\s*=\s*[\"']utf-8[\"']", src))
    assert has_guard, (
        f"{rel} prints non-ASCII (lines {_non_ascii_lines(src)[:8]}...) but has no "
        f"sys.stdout.reconfigure(encoding='utf-8') guard. On a cp1252 Windows "
        f"console this raises UnicodeEncodeError and aborts the run (H48).")

    # The guard must come before the first non-ASCII line, or it is useless.
    guard_line = next(
        i for i, line in enumerate(src.split("\n"), 1)
        if "sys.stdout.reconfigure" in line)
    first_na = min(
        i for i in _non_ascii_lines(src)
        # ignore the guard's own explanatory comment block
        if i > guard_line or "reconfigure" not in src.split("\n")[i - 1])
    # Non-ASCII inside the module docstring/comments is never printed, so only
    # require ordering against non-ASCII that appears in an actual statement.
    lines = src.split("\n")
    printed_na = [
        i for i in _non_ascii_lines(src)
        if ("print" in lines[i - 1] or 'f"' in lines[i - 1] or "f'" in lines[i - 1])
        and not lines[i - 1].lstrip().startswith("#")
    ]
    if printed_na:
        assert guard_line < min(printed_na), (
            f"{rel}: the encoding guard is at line {guard_line} but non-ASCII is "
            f"printed at line {min(printed_na)} -- the guard must precede any output")


def test_train_pca_runs_under_a_cp1252_stdout():
    """End-to-end: importing and --help-ing train_pca.py must not crash when the
    child process's stdout cannot encode the characters the module prints.

    Reproduces the real failure by forcing the child's IO encoding to cp1252,
    which is what a default Windows console gives. Fails against the pre-fix
    code, where this aborted with UnicodeEncodeError.
    """
    path, _ = _source("src/train_pca.py")
    env = {
        "PYTHONIOENCODING": "cp1252",
        "PYTHONPATH": str(REPO),
        "PATH": __import__("os").environ.get("PATH", ""),
        "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
    }
    r = subprocess.run([sys.executable, str(path), "--help"],
                       cwd=REPO, capture_output=True, text=True,
                       timeout=180, env=env)
    assert "UnicodeEncodeError" not in (r.stderr or ""), (
        "train_pca.py raised UnicodeEncodeError under a cp1252 stdout (H48):\n"
        + (r.stderr or "")[-1500:])
    assert r.returncode == 0, (
        f"train_pca.py --help exited {r.returncode} under cp1252 stdout:\n"
        + (r.stderr or "")[-1500:])


def test_resume_handler_prints_are_covered_by_the_guard():
    """The two prints in the crash-resume handler were the damaging ones: they
    turned a recoverable resume into a dead seed. Pin that they sit after the
    guard and that the handler still re-raises nothing silently."""
    path, src = _source("src/train_pca.py")
    assert "resume_checkpoint.pth" in src, "resume feature missing from train_pca.py"
    guard_line = next(
        (i for i, line in enumerate(src.split("\n"), 1)
         if "sys.stdout.reconfigure" in line), None)
    assert guard_line is not None, "no encoding guard in train_pca.py (H48)"
    resume_lines = [i for i, line in enumerate(src.split("\n"), 1)
                    if "Could not load" in line or "Resumed from" in line]
    assert resume_lines, "resume-handler status prints not found"
    assert guard_line < min(resume_lines), (
        f"encoding guard at line {guard_line} must precede the resume-handler "
        f"prints at lines {resume_lines} (H48)")
