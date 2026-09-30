"""
Source-level integrity checks on the manuscript that no other tool catches.

Motivation: a single stray carriage return inside "\\ref" split the command in
the submitted source, so the compiled PDF printed the literal string
"S-efsec:results" in the body of Section II. LaTeX reported ZERO errors,
`check_manuscript.py` reported all references resolved, and grep could not see
it, because a lone CR is a line terminator for TeX but not for any of those
tools. The defect survived every existing check and was only found by reading
the text extracted from the rendered PDF.

These tests operate on raw bytes for exactly that reason.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# Repointed 2026-09-30: the live manuscript is Submission_Array/manuscript.tex
# (Submission_JBHI/ is superseded, see its SUPERSEDED.md).
TEX = REPO / "Submission_Array" / "manuscript.tex"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="manuscript not present")


def _raw() -> bytes:
    return TEX.read_bytes()


def test_no_lone_carriage_returns():
    """A CR not followed by LF terminates a line for TeX while remaining
    invisible to grep and to the compiler's error output."""
    b = _raw()
    lone = [i for i in range(len(b))
            if b[i] == 13 and (i + 1 >= len(b) or b[i + 1] != 10)]
    ctx = [b[max(0, i - 40):i + 25].decode("utf-8", "replace") for i in lone[:5]]
    assert not lone, f"{len(lone)} lone CR(s); context: {ctx}"


def test_line_ending_counts_are_consistent():
    """CR and LF counts must match in a CRLF file; a mismatch means a stray
    bare CR or bare LF, which is how the defect above arose."""
    b = _raw()
    cr, lf = b.count(b"\r"), b.count(b"\n")
    assert cr in (0, lf), f"inconsistent line endings: {cr} CR vs {lf} LF"


def test_no_control_characters_in_body():
    """Anything outside tab/CR/LF in a LaTeX source is a transcription
    accident and will either break a command or print as garbage."""
    b = _raw()
    bad = {c for c in b if c < 32 and c not in (9, 10, 13)}
    assert not bad, f"control bytes present: {sorted(bad)}"


def test_no_orphaned_reference_fragments():
    """Catches a cross-reference command split by a stray line terminator.

    A well-formed '{sec:...}' / '{tab:...}' / '{fig:...}' / '{eq:...}' argument
    is always preceded by one of the reference commands. If a terminator split
    the command, the brace group survives with a bare fragment before it --
    which is precisely what produced 'S-efsec:results' in the rendered PDF.
    """
    text = _raw().decode("utf-8")
    commands = ("ref", "eqref", "autoref", "label", "cite", "Cref", "cref")
    offenders = []
    for m in re.finditer(r"\{(?:sec|tab|fig|eq|alg):[^}]*\}", text):
        before = text[:m.start()]
        if not any(before.endswith("\\" + c) for c in commands):
            offenders.append(text[max(0, m.start() - 25):m.end()])
    assert not offenders, f"orphaned reference argument(s): {offenders[:5]}"


def test_known_broken_reference_stays_fixed():
    """Regression pin for the exact defect that reached the compiled PDF."""
    b = _raw()
    assert b.count(b"\\S\\ref{sec:results}") >= 1, \
        "the sec:results reference is not well-formed"
    assert b"\\Sef{" not in b, "the malformed \\Sef form has reappeared"


def test_no_unescaped_percent_in_bib_note_fields():
    """H35/H39 pin: bib note fields broke the build once and printed internal
    audit history into the reference list once."""
    bib = REPO / "Submission_Array" / "references.bib"
    if not bib.exists():
        pytest.skip("references.bib absent")
    text = bib.read_text(encoding="utf-8", errors="replace")
    assert "note = {" not in text.replace("note = {}", ""), \
        "note fields are typeset by the bibliography style; keep provenance out of the .bib"
