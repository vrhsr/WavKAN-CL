"""Regression tests for the two gates added after AUDIT_FINDINGS.md H49.

The 2026-09-26 Phase 10 run trained its arms on different records from the
published reference arm, and passed every check that then existed:
  * the runner's preflight checked class COUNTS, which only prove the same
    beats were selected, and enforced stale counts besides;
  * the integrator checked artifacts, seed completeness, provenance and
    local-vs-remote statistics -- all internally consistent -- and printed
    CLEAN followed by an instruction to rewrite the paper's central claim.

Gate 1 (runner, preflight 2b): the published checkpoints must reproduce their
own recorded scores on the machine's arrays. Tested here through its pure
decision rule, equivalence_verdict(), using the real magnitudes observed.
Gate 2 (integrator): a run is refused unless its training counts equal the
PhysioNet ground truth AND it carries a passed data-equivalence record.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

GT = json.load(open(REPO / "configs" / "mitbih_split_counts.json", encoding="utf-8"))


# --------------------------------------------------------------- gate 1
def _rows(test_d=0.0, val_d=0.0, seeds=("42", "101", "1001")):
    return [{"seed": s, "test_saved": 0.3569, "test_got": 0.3569 + test_d,
             "val_recorded": 0.4397, "val_got": 0.4397 + val_d} for s in seeds]


def test_gate_passes_on_exact_reproduction():
    import run_remote_experiment as R
    v = R.equivalence_verdict(_rows(), tol=0.002)
    assert v["passed"] and not v["failures"]


def test_gate_tolerates_argmax_ties_within_tolerance():
    import run_remote_experiment as R
    assert R.equivalence_verdict(_rows(test_d=0.0015, val_d=-0.0019), tol=0.002)["passed"]


def test_gate_fails_on_the_real_wrong_split_magnitude():
    """Observed 2026-09-28: a published checkpoint on the pre-H16 validation
    set scores 0.04-0.19 away from its own recorded value (mean 0.5139 vs 0.4397)."""
    import run_remote_experiment as R
    v = R.equivalence_verdict(_rows(val_d=0.0742), tol=0.002)
    assert not v["passed"]
    assert any("validation" in f for f in v["failures"])


def test_gate_fails_on_test_mismatch_even_for_val_exempt_seed():
    import run_remote_experiment as R
    v = R.equivalence_verdict(_rows(test_d=0.01, seeds=("1001",)) + _rows(seeds=("42",)),
                              tol=0.002, val_exempt=[1001])
    assert not v["passed"]
    assert any("seed 1001: test" in f for f in v["failures"])


def test_val_exempt_seed_skips_only_validation():
    """H50: seed 1001 cannot vouch for the validation arrays, but must still
    reproduce on test."""
    import run_remote_experiment as R
    rows = [{"seed": "1001", "test_saved": 0.3528, "test_got": 0.3528,
             "val_recorded": 0.4183, "val_got": 0.3945}] + _rows(seeds=("42",))
    assert not R.equivalence_verdict(rows, tol=0.002)["passed"]
    assert R.equivalence_verdict(rows, tol=0.002, val_exempt=[1001])["passed"]


def test_gate_cannot_pass_vacuously():
    import run_remote_experiment as R
    assert not R.equivalence_verdict([], tol=0.002)["passed"]
    only_exempt = _rows(seeds=("1001",))
    assert not R.equivalence_verdict(only_exempt, tol=0.002, val_exempt=[1001])["passed"], \
        "a gate with no validation evidence at all must not pass"


def test_config_does_not_relax_the_tolerance():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(open(REPO / "configs" / "final_component_ablation.yaml", encoding="utf-8"))
    eq = cfg["data_equivalence"]
    assert float(eq["tolerance_macro_f1"]) <= 0.002
    assert [int(s) for s in eq["val_exempt_seeds"]] == [1001]
    assert eq["reference_dir"] == cfg["experiment"]["reference_arm"]["dir"]
    assert eq["model_kwargs"] == {"use_pcwi": True, "use_pwam": True, "use_rr_attn": False}


def test_config_trains_a_same_job_reference_replicate():
    """Every ablation must be compared first against the published configuration
    re-trained in the same job, so a batch effect cannot pose as a component effect."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(open(REPO / "configs" / "final_component_ablation.yaml", encoding="utf-8"))
    wb = cfg["experiment"]["within_batch_reference"]
    arms = {a["name"]: a for a in cfg["arms"]}
    assert wb in arms and arms[wb]["priority"] == "essential"
    assert arms[wb]["flags"] == ["--no-rr-attn"], "the replicate must be the adopted config exactly"
    for name in ("no_pcwi", "prior_swap_pt", "no_pwam"):
        extra = [f for f in arms[name]["flags"] if f != "--no-rr-attn"]
        assert extra, f"{name} must differ from the replicate"


# --------------------------------------------------------------- gate 2
def _run_dir(tmp_path, counts, equivalence):
    d = tmp_path / "run"
    d.mkdir()
    prov = {"data_verified": {s: {"n": sum(c.values()), "class_counts": c}
                              for s, c in counts.items()}}
    if equivalence is not None:
        prov["data_equivalence"] = equivalence
    (d / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return d


def _gt_counts():
    return {s: {str(k): int(v) for k, v in GT["counts"][s].items()} for s in ("train", "val", "test")}


PASSED = {"passed": True, "n_seeds": 20, "max_abs_test_diff": 0.0, "max_abs_val_diff": 0.0,
          "failures": []}


@pytest.fixture
def integ(monkeypatch):
    import integrate_remote_results as I
    monkeypatch.setattr(I, "problems", [])
    monkeypatch.setattr(I, "say", lambda *a, **k: None)
    monkeypatch.setattr(I, "head", lambda *a, **k: None)
    return I


def test_integrator_rejects_the_real_h49_run_shape(integ, tmp_path):
    """Stale training counts and no equivalence record: exactly the 2026-09-26 run."""
    c = _gt_counts()
    c["train"] = {str(k): v for k, v in enumerate([36396, 773, 3150, 399, 8])}
    c["val"] = {str(k): v for k, v in enumerate([9443, 170, 638, 15, 0])}
    integ.verify_comparability(_run_dir(tmp_path, c, None), {})
    msgs = " | ".join(integ.problems)
    assert "different records" in msgs
    assert "no data-equivalence record" in msgs


def test_integrator_rejects_correct_counts_without_equivalence(integ, tmp_path):
    """Counts alone are not enough: they prove selection, not the arrays."""
    integ.verify_comparability(_run_dir(tmp_path, _gt_counts(), None), {})
    assert any("data-equivalence" in p for p in integ.problems)


def test_integrator_rejects_a_failed_equivalence_gate(integ, tmp_path):
    failed = dict(PASSED, passed=False, failures=["seed 42: validation ..."])
    integ.verify_comparability(_run_dir(tmp_path, _gt_counts(), failed), {})
    assert any("did not pass" in p for p in integ.problems)


def test_integrator_accepts_a_comparable_run(integ, tmp_path):
    integ.verify_comparability(_run_dir(tmp_path, _gt_counts(), PASSED), {})
    assert integ.problems == []


# --------------------------------------------------------------- the Tee bug
def test_runner_tee_behaves_as_the_stream_it_wraps(tmp_path):
    """Found 2026-09-28 by running preflight 2b end to end: the runner replaces
    sys.stdout with Tee, and importing src/train_pca.py (as gate 2b must) runs
    its H48 console guard, which read sys.stdout.encoding -- absent on a bare
    Tee -- so the gate crashed before evaluating one checkpoint."""
    import run_remote_experiment as R
    t = R.Tee(tmp_path / "run.log")
    assert isinstance(t.encoding, str) and t.encoding
    t.reconfigure(errors="replace")        # must not raise
    t.write("café ✓\n")
    t.flush()
    assert "café" in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_trainer_imports_with_the_runner_tee_as_stdout(tmp_path):
    """The exact failing sequence, in a fresh interpreter so module state is
    real: install the runner's Tee as sys.stdout, then import the trainer."""
    import os
    import subprocess
    code = (
        "import sys, pathlib; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "import run_remote_experiment as R\n"
        "sys.stdout = R.Tee(pathlib.Path(%r))\n"
        "import src.train_pca\n"
        "print('imported ok')\n"
    ) % (str(REPO), str(REPO / "src"), str(tmp_path / "run.log"))
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=300, env=env, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "AttributeError" not in r.stderr


# --------------------------------------------------------------- resume guard
_FP = {"data_sha256": {"train": {"X": "a"}, "val": {"X": "b"}, "test": {"X": "c"}},
       "seeds": [42, 101], "arms": {"no_pcwi": ["--no-pcwi", "--no-rr-attn"]}}


def test_resume_refused_onto_a_run_without_fingerprint():
    """The 2026-09-28 box state: the invalid run's seeds were complete and it
    had no fingerprint. --allow-resume would have counted them as done."""
    import run_remote_experiment as R
    assert "predates" in R.resume_verdict(None, _FP)


def test_resume_refused_on_different_data():
    import run_remote_experiment as R
    other = dict(_FP, data_sha256={"train": {"X": "zzz"}, "val": {"X": "b"}, "test": {"X": "c"}})
    assert "different data" in R.resume_verdict(other, _FP)


def test_resume_refused_on_different_seeds_or_flags():
    import run_remote_experiment as R
    assert R.resume_verdict(dict(_FP, seeds=[42]), _FP)
    assert R.resume_verdict(dict(_FP, arms={"no_pcwi": ["--no-pcwi"]}), _FP)


def test_resume_allowed_for_the_same_job():
    import run_remote_experiment as R
    assert R.resume_verdict(dict(_FP), _FP) == ""


def test_output_gate_does_not_suggest_resume_for_a_foreign_run(tmp_path, monkeypatch):
    import run_remote_experiment as R
    monkeypatch.setattr(R, "REPO", tmp_path)
    msgs = []
    monkeypatch.setattr(R, "say", lambda m="": msgs.append(m))
    d = tmp_path / "results" / "x" / "no_pcwi" / "seed_42"
    d.mkdir(parents=True)
    (d / "test_metrics.json").write_text("{}")
    cfg = {"experiment": {"output_root": "results/x"}, "seeds": [42], "arms": []}
    with pytest.raises(SystemExit):
        R.check_output(cfg, allow_resume=False)
    text = "\n".join(msgs)
    assert "Do NOT use --allow-resume" in text
    with pytest.raises(SystemExit):
        R.check_output(cfg, allow_resume=True, data={s: {"sha256": {}} for s in ("train", "val", "test")})
