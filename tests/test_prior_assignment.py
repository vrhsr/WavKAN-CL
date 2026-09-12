"""
Regression tests for the permuted-prior control (PCWI morphology-specificity).

The control exists to separate two explanations of PCWI's measured benefit:
  (A) the priors carry ECG morphology information, versus
  (B) three groups at three different scales beat one isotropic scale.

For the control to isolate (A), the permuted variants must differ from the
physiological assignment in *exactly one* respect: which group receives which
prior. These tests pin that property, because a control that accidentally
changes capacity, block structure or the set of prior values would not answer
the question and would be worse than not running it at all.
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wavkan_pcwi import ECG_PRIORS, PRIOR_ASSIGNMENTS, PCWIWavKANLinear  # noqa: E402
from models.wavkan_v2 import WavKAN_v2  # noqa: E402


def _group_stats(layer, group):
    c0, c1 = ECG_PRIORS[group]["channels"]
    return (layer.translation.data[c0:c1].mean().item(),
            layer.scale.data[c0:c1].mean().item())


def test_default_is_physiological_and_unchanged():
    """The default must reproduce the published initialisation exactly."""
    torch.manual_seed(0)
    layer = PCWIWavKANLinear(360, 64)
    assert layer.prior_assignment == "physiological"
    for g in ("QRS", "P", "T"):
        mu, gamma = _group_stats(layer, g)
        assert mu == pytest.approx(ECG_PRIORS[g]["mu_center"], abs=0.01)
        assert gamma == pytest.approx(ECG_PRIORS[g]["gamma"], abs=0.01)


def test_swap_pt_exchanges_only_p_and_t():
    """swap_pt is the primary control: P and T blocks are both 16 channels, so
    block size is exactly preserved and only the mapping changes."""
    torch.manual_seed(0)
    layer = PCWIWavKANLinear(360, 64, prior_assignment="swap_pt")
    qrs_mu, qrs_g = _group_stats(layer, "QRS")
    p_mu, p_g = _group_stats(layer, "P")
    t_mu, t_g = _group_stats(layer, "T")

    assert qrs_mu == pytest.approx(ECG_PRIORS["QRS"]["mu_center"], abs=0.01)
    assert qrs_g == pytest.approx(ECG_PRIORS["QRS"]["gamma"], abs=0.01)
    # P block now carries T's prior and vice versa.
    assert p_mu == pytest.approx(ECG_PRIORS["T"]["mu_center"], abs=0.01)
    assert p_g == pytest.approx(ECG_PRIORS["T"]["gamma"], abs=0.01)
    assert t_mu == pytest.approx(ECG_PRIORS["P"]["mu_center"], abs=0.01)
    assert t_g == pytest.approx(ECG_PRIORS["P"]["gamma"], abs=0.01)


def test_cyclic_permutes_all_three():
    torch.manual_seed(0)
    layer = PCWIWavKANLinear(360, 64, prior_assignment="cyclic")
    for block, source in PRIOR_ASSIGNMENTS["cyclic"].items():
        mu, gamma = _group_stats(layer, block)
        assert mu == pytest.approx(ECG_PRIORS[source]["mu_center"], abs=0.01)
        assert gamma == pytest.approx(ECG_PRIORS[source]["gamma"], abs=0.01)


def test_controls_preserve_the_multiset_of_prior_values():
    """The whole point: the controls must reuse the SAME prior values, so any
    performance difference cannot be attributed to different scales being
    available. Only the group-to-prior mapping may differ."""
    for name in PRIOR_ASSIGNMENTS:
        torch.manual_seed(0)
        layer = PCWIWavKANLinear(360, 64, prior_assignment=name)
        got = sorted(round(_group_stats(layer, g)[1], 3) for g in ("QRS", "P", "T"))
        want = sorted(round(ECG_PRIORS[g]["gamma"], 3) for g in ("QRS", "P", "T"))
        assert got == want, f"{name} changed the set of dilations: {got} != {want}"


def test_controls_do_not_change_parameter_count():
    """A capacity change would confound the comparison."""
    counts = set()
    for name in PRIOR_ASSIGNMENTS:
        m = WavKAN_v2(use_rr_attn=False, prior_assignment=name)
        counts.add(sum(p.numel() for p in m.parameters() if p.requires_grad))
    assert len(counts) == 1, f"parameter count varies across controls: {counts}"
    assert counts.pop() == 153045


def test_permuted_variants_actually_differ_from_physiological():
    """A control that silently equals the default would give a false null."""
    torch.manual_seed(0)
    base = PCWIWavKANLinear(360, 64, prior_assignment="physiological")
    for name in ("swap_pt", "cyclic"):
        torch.manual_seed(0)
        alt = PCWIWavKANLinear(360, 64, prior_assignment=name)
        assert not torch.allclose(base.scale.data, alt.scale.data), \
            f"{name} produced the same dilations as physiological"


def test_invalid_assignment_rejected():
    with pytest.raises(ValueError, match="prior_assignment"):
        PCWIWavKANLinear(360, 64, prior_assignment="nonsense")


def test_forward_pass_works_for_every_assignment():
    for name in PRIOR_ASSIGNMENTS:
        m = WavKAN_v2(use_rr_attn=False, prior_assignment=name).eval()
        with torch.no_grad():
            out = m(torch.randn(4, 360), torch.rand(4, 5))
        assert out.shape == (4, 5)
        assert torch.isfinite(out).all()


def test_trainer_exposes_the_flag():
    """The control is useless if it cannot be selected from the command line."""
    src = (Path(__file__).resolve().parents[1] / "src" / "train_pca.py").read_text(
        encoding="utf-8")
    assert "--prior-assignment" in src
    assert "prior_assignment = args.prior_assignment" in src
    # and it must be recorded in the saved metrics, or an arm is unidentifiable
    assert '"prior_assignment": prior_assignment' in src
