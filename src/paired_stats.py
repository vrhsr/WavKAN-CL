"""
paired_stats.py -- the canonical paired-statistics helpers for this project.

Added 2026-09-03 to close AUDIT_FINDINGS.md H8, which recorded four distinct
statistical-implementation defects spread across three scripts that each rolled
their own version of the same test:

  H8(a) generate_all_figures.py paired Wilcoxon inputs by checking equal LIST
        LENGTH only, so two arms that had each dropped a *different* seed to the
        same resulting count would be silently mispaired.
  H8(b) statistical_validation.py truncated both arms to min(len) with no seed
        alignment whatsoever.
  H8(d) generate_all_figures.py used a ONE-SIDED test hardcoded to the
        alternative "ours > baseline", while metrics_full.py -- the function the
        live pipeline actually calls -- used two-sided. Same nominal test, a
        model-favouring alternative hypothesis depending on which script ran.

The fix is one implementation with the pairing done by seed identity, not by
position or length, and one sign convention stated in the docstring. Callers
pass dicts keyed by seed so mispairing is structurally impossible: there is no
positional API to misuse.

SIGN CONVENTION (matches the manuscript, which states it explicitly):
    positive Cohen's d favours arm A -- the proposed model, the adopted
    configuration, or the treatment arm. Callers must pass the arm they want a
    positive sign for as `a`.

Statistical choices, and why:
  * Two-sided Wilcoxon signed-rank. Non-parametric, so it makes no normality
    assumption about per-seed differences, and two-sided because we are testing
    whether the arms differ, not confirming a pre-committed direction.
  * Cohen's d for paired data, d = mean(diff) / sd(diff), which is the paired
    (within-subject) standardiser, not the pooled between-group one.
  * Holm-Bonferroni step-down within an explicitly declared family. Holm is
    uniformly more powerful than Bonferroni at the same family-wise error rate
    and makes no independence assumption, which matters here because the
    metrics within one comparison are correlated by construction.
  * Wilcoxon is not meaningful at very small n: the minimum achievable exact
    two-sided p-value is 0.0625 at n=5, so a "p<0.05" claim is unreachable
    there (this was H8(c), a real published caption error). We therefore
    refuse rather than return a misleading number below MIN_N.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats as _st

# Below this many paired observations the exact two-sided Wilcoxon p-value
# cannot reach 0.05, so reporting one would be misleading (H8c).
MIN_N = 6


def effect_size_label(d: float) -> str:
    """Cohen's conventional bands."""
    a = abs(d)
    if a >= 0.8:
        return "large"
    if a >= 0.5:
        return "medium"
    if a >= 0.2:
        return "small"
    return "negligible"


def align_by_seed(
    a: Mapping[str, float],
    b: Mapping[str, float],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Pair two arms by seed identity.

    Returns (seeds, values_a, values_b) over the sorted intersection of keys.
    This is the whole point of the module: pairing is by key, never by
    position or length (H8a, H8b).
    """
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise ValueError("no common seeds between the two arms")
    return (seeds,
            np.asarray([float(a[s]) for s in seeds], dtype=float),
            np.asarray([float(b[s]) for s in seeds], dtype=float))


def paired_compare(
    a: Mapping[str, float],
    b: Mapping[str, float],
    metric_name: str = "metric",
    ci: float = 0.95,
) -> Dict:
    """Paired comparison of arm `a` against arm `b`, aligned by seed.

    Positive `mean_diff` and positive `cohens_d` mean `a` scored higher.
    Returns raw (uncorrected) p; apply holm() across the declared family.
    """
    seeds, va, vb = align_by_seed(a, b)
    diff = va - vb
    n = len(diff)

    if n >= MIN_N and np.any(diff != 0):
        stat, p = _st.wilcoxon(va, vb, alternative="two-sided")
        stat, p = float(stat), float(p)
    else:
        stat, p = float("nan"), float("nan")

    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    d = float(diff.mean() / sd) if sd > 0 else 0.0

    if n > 1 and sd > 0:
        half = float(_st.t.ppf(0.5 + ci / 2.0, n - 1) * sd / math.sqrt(n))
    else:
        half = float("nan")

    return {
        "metric": metric_name,
        "n_paired": n,
        "paired_seeds": seeds,
        "mean_a": float(va.mean()),
        "std_a": float(va.std(ddof=1)) if n > 1 else 0.0,
        "mean_b": float(vb.mean()),
        "std_b": float(vb.std(ddof=1)) if n > 1 else 0.0,
        "mean_diff": float(diff.mean()),
        "ci_level": ci,
        "ci_low": float(diff.mean() - half) if half == half else float("nan"),
        "ci_high": float(diff.mean() + half) if half == half else float("nan"),
        "wilcoxon_stat": stat,
        "p_value": p,
        "cohens_d": d,
        "effect_size": effect_size_label(d),
        "alternative": "two-sided",
    }


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment, returned in input order.

    NaN p-values (arms we refused to test, see MIN_N) are passed through as NaN
    and excluded from the family size, so an untestable arm cannot inflate the
    correction applied to the testable ones.
    """
    idx = [i for i, p in enumerate(p_values) if p == p]  # drop NaN
    m = len(idx)
    out: List[float] = [float("nan")] * len(p_values)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: p_values[j])):
        adj = max(running, float(p_values[i]) * (m - rank))
        running = adj
        out[i] = min(adj, 1.0)
    return out


def holm_family(comparisons: Mapping[str, Dict]) -> Dict[str, Dict]:
    """Apply Holm across a declared family of paired_compare() results.

    `comparisons` maps a label to a paired_compare() dict. Each result gains
    `holm_p`, `family_size` and `significant_after_holm` in place, so the family
    a correction was applied over is recorded alongside the number itself and
    cannot be lost when the result is serialised.
    """
    labels = list(comparisons)
    adj = holm([comparisons[k]["p_value"] for k in labels])
    testable = sum(1 for k in labels if comparisons[k]["p_value"] == comparisons[k]["p_value"])
    for k, ap in zip(labels, adj):
        comparisons[k]["holm_p"] = ap
        comparisons[k]["family_size"] = testable
        comparisons[k]["significant_after_holm"] = bool(ap == ap and ap < 0.05)
    return dict(comparisons)


def describe(values: Iterable[float]) -> Dict:
    """mean / std / n, with the sample (ddof=1) standard deviation."""
    v = np.asarray(list(values), dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "n": int(v.size),
    }
