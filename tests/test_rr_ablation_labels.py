"""Regression test for src/rr_ablation.py's RR_POSITIONS labels.

Was: RR_POSITIONS = ["t-4","t-3","t-2","t-1","t"], which assumes the 5-element
RR-history array is purely causal. It is not -- src/process_data.py (sole
writer of data/processed_rr_history, the canonical training data for every
checkpoint in this paper) builds the array as
[RR_-2, RR_-1, RR_0(Pre), RR_+1(Post), RR_+2]: index 3 is a future interval,
not "t-1", and index 2 is the classified beat's own immediate pre-RR
interval, not "two beats prior" (AUDIT_FINDINGS.md C21). This test
independently re-derives the array using the exact same construction
process_data.py uses (not a copy-paste of RR_POSITIONS itself) and checks
each index's real semantics against its label, so a future accidental
reorder/revert of RR_POSITIONS would be caught here even if nobody re-reads
process_data.py's comments.
"""
import numpy as np

from src.rr_ablation import RR_POSITIONS


def _build_rr_window(r_peaks, i):
    """Mirrors process_data.py's real construction exactly (get_rr + window
    order), on synthetic R-peak timestamps, so this test is not just
    re-asserting the same hardcoded list twice."""
    def get_rr(idx_current, idx_prev):
        if idx_prev < 0 or idx_current >= len(r_peaks):
            return 0.8
        return r_peaks[idx_current] - r_peaks[idx_prev]

    rr_0 = get_rr(i, i - 1)
    rr_m1 = get_rr(i - 1, i - 2)
    rr_m2 = get_rr(i - 2, i - 3)
    rr_p1 = get_rr(i + 1, i)
    rr_p2 = get_rr(i + 2, i + 1)
    return [rr_m2, rr_m1, rr_0, rr_p1, rr_p2]


def test_rr_positions_has_five_entries():
    assert len(RR_POSITIONS) == 5


def test_rr_positions_labels_match_the_real_array_semantics():
    # Evenly-spaced synthetic R-peaks so each interval is distinguishable
    # by its position alone (interval k has length k+1).
    r_peaks = np.cumsum([1, 2, 3, 4, 5, 6, 7]).astype(float)  # 7 beats, i=3 is centered
    i = 3

    window = _build_rr_window(r_peaks, i)
    assert len(window) == len(RR_POSITIONS)

    # index 2 must be the interval immediately preceding beat i (the "Pre" /
    # current interval) -- the position the manuscript's headline RR-ablation
    # finding is actually about.
    assert window[2] == r_peaks[i] - r_peaks[i - 1]
    assert "0" in RR_POSITIONS[2] or "pre" in RR_POSITIONS[2].lower()

    # index 3 must be a FUTURE interval (uses beat i+1) -- must NOT be
    # labeled as any "preceding" / "t-1" style causal position.
    assert window[3] == r_peaks[i + 1] - r_peaks[i]
    assert "+1" in RR_POSITIONS[3] or "post" in RR_POSITIONS[3].lower()
    assert "t-1" not in RR_POSITIONS[3]

    # index 4 must also be a FUTURE interval (uses beat i+2).
    assert window[4] == r_peaks[i + 2] - r_peaks[i + 1]
    assert "+2" in RR_POSITIONS[4] or "post" in RR_POSITIONS[4].lower()
    assert RR_POSITIONS[4].strip().lower() != "t"  # must not be labeled "current"

    # Neither of the two preceding-only positions (index 0, 1) may claim to
    # be "current" or "post" -- they are genuinely causal.
    for idx in (0, 1):
        assert "post" not in RR_POSITIONS[idx].lower()
        assert "pre)" not in RR_POSITIONS[idx].lower()
