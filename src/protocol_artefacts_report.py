"""
protocol_artefacts_report.py -- quantify two protocol artefacts from the MIT-BIH
annotation files and write them to a small, git-tracked JSON so the manuscript's
numbers can be checked without the raw data being present.

Added 2026-09-30 (final pre-submission audit; AUDIT_FINDINGS.md H59, H63).

  1. Annotation-derived RR artefact (H59). process_data.py computes RR intervals
     between consecutive entries of ann.sample, which includes non-beat annotations.
     For every extracted beat we compare the published interval vector with one
     computed between beat annotations only (src/extract_matched.py, rr_mode
     "beats_only"), and count beats with any element changed and with RR_0 changed,
     per split and per class.
  2. Post-R lookahead in the window (H63). The window spans 250 ms before and 750 ms
     after the R-peak, so it can contain the next beat. We count extracted beats whose
     next beat annotation falls inside the window.

Needs data/raw (the 44 MIT-BIH records). Usage:
    python src/protocol_artefacts_report.py --out results/rr_sensitivity/protocol_artefacts.json
"""
import argparse
import json
import os
import sys

import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(__file__))
from split import TRAIN_RECORDS, VAL_RECORDS, TEST_RECORDS          # noqa: E402
import extract_matched as em                                          # noqa: E402

CLASSES = "NSVFQ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/rr_sensitivity/protocol_artefacts.json")
    a = ap.parse_args()
    report = {"source": "MIT-BIH annotation files (PhysioNet mitdb 1.0.0), data/raw",
              "window": {"pre_samples": em.PRE_SAMPLES, "post_samples": em.POST_SAMPLES, "fs": em.FS}}
    for split, recs in [("train", TRAIN_RECORDS), ("val", VAL_RECORDS), ("test", TEST_RECORDS)]:
        tot = np.zeros(5, int); anyc = np.zeros(5, int); rr0 = np.zeros(5, int); rr0_short = np.zeros(5, int)
        look = 0
        nonbeat = {}
        for rec in recs:
            ann = wfdb.rdann(os.path.join(a.raw, rec), "atr")
            n = wfdb.rdheader(os.path.join(a.raw, rec)).sig_len
            dummy = np.zeros(n)
            for s in ann.symbol:
                if s not in em.BEAT_SYMBOLS:
                    nonbeat[s] = nonbeat.get(s, 0) + 1
            # extract_beats on a constant-zero signal would drop every beat (std=0), so
            # use a unit ramp: the window/boundary selection is signal-independent apart
            # from the degenerate-variance guard, which a ramp never trips.
            ramp = np.arange(n, dtype=float)
            _, r_all, y = em.extract_beats(ramp, ann.sample, ann.symbol, "all_annotations")
            _, r_bo, y2 = em.extract_beats(ramp, ann.sample, ann.symbol, "beats_only")
            assert np.array_equal(y, y2)
            diff = np.abs(r_all - r_bo) > 1e-6
            for c in range(5):
                m = y == c
                tot[c] += m.sum(); anyc[c] += diff[m].any(1).sum(); rr0[c] += diff[m, 2].sum()
                rr0_short[c] += (diff[m, 2] & (r_all[m, 2] < r_bo[m, 2])).sum()
            beats = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in em.BEAT_SYMBOLS])
            for s, sym in zip(ann.sample, ann.symbol):
                if sym not in em.AAMI_MAP or s - em.PRE_SAMPLES < 0 or s + em.POST_SAMPLES > n:
                    continue
                j = np.searchsorted(beats, s, side="right")
                look += bool(j < len(beats) and beats[j] - s < em.POST_SAMPLES)
            del dummy
        T = int(tot.sum())
        report[split] = {
            "beats": T,
            "any_rr_element_changed": int(anyc.sum()), "any_rr_element_changed_pct": 100 * anyc.sum() / T,
            "rr0_changed": int(rr0.sum()), "rr0_changed_pct": 100 * rr0.sum() / T,
            "rr0_changed_and_shorter": int(rr0_short.sum()),
            "per_class": {CLASSES[c]: {"n": int(tot[c]), "any": int(anyc[c]), "rr0": int(rr0[c]),
                                       "rr0_pct": (100 * rr0[c] / tot[c]) if tot[c] else None}
                          for c in range(5)},
            "next_beat_inside_window": look, "next_beat_inside_window_pct": 100 * look / T,
            "non_beat_annotation_counts": nonbeat,
        }
        print(f"{split}: {T} beats; any RR changed {anyc.sum()} ({100*anyc.sum()/T:.2f}%); "
              f"RR0 changed {rr0.sum()} ({100*rr0.sum()/T:.2f}%); next beat in window {100*look/T:.1f}%")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
