"""
derive_split_counts.py -- exact per-record and per-split AAMI beat counts for
MIT-BIH, derived from the PhysioNet annotation files with process_data.py's own
selection rule.

Why this exists (AUDIT_FINDINGS.md H49). The manuscript's class-distribution
table and the Phase 10 GPU config both carried per-split counts from an
extraction made BEFORE the H16 fix, when process_data.py hardcoded validation
records 215/220/223/230. H16 switched it to split.py's 208/209/223/230 -- the
list the manuscript's prose states -- and every published arm was trained on
that corrected split. But nobody re-derived the counts, so:

  * the class-distribution table reported train 40,726 / val 10,266 while the
    published models were trained on 40,177 / 10,815 (F: 28 training beats, not
    399 -- record 208 holds 372 of the database's 802 fusion beats and sits in
    validation);
  * the Phase 10 preflight, copying those stale counts, rejected the GPU box's
    correct extraction for three weeks, and the box was then made to "pass" by
    regenerating the pre-H16 split, so the Phase 10 arms were trained and
    selected on different records from the arm they were compared against.

Counts copied between documents drift. This script re-derives them from the
source annotations so the table, the config and the tests can all be checked
against one ground truth. Only annotation and header files are read (no
signals), so it needs network access to PhysioNet or a local mirror, and no GPU.

The selection rule is process_data.py's exactly: a beat is kept iff its symbol
is in AAMI_MAP and its 360-sample window fits inside the record
(r - PRE_SAMPLES >= 0 and r + POST_SAMPLES <= sig_len). process_data.py also
drops windows with std < 1e-7; that never fires on MIT-BIH, which the forensic
regeneration of 2026-09-28 confirmed (regenerated arrays match these counts to
the beat).

Usage:
    python src/derive_split_counts.py                 # PhysioNet
    python src/derive_split_counts.py --local data/raw
    python src/derive_split_counts.py --check         # compare with the committed file
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from split import DS1_RECORDS, TEST_RECORDS, TRAIN_RECORDS, VAL_RECORDS  # noqa: E402

OUT = REPO / "configs" / "mitbih_split_counts.json"

# Kept in lock-step with process_data.py; test_split_counts.py asserts they match.
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'Q': 4, '/': 4, 'f': 4,
}
FS = 360
PRE_SAMPLES = int(0.25 * FS)   # 90
POST_SAMPLES = int(0.75 * FS)  # 270
CLASSES = ["N", "S", "V", "F", "Q"]
# The pre-H16 list, kept only so the stale counts can be explained, never used.
PRE_H16_VAL = ["215", "220", "223", "230"]


def record_counts(rec, local=None):
    import wfdb
    if local:
        path = os.path.join(local, rec)
        ann = wfdb.rdann(path, "atr")
        sig_len = wfdb.rdheader(path).sig_len
    else:
        ann = wfdb.rdann(rec, "atr", pn_dir="mitdb")
        sig_len = wfdb.rdheader(rec, pn_dir="mitdb").sig_len
    c = Counter()
    for r, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI_MAP and r - PRE_SAMPLES >= 0 and r + POST_SAMPLES <= sig_len:
            c[AAMI_MAP[sym]] += 1
    return [int(c[k]) for k in range(5)]


def tally(per, recs):
    t = [0] * 5
    for r in recs:
        t = [a + b for a, b in zip(t, per[r])]
    return t


def derive(local=None):
    per = {r: record_counts(r, local) for r in sorted(set(DS1_RECORDS) | set(TEST_RECORDS))}
    splits = {"train": sorted(TRAIN_RECORDS), "val": sorted(VAL_RECORDS),
              "test": sorted(TEST_RECORDS)}
    counts = {s: dict(zip(map(str, range(5)), tally(per, recs))) for s, recs in splits.items()}
    stale_train = [r for r in DS1_RECORDS if r not in PRE_H16_VAL]
    return {
        "source": "PhysioNet mitdb annotations (.atr) + headers, process_data.py selection rule",
        "selection_rule": {"aami_map": AAMI_MAP, "pre_samples": PRE_SAMPLES,
                           "post_samples": POST_SAMPLES},
        "classes": CLASSES,
        "records": splits,
        "counts": counts,
        "totals": {s: sum(v.values()) for s, v in counts.items()},
        "per_record": per,
        "stale_pre_h16_for_reference_only": {
            "val_records": PRE_H16_VAL,
            "train": tally(per, stale_train), "val": tally(per, PRE_H16_VAL),
            "note": ("counts of the pre-H16 split that the manuscript's class table and the "
                     "Phase 10 config carried until 2026-09-28 (AUDIT_FINDINGS.md H49). "
                     "No published arm was trained on this split."),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="directory holding MIT-BIH .atr/.hea files")
    ap.add_argument("--check", action="store_true",
                    help="re-derive and compare with the committed file; exit 1 on mismatch")
    a = ap.parse_args()
    got = derive(a.local)
    if a.check:
        ref = json.load(open(OUT))
        ok = ref["counts"] == got["counts"] and ref["records"] == got["records"]
        print("committed counts", "MATCH" if ok else "DIFFER", "the re-derived counts")
        return 0 if ok else 1
    OUT.write_text(json.dumps(got, indent=1) + "\n")
    for s in ("train", "val", "test"):
        print("%-5s %2d records  %6d beats  %s" % (
            s, len(got["records"][s]), got["totals"][s],
            {CLASSES[int(k)]: v for k, v in got["counts"][s].items()}))
    print("wrote", OUT.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
