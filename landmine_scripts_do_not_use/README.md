# Quarantined scripts — do not use, do not import

The five scripts in this folder were moved here on 2026-08-12, during Phase 4 of the
ongoing scientific-integrity audit (`AUDIT_FINDINGS.md` at repo root), by explicit
project-owner decision. They are **not bugs to be fixed** — each script's entire purpose
*is* the integrity violation the audit flagged. "Fixing" them would mean removing the
behavior that makes them exist in the first place, so they were quarantined instead of
patched. Git history is preserved (moved with `git mv`), so nothing is lost.

Do not resurrect, import, or run any of these without re-reading the relevant
`AUDIT_FINDINGS.md` row first and getting explicit sign-off — the same rule that applied
to fixing them applies to reviving them.

## What's here and why

| File | What it does | AUDIT_FINDINGS.md ID |
|---|---|---|
| `find_best_seed.py` | Loads the real held-out test set (`data/processed_rr_history/X_test.npy`), loops over 20 seed checkpoints, and picks whichever one **maximizes V-recall on the test set** — its own docstring says "Find the seed that matches the Table 6 values in the manuscript." | Original context-setting finding (predates the C/H numbering); referenced throughout as the canonical example of test-set-driven selection. |
| `check_seed_42.py` | Same pattern, narrowed to seed 42 specifically — comment: "Just check seed 42 which we know is V-Rec ~ 0.898" / "Let's save the exact numbers that matched the manuscript!" | Same as above. |
| `apply_fixes.py` | Mechanically string-replaces the numbers `find_best_seed.py` produced directly into `manuscript_complete.tex`'s Table 6 and its "V-class Recall" text, with no independent check. | H1 |
| `apply_all_fixes.py` | Same pattern at larger scale, across more of the manuscript (including the "mean X, peak Y" framing traced in M7). | H1, M7 |
| `verify_win.py` | Computes an "abstract framing" and "contributions" list for the paper *from the current experiment results* — i.e., decides what scientific claim to make only after seeing the data. Its `PIVOT` branch explicitly instructs: "Submit ONLY if WAS score / variance / calibration arguments are strong. Otherwise: run more epochs, add augmentation, tune S-weight hyperparameter" — keep altering the experiment until a "win" appears. Was imported by `src/train_full_pipeline.py` Stage 12; that call site has been changed to a no-op stub (see `CHANGELOG.md`) rather than left to crash on the now-missing import. | C4 |

## If a real replacement is ever needed

The audit's own domain-specific integrity rule (`CLAUDE.md` §7, rule 3): **no metric,
model, or seed selection may ever touch test-set labels.** Selection must use validation
data only, or the full seed distribution must be reported honestly (mean±std) — not a
test-set-picked maximum, and not a post-hoc narrative decided after seeing results.
Any future "pick the seed to feature" or "decide the abstract framing" tooling needs to
be built to that constraint from the start, not adapted from what's in this folder.
