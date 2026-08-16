# Quarantined scripts — do not use, do not import

The five scripts in this folder were moved here on 2026-08-12, during Phase 4 of the
ongoing scientific-integrity audit (`AUDIT_FINDINGS.md` at repo root), by explicit
project-owner decision. They are **not bugs to be fixed** — each script's entire purpose
*is* the integrity violation the audit flagged. "Fixing" them would mean removing the
behavior that makes them exist in the first place, so they were quarantined instead of
patched. Git history is preserved (moved with `git mv`), so nothing is lost.

**A sixth script, `generate_full_class_table.py`, was found and quarantined the same way
on 2026-08-14**, during a repository-cleanup pass — it wasn't caught by the original
Phase 2 audit because it had zero references anywhere in the codebase (not imported, not
called from any pipeline stage) and only surfaced via a systematic zero-reference scan.
It matches the exact same pattern as `find_best_seed.py` below (same "Table 6" target,
overlapping seed list, test-set-label-driven selection) — applying the same
already-established policy (quarantine, not delete, not silently fix) to a new instance
of the same problem, not a new decision.

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
| `generate_full_class_table.py` (added 2026-08-14) | Loops over 10 seeds, loads each seed's `predictions.npy`/`true_labels.npy` (test-set predictions and ground truth), computes S-Recall via `classification_report` **on the test set**, and keeps whichever seed maximizes it — comments say "The user mentioned 'Best' in Table 2 has S-RECALL 0.45. I need to find WHICH seed that was... I need to find the seed that gave that result to generate the correct table," then prints a "Table 6: Detailed Per-Class Performance (Best Model)" using that cherry-picked seed. | H26 |
| `generate_rr_ablation_figure.py` (added 2026-08-16) | Already found and documented by the original Phase 2 audit (H17) but left in `src/` until now. Hardcodes fabricated RR-ablation delta/std values as Python literals (`delta_s_recall = [-0.004, -0.006, 0.062, -0.002, 0.004]`) under the comment "Data from ablation study (mean ± std across 10 seeds)" — no such study backs these numbers; confirmed by file hash to be what actually produced the root-level `fig_rr_ablation_v2.pdf`-named copy. Quarantined now because its honest replacement (`src/rr_ablation.py`, run for real against all 20 curriculum seeds — `results/rr_ablation_real/`) exists and works, removing any reason to keep this one reachable. | H17 |

## If a real replacement is ever needed

The audit's own domain-specific integrity rule (`CLAUDE.md` §7, rule 3): **no metric,
model, or seed selection may ever touch test-set labels.** Selection must use validation
data only, or the full seed distribution must be reported honestly (mean±std) — not a
test-set-picked maximum, and not a post-hoc narrative decided after seeing results.
Any future "pick the seed to feature" or "decide the abstract framing" tooling needs to
be built to that constraint from the start, not adapted from what's in this folder.
