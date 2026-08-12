# CLAUDE.md — WavKAN-CL Project Guide

This file orients an LLM (or a new contributor) working in this repository. It reflects the actual state of the code as of a full audit completed 2026-08-11, not the aspirational description in `README.md` — several claims in the README and in the project's own docstrings do not match what the code currently does. Where this file disagrees with a comment or docstring in the code, trust this file (and cite the audit finding) until the underlying issue is resolved.

**There is a live, ongoing scientific-integrity audit of this codebase.** Its full findings register is `AUDIT_FINDINGS.md` at the repo root. Read that file before making any change that touches training, evaluation, statistics, figure generation, or the manuscript. This CLAUDE.md summarizes the parts of it every session needs; `AUDIT_FINDINGS.md` has the exact file:line evidence for every claim below.

---

## 1. What this project actually is

WavKAN-CL is ECG arrhythmia classification research: a Kolmogorov-Arnold Network (KAN) whose basis functions are learnable wavelets, combined with a BiGRU, evaluated on the MIT-BIH Arrhythmia Database under the strict inter-patient DS1/DS2 protocol (de Chazal et al.), with cross-dataset zero-shot evaluation on PTB-XL/INCART/SVDB. The intended contribution is parameter efficiency + structural interpretability (the wavelet basis itself is inspectable, unlike post-hoc attribution methods) + curriculum learning for severe class imbalance (AAMI classes N/S/V/F/Q, with V — ventricular — the clinically critical minority class).

This is a pure ML research repo: no service, no API, no DB, no CI/CD. Everything is standalone Python scripts run manually. `src/` has ~90 files (~15k LOC), `models/` has 6 files, `results/` holds 230 real, git-tracked `.pth` checkpoints (~192MB) from actual past training runs.

**There is no single canonical training entry point that's known to be trustworthy end-to-end.** The closest thing is `src/train_full_pipeline.py` (18 stages, described in §6), but per the audit, several of those stages contain serious, verified integrity problems (narrative-selection code, fabrication fallbacks, rubber-stamp verification). Do not treat "it's wired into the pipeline" as a mark of legitimacy — read `AUDIT_FINDINGS.md` first.

---

## 2. Repository map

```
src/            Core library + ~90 standalone scripts (preprocessing, models, training, eval, stats, figures)
models/         Model class definitions (see §4 — two incompatible "the model" candidates live here)
data/           NOT present in this checkout (gitignored — too large for git). Must be regenerated from
                PhysioNet (MIT-BIH, PTB-XL, INCART, SVDB) via the process_*.py scripts. wfdb is the download/read library.
results/        Real, git-tracked checkpoints and a few JSON metric files. Most result JSONs that later
                pipeline stages expect are ABSENT from this checkout (see AUDIT_FINDINGS.md C5, H2).
paper/          Older manuscript drafts + patent/pitch docs. Not the current submission target.
Submission_JBHI/   THE CURRENT CANONICAL SUBMISSION (see §5). Missing its own references.bib (C7) and two
                    embedded figure files (H14) — the folder as committed will not compile as-is.
Final_Submission_Files/, Medium_Article/   Older/adjacent artifacts, not the live submission.
AUDIT_FINDINGS.md   The full Phase-2 findings register from the ongoing integrity audit. Living document —
                    rows get a Status update, never deleted.
CHANGELOG.md        Fix-by-fix record of every Phase 4+ change (rule 4): what was wrong, what changed, exact
                    file/line, whether it changes any previously-reported number.
PHASE3_MANUSCRIPT_CROSSREF.md   Phase 3 deliverable: claim-by-claim gap table, table/figure-by-table/figure,
                    plus the full citation-integrity resolution table.
PHASE5_SCOPE_PLAN.md   Phase 5 deliverable: exact data-regeneration and 20-seed retraining commands for
                    WavKAN_v2 (the decided canonical model, C3), to be run on GPU infrastructure outside
                    this session and the results brought back for verification.
tests/              Real pytest regression coverage, added starting Phase 4 (previously ~none) —
                    test_split_integrity.py, test_e3c_compliance.py, test_aggregate_seed_results.py.
src/aggregate_seed_results.py   New (2026-08-12): honest seed-paired-by-identity aggregation for the
                    Phase 5 re-run, written to avoid the seed-pairing bugs (H8) found elsewhere.
landmine_scripts_do_not_use/   Quarantined scripts (find_best_seed.py, check_seed_42.py, apply_fixes.py,
                    apply_all_fixes.py, verify_win.py) whose entire purpose was an integrity violation —
                    see that folder's README.md. Moved here 2026-08-12 by project-owner decision.
```

Six root-level `.tex` files exist across the repo (`elsarticle_manuscript.tex`, `manuscript_complete.tex`, `manuscript_elsevier.tex`, `paper/WavKAN-CL.tex`, `Final_Submission_Files/manuscript_complete.tex`, `Submission_JBHI/ieee_manuscript.tex`) plus two divergent `references.bib` files (`paper/references.bib`, `paper/IEEE PAPER/references.bib`). This is not a documentation error to "clean up" casually — the duplicates disagree on real content (different parameter counts, different abstracts, different Table 6 numbers). See §5 before touching any of them.

---

## 3. Data pipeline & the inter-patient split — read this before touching any `process_*.py` or `train_*.py`

- **No raw or processed data ships in this repo.** `data/` is gitignored. To do anything that needs data, you must run the relevant `process_*.py` script against PhysioNet downloads (`wfdb` handles fetch/read). There is no cached copy anywhere in this checkout.
- **The canonical inter-patient split** is `src/split.py`: `TRAIN_RECORDS`/`VAL_RECORDS`/`TEST_RECORDS`, DS1 (train+val) vs DS2 (test), no record overlap — this is the safety-critical guarantee the whole paper depends on. It is correctly implemented (record-level, not shuffled-then-indexed) everywhere the audit checked, **but it has zero automated regression coverage** — `split.py`'s own disjointness assertions only run under `python src/split.py` directly, and neither of the repo's two test-like files touches this at all. **If you're asked to fix anything about data splitting, add a real pytest-collectible regression test for this — there currently is none.**
- **`VAL_RECORDS` is inconsistent across pipelines.** `src/split.py` (used by `preprocessing.py`) has one list; `src/process_data.py`, `process_extended_rr.py`, `process_sequence.py` independently hardcode a *different* `VAL_RECORDS` list (not imported from `split.py`). `TEST_RECORDS`/DS2 is identical everywhere — so there is no train/test leakage in either version — but different model variants are validated on non-identical DS1 subsets, which undermines claims like "near-zero validation-test gap" that implicitly assume comparable validation populations. (`AUDIT_FINDINGS.md` H16.)
- **PTB-XL path naming is inconsistent** between `process_ptbxl.py` (`data/processed_ptbxl`) and `eval_fewshot_ptbxl.py`/`train_full_pipeline.py` (`data/ptbxl_processed`, reversed). Following the documented default pipeline end-to-end currently misses the data and silently falls through to a fabricated-data code path (see §7, C8/H11) instead of erroring. Fix the path convention before relying on this.
- **PTB-XL labels are record-level, not beat-level.** PTB-XL SCP diagnosis codes describe a whole recording; `process_ptbxl.py` blanket-applies one AAMI label to every beat extracted from that recording. This is a real label-granularity mismatch versus MIT-BIH's genuine per-beat annotations — keep this in mind when interpreting or reporting any PTB-XL cross-dataset number (H12).
- **INCART's `val` split is currently a byte-identical copy of its `test` split** (`process_incart.py`). Dormant today (nothing reads it), but if you ever wire early-stopping/model-selection against INCART using the repo's usual `X_val.npy` convention, you will be selecting on the test set. Fix the copy before that happens (M11).
- Positive finding: no augmentation-before-split, no joint train+test normalization statistics, and a genuine patient-ID-level train/test split for PTB-XL specifically — the core independence property holds everywhere the audit checked it.

---

## 4. Model architecture — there are two incompatible candidates for "the" model

This is the single most consequential unresolved ambiguity in the repo (`AUDIT_FINDINGS.md` C3). Know which one you're working with before changing anything architecture-related.

| | `HybridWavKAN_RR` | `WavKAN_v2` (a.k.a. "WavKAN-v2") |
|---|---|---|
| Defined in | Copy-pasted verbatim across 12 files (canonical-ish source: `src/wavkan.py::WavKANLinear` + this class) | `models/wavkan_v2.py` |
| Composition | Plain `WavKANLinear` (mexican_hat) + LayerNorm + BiGRU + RR-history MLP | `PCWIWavKANLinear` (physiology-constrained init) + `PWaveAttentionModule` + RR self-attention |
| Measured params | 95,189 (verified by instantiation) | 154,325 (verified by instantiation — the file's own docstring undercounts this by ~45K, don't trust the docstring's math) |
| Matches | Older abstract language ("95,189 parameters", "0.38 MB") | Current `Submission_JBHI` abstract/footnote ("~154K parameters") |
| Docstring says | — | `"# Full model (paper submission)"` |

Both cannot be "the" 95,189-parameter model the abstract's efficiency claim describes. **This needs a project-owner decision, not a code fix** — it determines which architecture the paper is actually about. Don't silently pick one and "fix" the other's references without that decision (this is an architectural/methodological question per the audit's own ground rules, not a bug).

Other architecture notes:
- `models/wavkan_bigru.py` and `src/interpretability.py` both import `models.wavkan`, a module that was deleted (see `.gitignore`, "Duplicate/Deprecated") and does not exist in this checkout — both files are currently unimportable, dead code.
- `src/focal_loss.py::FocalLoss` is dead code; at least 5 other independent, API-divergent reimplementations of focal loss exist across the training scripts. If you need focal loss, check which one a given script actually imports — don't assume the "canonical"-looking module is the one in use.
- Within-run checkpoint/early-stopping selection is validation-based (not test-based) in every training script the audit checked — this part is done correctly. The test-set-contamination problem in this codebase is entirely at the *cross-run* level (see §7).

---

## 5. Manuscript & submission state

- **Canonical manuscript for all current work: `Submission_JBHI/ieee_manuscript.tex`.** Confirmed newest by both mtime and git log, matches the cover letter's target (IEEE J-BHI), matches the most recent commits ("fig3: Final Reviewer captions + results rewrite"). The other 5 `.tex` files are stale forks from earlier venue attempts (Elsevier, an earlier IEEE draft) — do not edit them under the assumption they're in sync with the live draft; several checked findings show they are *not* (different parameter counts, different Table 6 numbers, different abstracts).
- **Submission status (per project owner, 2026-08-11): previously submitted to JBHI and/or rejected, currently being revised for resubmission.** There is no reviewer decision letter or feedback document anywhere in this repo — if you need to address specific reviewer comments, ask the project owner for that content; it isn't recoverable from the codebase.
- **The submission package as committed will not currently compile.** `Submission_JBHI/ieee_manuscript.tex:473` requires `\bibliography{references}`, but the `Submission_JBHI/` folder has no `references.bib` file at all. Two candidate `.bib` files exist elsewhere in the repo (`paper/references.bib`, `paper/IEEE PAPER/references.bib`) but they disagree with each other on at least one entry's content (see C7 below) — don't just copy one over without reconciling it first. Two embedded figures (`final_methodology_workflow_v2.pdf`, `wavkan_micro_architecture_v2.pdf`) are also absent from `Submission_JBHI/` itself; only same-named root-level files exist and haven't been hash-confirmed as identical to what was actually submitted.
- **`train_full_pipeline.py`'s own docstring targets a different venue** ("TBME/TNSRE Submission Pipeline") than the actual current manuscript (JBHI). This is internal-tooling naming lag, not evidence the paper is being dual-submitted — but it's one more sign the various pipeline scripts and the live manuscript have drifted out of sync with each other over time. Don't assume any given script's docstring describes what currently happens with the live paper.
- The `JBHI_SUBMISSION_CHECKLIST.md` (`paper/IEEE PAPER/`) is a self-authored, all-green checklist written by/for an earlier draft (`paper/IEEE PAPER/ieee_wavkan.tex`), not the current canonical file. Treat it as a stale hypothesis, not evidence of submission-readiness.

---

## 6. `train_full_pipeline.py` — the closest thing to a single entry point

18 stages, dispatched from one script (`python src/train_full_pipeline.py --seeds 42 101 777 2026 31415 --epochs 100`, or `--from-stage N` to resume). Documented runtime: ~6–12 hours CPU for 3 seeds/100 epochs; GPU strongly recommended. Given the audit's findings, treat each stage's output as **unverified until cross-checked against `AUDIT_FINDINGS.md`** — several stages are exactly where the worst issues live:

| Stage | What it does | Audit concern |
|---|---|---|
| 1 | Multi-seed training (WavKAN-v2 + PCA curriculum) | — |
| 2 | Baseline training (ResNet1D, Transformer, CNN+Focal, B-Spline KAN) | Hyperparameters not matched to Stage 1 (H6) |
| 3 | Full metrics & calibration per model | Seed-selection-by-test-metric pattern present (`stage3_full_metrics`, LOW list) |
| 4 | Statistical comparison (Wilcoxon + effect size) | Pairing/one-sided-vs-two-sided issues exist in the *sibling* scripts this stage doesn't use — this specific call path was checked clean (H8 positive note) |
| 5 | Ablation study (`run_ablation_v2`) | Only 9 of 11 documented ablation arms actually implemented (M9) |
| 6 | Interpretability (WAS scores) | — |
| 7 | Few-shot PTB-XL | **`eval_fewshot_ptbxl.py`'s own docstring documents an intent to replace an honest zero-shot result because it "hurt the narrative," plus a silent data-fabrication fallback. Not currently exploited in the live manuscript — verified the live paper still reports the honest number — but read C8 before touching this stage.** |
| 8 | Deployment/Green AI benchmark | Latency claim compares CPU (this repo) vs. a competitor's own GPU number (H9); no `results/latency_report.json` exists in this checkout to substantiate the published figure |
| 9–11 | Comparison tables, multi-dataset table, clinical story | — |
| 12 | **"Win condition verification — run BEFORE writing paper"** | **`verify_win.py` decides the abstract's framing text based on which outcome bucket the numbers land in, with an explicit instruction to keep re-running experiments until a "win" appears. This is real, wired-in code — see C4.** |
| 13 | Leave-one-out RR ablation | Real generating path for the manuscript's RR-ablation figure is untraceable (H17); a separate hardcoded-literal decoy script targets the same output filename |
| 14 | Per-patient consistency | — |
| 15 | Temperature scaling / calibration | — |
| 16 | Noise augmentation study | — |
| 17 | E3C compliance ("4.1% certificate") | **3 of 4 criteria are hardcoded to always PASS regardless of input — see C6. The underlying "4.1% of studies" statistic is real (verified against arXiv:2503.07276) but the in-repo citation for it is wrong in both available `.bib` entries (C7).** |
| 18 | Generate all publication figures | **The two figures this stage's functions (`fig3_seed_stability`, `fig4_convergence`) produce do not match their own captions once embedded in the live manuscript — verified by directly rendering the PDFs. See C1, the single most reviewer-visible finding in the whole audit.** |

`data/processed_rr_history`, `data/ptbxl_processed`, `data/incart_processed`, `data/svdb_processed`, and `results/full_pipeline` are all absent from this checkout — running this end-to-end today requires regenerating everything from PhysioNet first.

---

## 7. Scientific-integrity ground rules (active for all work on this project)

These were established for the ongoing audit and apply to any future change, not just the audit itself:

1. **Never present synthetic, placeholder, or illustrative data as measured data** — in code, comments, figures, or the manuscript. If something can't be regenerated from a real log/checkpoint/dataset, either don't generate it or label it explicitly as schematic/illustrative.
2. **Never invent a citation, a number, or a result.** Verify any citation resolves to a real, findable paper before adding it. This audit ran a full citation-integrity sweep (`AUDIT_FINDINGS.md` C14, all ~21 `\cite{}` keys checked via web search, not just key-matching): 13 of 21 had a documented problem and 9 involved outright fabricated bibliographic content, including 3 that attached a real-format DOI to a real but completely unrelated paper. 12 of the 13 have since been corrected in-place in both `.bib` files (each with a `note` field documenting the fix — see `CHANGELOG.md`); `xiao_deep_2023` remains genuinely unresolved (no real paper found matching its claim) and is explicitly flagged `UNRESOLVED` in both files rather than guessing a replacement. `Submission_JBHI/` still has no `references.bib` of its own (C7) — a Phase 6 item.
3. **No metric, model, or seed selection may ever touch test-set labels.** This is the project's domain-specific integrity rule, chosen because the audit found exactly this failure mode multiple times: `find_best_seed.py`/`check_seed_42.py` (select the seed maximizing V-recall computed on `X_test.npy`), and `train_full_pipeline.py`'s own Stage 3 (`stage3_full_metrics`, same pattern). Selection must use validation data only, or the full seed distribution must be reported honestly (mean±std) rather than a test-set-picked maximum. Disclosing a number as "best-seed" in a caption does **not** cure this — the selection mechanism itself is the problem, independent of disclosure. These specific landmine scripts have not yet been fixed/neutered as of this writing — see §8.
4. **For every bug fixed, add or update a regression test** that would have caught it. As of Phase 4 (2026-08-12), this is no longer "essentially no coverage" — see `tests/` (new directory: `test_split_integrity.py`, `test_e3c_compliance.py`) and the fixed `test_ablation.py` (now real pytest functions, skips cleanly without data rather than running live training on collection). `smoke_test.py` still has only one real assertion in 50 lines — not yet fixed.
5. **Keep `AUDIT_FINDINGS.md`'s Status column current** — update, don't delete, as findings are fixed or resolved as benign. It's the project's audit trail. `CHANGELOG.md` (new, repo root) has the detailed fix-by-fix record this rule and rule 4 require.
6. **Distinguish bug fixes (correcting against stated intent) from architectural/methodological changes (a new design choice that changes the scientific claim).** The latter needs explicit sign-off before it touches the manuscript — e.g., deciding which of the two model identities in §4 is "the" model is a methodological decision, not a fix.
7. Work one phase at a time (orient → audit → cross-reference manuscript → fix → re-verify → update manuscript → final checklist) and get sign-off between phases before changing anything that affects reported numbers.

---

## 8. Known landmine scripts — do not trust the output of these without scrutiny

| Script | What it actually does | Why it's dangerous |
|---|---|---|
| `find_best_seed.py`, `check_seed_42.py` | Load the real test set, pick the checkpoint that maximizes V-recall on it | Test-set-driven model selection (rule 3 above) |
| `apply_fixes.py`, `apply_all_fixes.py` | Mechanically string-replace numbers into `manuscript_complete.tex` | Writes cherry-picked numbers directly into paper source with no independent check |
| `src/fusion_engine.py` | Ensembles ~11 checkpoints across 2 architectures with hand-picked weights, feeds `results/thesis_fusion_metrics.json` | This file is what `verify_manuscript.py` labels "Table 6 — Best Seed" — it is not a single seed, it's an ensemble |
| `verify_manuscript.py` | Compares `results/*.json` values to hardcoded Python literals with comments like `# manuscript says 0.87` | Never reads the actual `.tex` file; already stale against the current abstract; 5 of 6 required JSON files don't exist in this checkout so it can't currently even run |
| `check_manuscript.py`, `check_citations.py` | **FIXED 2026-08-12.** Was: hardcoded paths to the stale `manuscript_complete.tex` and a repo-root `references.bib` that doesn't exist; citation check's own bib-key regex (`\w+`) silently truncated every hyphenated key. Now: point at `Submission_JBHI/ieee_manuscript.tex` + `paper/IEEE PAPER/references.bib`, regex fixed. Still true: citation check is key-matching only, cannot by itself detect a fabricated-but-key-matched citation (that's why the C14 sweep used web search, not this script). Running the fixed `check_manuscript.py` immediately surfaced 2 new findings (H24: a broken `\ref`, two missing author-photo files). |
| `src/e3c_compliance.py` | **FIXED 2026-08-12.** Was: 3 of its 4 pass/fail criteria unconditionally hardcoded to PASS. Now: each criterion derives PASS/PARTIAL/FAIL from real evidence where available (real generated split-ID arrays, real label files, real threshold comparisons), PARTIAL when that evidence doesn't exist yet. Regression tests in `tests/test_e3c_compliance.py`. |
| `src/verify_win.py` | Computes an "abstract framing" and "contributions" list from the current experiment results | **QUARANTINED 2026-08-12** — moved to `landmine_scripts_do_not_use/verify_win.py` (project-owner decision; see that folder's README.md and `AUDIT_FINDINGS.md` C4). `train_full_pipeline.py`'s Stage 12 no longer imports it — it's now a documented no-op stub rather than a crash. |
| `src/eval_fewshot_ptbxl.py` | Few-shot fine-tune + eval on PTB-XL | Docstring documents intent to replace an honest, worse-looking zero-shot result; has a silent numeric-fabrication fallback when data is missing. **Not yet fixed** — not currently exploited in the live manuscript, lower urgency than the quarantined scripts. |
| `run_audit.py` | Globs `results/multiseed/{baseline,curriculum}/seed_*` | `baseline/` doesn't exist in this checkout at all; `curriculum/` has only 10 of the claimed 20 seeds; hardcodes the target value it's "checking". **Not yet fixed.** |
| `test_ablation.py` | **FIXED 2026-08-12.** Was: zero assertions, swallowed all exceptions, ran two live training epochs as a side effect of test *collection*. Now: real pytest functions with real assertions, training calls only run inside the test body, skips cleanly (in <0.1s) when `data/processed_rr_history/` is absent, which it is in this checkout. |
| `find_best_seed.py`, `check_seed_42.py`, `apply_fixes.py`, `apply_all_fixes.py` | Test-set-driven seed selection / mechanical number-injection into the manuscript | **QUARANTINED 2026-08-12** — moved to `landmine_scripts_do_not_use/` (project-owner decision, "quarantine" chosen over delete/warning-header/leave-untouched). Git history preserved via `git mv`. |

Full detail and file:line citations for every item above are in `AUDIT_FINDINGS.md`. Fix-by-fix detail for everything marked FIXED/QUARANTINED is in `CHANGELOG.md`. See `landmine_scripts_do_not_use/README.md` for what's quarantined and why.

---

## 9. Environment & running things

- Dependencies: `requirements.txt` (torch, numpy, scikit-learn, scipy, matplotlib, wfdb, tqdm, codecarbon, psutil). No lockfile; no virtualenv committed.
- No data ships with the repo. First step for any real work: regenerate MIT-BIH (and PTB-XL/INCART/SVDB as needed) via the `process_*.py` scripts against PhysioNet, using `wfdb`.
- No CI. No pre-commit hooks beyond whatever your own git client runs. Nothing automatically checks any of this — that's exactly why the audit exists.
- The project owner has confirmed unlimited GPU/compute access for future re-runs (relevant once Phase 4/5 of the audit reaches the point of needing real re-training).

---

## 10. Current project status (update this section as the audit progresses)

- Audit Phase 1 (orientation), Phase 2 (deep audit), and Phase 3 (manuscript cross-reference) are complete. Findings are in `AUDIT_FINDINGS.md`: 14 CRITICAL, 23 HIGH (H24 added during Phase 4 — see below), 17 MEDIUM, ~20 LOW findings, plus a positive-findings section. The Phase 3 deliverable — a claim-by-claim gap table walking the manuscript table-by-table and figure-by-figure, plus a full citation-integrity resolution table — is `PHASE3_MANUSCRIPT_CROSSREF.md`.
- **Phase 4 (fix) is in progress**, scoped to code/data/citation artifacts only — no manuscript `.tex` edits (that's Phase 6, after Phase 5 produces real re-verified numbers). Done so far, all logged in `CHANGELOG.md` with regression tests: added `tests/test_split_integrity.py` and `tests/test_e3c_compliance.py` (new `tests/` directory); fixed `test_ablation.py` (real assertions, no more live-training-on-collection); fixed `src/e3c_compliance.py`'s 3 hardcoded-PASS criteria; fixed `check_manuscript.py`/`check_citations.py`'s wrong file paths (which also surfaced two more real bugs in those scripts themselves, now also fixed, and two brand-new findings, H24); corrected 12 of the 13 fabricated/wrong citation entries found in C14, in-place in both `.bib` files. **Also done**: the 5 "landmine" scripts whose entire purpose was an integrity violation (`find_best_seed.py`, `check_seed_42.py`, `apply_fixes.py`, `apply_all_fixes.py`, `src/verify_win.py`) have been quarantined to `landmine_scripts_do_not_use/` (project-owner decision: quarantine over delete/warning-header/leave-untouched), with `train_full_pipeline.py`'s Stage 12 updated to a documented no-op instead of crashing on the now-missing import.
- **Model identity (C3/C11) is now DECIDED: WavKAN_v2 (154,325 params).** This unblocked concrete Phase 5 script preparation: `src/split.py`'s canonical record lists are now actually used by `process_data.py`/`process_extended_rr.py`/`process_sequence.py` (H16 fixed — they used to independently diverge); the PTB-XL path-naming mismatch is fixed (H11); `train_pca.py` gained a `--no-curriculum` flag so the baseline arm is a genuine single-variable ablation against the curriculum arm (H6/H20); and a new, regression-tested `src/aggregate_seed_results.py` exists to honestly aggregate the resulting 20-seed run without H8's pairing bugs. **`PHASE5_SCOPE_PLAN.md` has the exact commands** — this session has no GPU (confirmed: `torch 2.6.0+cpu`, no `nvidia-smi`), so the actual training/data-regen runs happen on the project owner's separate GPU infrastructure; results come back for verification and Phase 6.
- **Phase 3 surfaced two categories of finding Phase 2 missed by only reading code:** (1) the manuscript's own prose systematically conflates the two model identities from §4 — every "WavKAN-CL" mention pairs with 95,189 params, every "WavKAN-v2" mention pairs with 154,325 params, sometimes within the same paragraph (`AUDIT_FINDINGS.md` C11); and a direct in-text contradiction about what SMOTE does to S-Recall, stated as both a decrease (0.290→0.245) and an increase to the study's highest value (≈0.46) in two different sections (C12). (2) A full citation-integrity sweep (all ~21 `\cite{}` keys, verified against real sources, not just key-matching) found 13 of 21 have a documented problem and 9 involve outright fabricated bibliographic content — including 3 that attach a real-format DOI to a real but completely unrelated paper (C14). The foundational `chazal_automatic_2004` DS1/DS2-protocol citation is fully verified correct.
- Scope agreed with project owner: **audit-only through Phase 3** (now complete), then stop for explicit sign-off before any fix touches the manuscript or re-runs any experiment. Currently awaiting that sign-off before Phase 4.
- **Do not edit the manuscript, "fix" any reported number, or re-run any training that would change a previously-reported result without going through that sign-off process.** This applies even to changes that look like obvious corrections — e.g., don't just delete `verify_win.py` or edit Table 6 numbers unilaterally; log it as a proposed fix against a specific `AUDIT_FINDINGS.md` row first.
- Open questions blocking Phase 4, listed with the rest of the Phase 3 sign-off checklist at the end of `PHASE3_MANUSCRIPT_CROSSREF.md`: which manuscript/model identity is canonical going forward (§4/§5 above), which of the two contradictory SMOTE numbers is real, whether Figs. 3/4/6 and the 20-seed headline statistics can be re-derived from what currently exists or require new training runs, and the citation-correction plan for the 9 fabricated/wrong references.
