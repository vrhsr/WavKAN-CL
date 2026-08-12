# Changelog — Scientific-Integrity Audit Fixes

Per audit ground rule 4: every entry states what was wrong, what changed, the exact file/line, and — critically — whether it changes any previously-reported result. **None of the entries below change any number previously reported in the manuscript.** They fix code behavior, test coverage, citation metadata, and dev-tool scripts. Any fix that *would* change a reported number is explicitly called out as NOT YET DONE, pending the Phase 3 sign-off decisions (see `PHASE3_MANUSCRIPT_CROSSREF.md` §7) and Phase 5 re-verification.

Location for this file confirmed by default (repo root) — no explicit preference was given.

---

## 2026-08-12 — Phase 4, session 1

### Regression test coverage added (rule 3)

- **`tests/test_split_integrity.py` (new file).** AUDIT_FINDINGS.md's "Zero-automated-coverage note": the DS1/DS2 inter-patient split — the single most safety-critical guarantee in the codebase — had zero automated regression coverage; `src/split.py`'s own disjointness assertions only ran under `python src/split.py` directly. Added 7 pytest tests covering: three-way disjointness of `TRAIN_RECORDS`/`VAL_RECORDS`/`TEST_RECORDS`, that they exactly partition `DS1_RECORDS`, that records 201/202 land in different sets, and — per H16 — that the *independently hardcoded* `VAL_RECORDS` lists in `process_data.py`/`process_extended_rr.py`/`process_sequence.py` never overlap `TEST_RECORDS` either, while explicitly asserting (not silently resolving) that those lists still diverge from `split.py`'s own list. Record lists are parsed from source text rather than imported, since some of those modules have import-time side effects (`os.makedirs`) and require `neurokit2`, which isn't even in `requirements.txt`. **Does not change any reported number** — this is new coverage of an already-correct mechanism.
  - Verified: `python -m pytest tests/test_split_integrity.py -v` → 7 passed.

- **`tests/test_e3c_compliance.py` (new file).** Regression coverage for the `e3c_compliance.py` fix below — asserts each of the 4 criteria produces PARTIAL/FAIL under the specific missing-evidence / bad-input conditions the original hardcoded-PASS logic could never have caught.
  - Verified: `python -m pytest tests/test_e3c_compliance.py -v` → 5 passed.

### Bug fixes (correcting code against its own stated intent — rule 6)

- **`test_ablation.py` (AUDIT_FINDINGS.md H5).** Was: zero `assert` statements, wrapped in a bare `try/except` that only printed a traceback (no `raise`/`sys.exit`), so it could never fail under pytest despite the `test_*.py` name — and the two live training calls sat at module level, running as a side effect of test *collection*, not just test *execution*. Fixed: converted to two real pytest functions (`test_bspline_ablation_config_trains_without_error`, `test_mexican_hat_focal_loss_ablation_config_trains_without_error`) with real assertions on the returned metrics dict (valid `[0,1]` scores, non-zero trainable params); training calls moved inside the functions (lazy import too, so a skip is instant); both skip cleanly with an explicit reason when `data/processed_rr_history/` is absent, which it is in this checkout.
  - Verified: `python -m pytest test_ablation.py -v` → 2 skipped (cleanly, 0.08s — was 51s before moving the import inside the functions).

- **`src/e3c_compliance.py` (AUDIT_FINDINGS.md C6).** Was: `check_criterion_1`, `check_criterion_3`, `check_criterion_4` each set `status="PASS", score=1.0` unconditionally at the end of the function regardless of what evidence (if any) was gathered above — none of the three could ever fail. Fixed:
  - `check_criterion_1`: now additionally loads real generated `data/processed_rr_history/ids_train.npy`/`ids_test.npy` (the actual record-ID arrays `process_data.py:164` produces) and checks *those* for overlap — FAILs on real overlap, PARTIALs (not PASSes) when those files don't exist yet, only PASSes when the real arrays are verified disjoint. The original hardcoded-literal DS1/DS2 comparison is kept as a secondary, clearly-labeled check, not the sole basis for PASS.
  - `check_criterion_3`: now PARTIALs (not PASSes) when `y_test.npy` is absent, and FAILs if real labels fall outside the AAMI 0-4 mapping; only PASSes when real labels are actually checked and valid.
  - `check_criterion_4`: now derives PASS/PARTIAL/FAIL from whether the supplied `n_params`/`latency_ms` actually meet `max_params`/`max_latency` (PASS if both, PARTIAL if one, FAIL if neither) instead of always PASS; also now notes when `results/latency_report.json` doesn't exist to substantiate a CLI-supplied `latency_ms`.
  - Also fixed: wrong citation year ("Silva et al. 2022" → "Silva et al. 2025, arXiv:2503.07276", 3 occurrences) and a dangling `\citeauthor{silva2022}` (a bib key that doesn't exist anywhere) in `generate_latex_certificate`, replaced with the real key `silva_systematic_2025`; that function's caption text also used to unconditionally claim "satisfies all four criteria... 4.1%" regardless of `report.compliant` — now branches on the actual result. Softened the module docstring's "PROVES it programmatically" framing to match what the script now actually does.
  - **Does not change any reported number** — no live pipeline stage's output is currently affected by this (verified: none of the manuscript's tables/figures currently trace to a `results/e3c_compliance/` output in this checkout — see `PHASE3_MANUSCRIPT_CROSSREF.md` §3, E3C row).

- **`check_manuscript.py`, `check_citations.py` (AUDIT_FINDINGS.md H3).** Was: both hardcoded `manuscript_complete.tex` (a stale draft, materially different from the live submission) and a repo-root `references.bib` that doesn't exist anywhere — running either crashed or checked the wrong document. Fixed: both now point at `Submission_JBHI/ieee_manuscript.tex` (confirmed canonical) and `paper/IEEE PAPER/references.bib` (the more-corrected candidate bib file, see citation fixes below).
  - **Fixing this surfaced two more real bugs that were previously latent** (the scripts had never successfully run before, since the file they pointed at didn't exist):
    - `check_citations.py`'s bib-key regex was `r'@\w+\{(\w+)'` — `\w` doesn't match hyphens, so every hyphenated key (`takalo-mattila_inter-patient_2018`, `zhao_mak-net_2025`, etc.) was truncated at its first hyphen and reported as "missing" even when present. Fixed to `r'@\w+\{([\w-]+)'`.
    - `check_manuscript.py` crashed with `UnicodeEncodeError` printing a ✅ character on a Windows console using cp1252. Fixed by reconfiguring stdout to UTF-8 at the top of the script.
  - **Running the now-fixed `check_manuscript.py` for the first time surfaced two genuinely new findings**, logged as **H24**: `ieee_manuscript.tex:355` references `Table \ref{tab:ptbxl_zero_shot}`, but no matching `\label` exists anywhere in the document (would render as "??" with an undefined-reference warning); and the `\IEEEbiography` blocks require `venkateramanan.jpg`/`ramanathan.jpg`, which are absent from the repository entirely (not just from `Submission_JBHI/`, unlike the H14 figures).
  - **Does not change any reported number.**

### Citation-metadata corrections (AUDIT_FINDINGS.md C14 — rule 1/2: don't present fabricated data as real, verify before including)

Corrected in-place in both `paper/references.bib` and `paper/IEEE PAPER/references.bib` (each entry now carries a `note` field documenting exactly what was wrong and the source of the correction, so the audit trail survives in the bib file itself, not just this changelog):

| Key | What changed |
|---|---|
| `guo_inter-patient_2018` → `guo_inter-patient_2019` | Key mismatch fixed in `paper/references.bib` (content was already correct; manuscript cites `_2019`). Added the entry to `paper/IEEE PAPER/references.bib`, which had none at all. |
| `liu_kan_2025` → `liu_kan_2024` | Key mismatch fixed in `paper/references.bib` (content already correct; manuscript cites `_2024`). |
| `akan_ecgformer_2024` | Replaced fabricated title/authors/journal/DOI (resolved to an unrelated real paper) with the real ECGformer paper's actual details (Akan, Taymaz; Alp, Sait; Bhuiyan, Mohammad Alfrad Nobel; arXiv:2401.05434 / IEEE CSCI 2023). Fixed in both files. |
| `taleban_explainable_2026` | Replaced `paper/references.bib`'s fabricated/unconfirmed entry with the verified-real one (already correct in the IEEE PAPER file). |
| `bahrami_investigation_2025` | Same pattern — replaced fabricated entry with the verified-real one. |
| `elsheikhy_lightweight_2025` | Same pattern. |
| `schmale_curriculum_2025` | Same pattern; also changed `@article`→`@inproceedings` (it's an EMBC conference paper). |
| `zhou_inter-patient_2024` | Same pattern; corrected volume 88→90 per the verifying agent's finding. |
| `bae_handling_2025` | Replaced fabricated entry; also corrected the real entry's own venue field (manuscript/bib said "ICCE 2025," independently confirmed to actually be "ICAIIC 2025"); `@article`→`@inproceedings`. |
| `silva_systematic_2025` | Fixed author first name in both files ("Igor"/"Lucas R." → real first author "Guilherme A. L."); fixed `paper/references.bib`'s fabricated title/volume/pages to the real title (already correct in the IEEE PAPER file); noted the paper is currently only confirmed as arXiv:2503.07276, not (yet) the claimed journal placement. |
| `zhao_mak-net_2025` | Fixed in **both** files — real paper exists but was misattributed to the wrong journal/title/first-author in both. Corrected to the real paper (Zhao, Cong et al., *Sensors* 25(13):3928, DOI 10.3390/s25133928). |
| `mousavi_inter-_2019` | Fixed in **both** files — real paper only exists as an ICASSP 2019 conference paper; both files had invented a fake *Information Sciences* journal identity for it with a DOI resolving to a completely unrelated paper. Corrected to the real ICASSP entry; `@article`→`@inproceedings`. |
| `bengio_curriculum_2009` | Added to `paper/references.bib`, which was missing it entirely despite the manuscript citing it (already present and correct in the IEEE PAPER file). |
| `xiao_deep_2023` | **NOT fixed — flagged instead**, per rule 2 ("if you can't verify it, flag it for me instead of including it"). No real paper matching this specific title/author/claim was found by an active search; its current DOI resolves to a real but completely unrelated paper. Left in place with an explicit `UNRESOLVED` note in both files rather than inventing a replacement citation. **Needs a project-owner decision**: find a real citation for whatever claim this was meant to support, or remove that claim from the manuscript (a Phase 6 action, blocked on this decision). |

**Does not change any reported number** — these are bibliographic metadata corrections, not changes to any claimed result. Three of the corrected citations are used substantively for specific numeric claims in the manuscript (`zhao_mak-net_2025`: 6.1M params/24ms/accuracy figures for the main competing baseline; `mousavi_inter-_2019`: S-recall/memory-footprint claims; `bae_handling_2025`/`schmale_curriculum_2025`: specific percentages/claims in the Discussion) — those specific numbers have **not** been re-verified against the real papers' actual reported figures yet. Flagged for Phase 5, not done here.

### Quarantined (project-owner decision, not a bug fix — rule 6)

- **`find_best_seed.py`, `check_seed_42.py`, `apply_fixes.py`, `apply_all_fixes.py`, `src/verify_win.py` → `landmine_scripts_do_not_use/`.** Each script's entire purpose is the integrity violation `AUDIT_FINDINGS.md` flags (test-set-driven seed selection; mechanical injection of those numbers into the manuscript; post-hoc "decide the claim after seeing the data" framing) — there's no bug to fix without removing that behavior entirely, so per the audit's own instruction ("don't just delete verify_win.py... log it as a proposed fix first"), the disposition was put to the project owner rather than decided unilaterally. Chosen: **quarantine** (over delete / warning-header / leave-untouched). Moved via `git mv` (history preserved) into a new `landmine_scripts_do_not_use/` folder with a `README.md` explaining what's there and why, cross-referencing `AUDIT_FINDINGS.md` C4 (verify_win.py) and H1 (apply_fixes.py/apply_all_fixes.py).
  - **Follow-on fix required and done**: `src/verify_win.py` was actually imported by `src/train_full_pipeline.py` (Stage 12) — quarantining it without touching the call site would have broken every other stage of that pipeline on import. `train_full_pipeline.py`'s top-level `from src.verify_win import ...` was removed, and `stage12_verify_win()`'s body was replaced with a documented no-op that explains the removal instead of crashing. Verified: `python -m py_compile src/train_full_pipeline.py` → syntax OK.
  - Repo-wide grep confirmed no other file imports any of the 4 root-level scripts (they were already standalone, zero-caller scripts per the original Phase 2 audit).
  - **Does not change any reported number** — the manuscript numbers `apply_fixes.py` already wrote in the past are unaffected by quarantining the script itself; whether those numbers get honestly re-derived is a Phase 5/6 question, not something quarantining undoes.

**Still open, not done in this session:**
- `Submission_JBHI/references.bib` still doesn't exist (C7) — placing a corrected bib file into the actual submission folder is a Phase 6 manuscript-package task.
- The `xiao_deep_2023` citation gap (needs a project-owner decision).
- Re-verifying the specific manuscript numbers attributed to the now-corrected citations against the real papers (Phase 5).
- All manuscript-text edits (model-identity conflation C11, the SMOTE contradiction C12, caption fixes, etc.) — explicitly out of scope for Phase 4 per the audit's own rules; deferred to Phase 6, after Phase 5 produces real re-verified numbers.
- `src/eval_fewshot_ptbxl.py` (C8) and `run_audit.py` — not yet fixed or quarantined; lower urgency since neither is currently exploited in the live manuscript.

---

## 2026-08-12 — Phase 5 scoping prep (model-identity decision made: WavKAN_v2, 154,325 params)

### Bug fixes enabling a correct Phase 5 re-run

- **`src/process_data.py`, `src/process_extended_rr.py`, `src/process_sequence.py` (AUDIT_FINDINGS.md H16, also resolves H19).** Was: each independently hardcoded its own `VAL_RECORDS` (`215`/`220`/`223`/`230`), diverging from `src/split.py`'s canonical list (`208`/`209`/`223`/`230`) — the split the manuscript's own Table 1 actually describes. This was accidental duplication drift, not a deliberate choice (`split.py` was already documented elsewhere as canonical), so fixing it is a bug fix, not a methodological change. Fixed: all three now `from split import DS1_RECORDS, VAL_RECORDS, TRAIN_RECORDS, TEST_RECORDS` instead of re-hardcoding. Verified `process_data.py` now produces the exact 18/4/22 split Table 1 describes. Regression test updated (`tests/test_split_integrity.py`) to assert convergence instead of documenting the prior divergence.
  - **Does not change any reported number** (nothing has been re-run yet) — this only affects data these scripts will produce *going forward*.

- **`src/eval_fewshot_ptbxl.py`, `src/train_full_pipeline.py` (AUDIT_FINDINGS.md H11).** Was: defaulted to `data/ptbxl_processed` (reversed word order vs. the producer `process_ptbxl.py` and the already-verified-correct consumer `eval_ptbxl.py`, both of which use `data/processed_ptbxl`) — following the documented default pipeline silently missed the data and fell through to `eval_fewshot_ptbxl.py`'s synthetic-fabrication fallback (C8) instead of erroring. Fixed both to `data/processed_ptbxl`, matching the producer.

- **`test_ablation.py` skip condition hardened.** While verifying the H16 fix by importing `process_data.py` directly, its module-level `os.makedirs(OUT_DIR, exist_ok=True)` created an empty `data/processed_rr_history/` directory as a side effect — which then made `test_ablation.py`'s skip check (`DATA_DIR.exists()`) wrongly conclude real data was present, and both tests failed with `FileNotFoundError` instead of skipping. Removed the accidentally-created empty directory, and changed the skip condition to check for an actual data file (`y_train.npy`) rather than just the directory, which several scripts in this repo create empty as an import-time side effect.

### Feature added for a fair 20-seed comparison

- **`src/train_pca.py`: added `use_curriculum` parameter / `--no-curriculum` CLI flag (addresses H6/H20 for the specific arms Phase 5 will actually run).** `train_pca.py` (Progressive Curriculum Anchoring) had no way to produce a baseline that shares its architecture/optimizer/schedule/epoch-budget/augmentation, differing *only* in the curriculum/sampling strategy — which is what a genuine single-variable ablation needs (Table 3's own stated design intent). `--no-curriculum` now runs the identical setup with natural-distribution sampling and standard class-weighted cross-entropy for every epoch (matching Table 3's stated "Standard CE Loss" baseline description) instead of the warm-up-then-anneal schedule. Existing default behavior (curriculum enabled) is unchanged.
  - **Does not change any reported number** — this is new capability for the upcoming Phase 5 run, not a change to anything previously reported.

### Decision recorded

- **Model identity (AUDIT_FINDINGS.md C3/C11) — DECIDED 2026-08-12 by project owner: WavKAN_v2 (154,325 params, PCWI+PWAM+RR-self-attention) is the canonical submission model going forward.** This does not retroactively change anything already reported (Phase 6 will reconcile the manuscript text once Phase 5 produces real numbers) — it resolves which architecture Phase 5's training runs should target.

See `PHASE5_SCOPE_PLAN.md` for the concrete data-regeneration and training commands this unblocks.

---

## 2026-08-12 — `run_gpu_pipeline.sh` bug found live on the real GPU box, fixed same session

- **`run_gpu_pipeline.sh`: `set -o pipefail` added; the script's own sanity check silently reported success after actually failing.** On the first real run (RTX A4000, remote GPU box), `python3 -m pytest ...` failed with `No module named pytest` (pytest was never listed as a dependency anywhere — see below), but because the command was piped into `tee` for logging, `set -e` alone couldn't see the failure (a pipeline's exit status is its *last* command's, i.e. `tee`'s, which is essentially always 0) — the script printed "✓ Test suite passed/skipped cleanly" and moved on. The *next* pytest call happened to not be piped, genuinely failed, and correctly killed the job via `set -e` — which is the only reason this was noticed at all rather than silently sailing through every subsequent `| tee` in the script the same way. Fixed: added `set -o pipefail` immediately after `set -e`, so any command's failure anywhere in a pipe now actually stops the script.
- **Added a preflight dependency check** (`pytest`, `wfdb`, `neurokit2`, `torch`, `numpy`, `sklearn`, `scipy`) at the top of the script, so a missing package fails immediately with a clear message instead of surfacing confusingly mid-log (or, per the bug above, not surfacing at all).
- **`pytest` added to the install instructions** in `PHASE5_SCOPE_PLAN.md` (both the Quick Start and the detailed Section 0) — same class of gap as the `neurokit2` miss (Phase 5 scoping session): a real dependency the pipeline needs that was never listed anywhere.
- **Does not change any reported number** — this is a fix to the orchestration script's own error-handling, not to any training/eval logic.

- **`src/e3c_compliance.py` / `tests/test_e3c_compliance.py`: fixed a test-isolation bug found live on the real GPU box.** `check_criterion_1`/`check_criterion_3` hardcoded the data path (`data/processed_rr_history`) instead of taking it as a parameter. On a machine that had already run the data pipeline (the GPU box), the tests' `NONEXISTENT_DIR` argument (meant to simulate "no real evidence available") did nothing to the hardcoded path — the functions correctly found the real `ids_train.npy`/`ids_test.npy`/`y_test.npy` that genuinely existed there and returned PASS, which is actually correct behavior; the test's assumption that this data wouldn't exist was wrong, not the underlying logic. Fixed: both functions now take an optional `data_dir` parameter (default unchanged, so no behavior change for existing callers); the two affected tests now pass an explicit, guaranteed-empty `tmp_path`-based directory instead of relying on ambient filesystem state. Added two more tests while fixing this (`test_criterion_1_passes_with_real_disjoint_split_artifacts`, `test_criterion_1_fails_with_real_overlapping_split_artifacts`) covering the PASS and FAIL paths with synthetic real-looking data, which weren't covered before (only the PARTIAL/no-data path was).
  - Verified: `python -m pytest tests/ test_ablation.py -v` → 18 passed, 2 skipped.
  - **Does not change any reported number.**
