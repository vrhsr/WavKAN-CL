# Phase 3 — Manuscript Cross-Reference: Claim-by-Claim Gap Table

**Manuscript audited:** `Submission_JBHI/ieee_manuscript.tex` (488 lines, full read).
**Companion document:** `AUDIT_FINDINGS.md` — this file cross-references that register's IDs rather than repeating their full detail. Read that file for exact file:line evidence.
**No external venue checklist was supplied** (the in-repo `paper/IEEE PAPER/JBHI_SUBMISSION_CHECKLIST.md` is a stale, self-authored, all-green document for an earlier draft — treated as a hypothesis per audit ground rules, not used as the checklist here). This document instead uses a "would a skeptical reviewer's claims survive re-derivation" definition of done: for every table, figure, and load-bearing claim — does it compile, does it trace to real runnable code and real data, is it internally consistent, and is the statistical/fairness framing sound.

Status legend: 🟢 clean/verified · 🟡 traceable in principle but currently unverifiable (data/results absent) · 🔴 contradicts code, contradicts itself, or untraceable · ⚪ needs a project-owner decision, not a fix.

---

## 1. Does the submission package compile?

| Item | Status | Detail |
|---|---|---|
| `\bibliography{references}` (`:473`) | 🔴 | `Submission_JBHI/` has no `references.bib` at all. C7. |
| Fig. 1 `final_methodology_workflow_v2.pdf` (`:103`) | 🔴 | Absent from `Submission_JBHI/`; root-level file unconfirmed as the same content; no generator would write into this repo tree if rerun. H14. |
| Fig. 2 `wavkan_micro_architecture_v2.pdf` (`:123`) | 🔴 | Same as above. H14. |
| Figs. 3–7, remaining `\includegraphics` targets | 🟢 (present as files) | All present in `Submission_JBHI/`; whether their *content* matches their captions is a separate question — see §3. |
| Citations (all `\cite{}` keys) | 🔴 | Even if a bib file existed, 13 of 21 keys have a documented problem (C14) — a naive copy of either candidate `.bib` would still leave several citations wrong or unresolved. |

**Bottom line: the submission package as currently committed cannot be compiled to a clean PDF.** This must be fixed before any resubmission, independent of every other finding below.

---

## 2. Internal contradictions (manuscript vs. itself)

| # | Contradiction | Register ID |
|---|---|---|
| 1 | Two model identities used interchangeably throughout the prose — every "WavKAN-CL" mention pairs with 95,189 params, every "WavKAN-v2" mention pairs with 154,325 params, sometimes in the same paragraph with different V-Recall values attached to each (`:375` is the sharpest example: body text says "WavKAN-CL... 95,189 parameters (0.38 MB)... V-Recall of 0.86," the table two sentences later says "WavKAN-v2... 0.60 MB," V-Recall 0.85). | **C11** |
| 2 | SMOTE's effect on S-Recall is stated two different, incompatible ways: `:375` says it *reduces* S-Recall (0.290→0.245); `:414` says it *achieves the highest* S-Recall (≈0.46). | **C12** |
| 3 | Table 4's caption claims "evaluated across multiple random seeds"; 5 of its 8 rows are confirmed single-seed (seed 42 only) in the generating code. | **C13** |
| 4 | Table 1's stated DS1 train/val split matches `src/split.py`, but the actual pipeline that produces the RR-history-based results (Tables 4/5/6, Figs 3/4/6) may use a *different* validation-record list (`process_extended_rr.py`'s independently hardcoded `VAL_RECORDS`) — unconfirmed which pipeline produced which published number. | **H19** (uncertain, needs execution to confirm) |
| 5 | Fig. 3's caption describes a 5-model, Wilcoxon+Cohen's-d+Levene's-test comparison; the actual embedded PDF shows one model's data and none of the statistical annotations. | **C1** |
| 6 | Fig. 4's caption describes a 50-epoch, two-curve Baseline-vs-Curriculum comparison with the phase transition at epoch 15; the actual embedded PDF shows one model, ~37 epochs, phase transition at ~epoch 26. | **C1** |
| 7 | RR-ablation figure text (`:420`) claims the $t-2$ interval's degradation is uniquely "statistically significant"; every bar in the figure carries the same mechanical p-floor (p=0.031) regardless of position, so significance alone cannot support that ranking (the effect-size ranking might still be valid — the *significance* framing is what's unsupported). | **H18** |

---

## 3. Table-by-table and figure-by-figure traceability

| Ref | Claim | Status | Detail / Register ID |
|---|---|---|---|
| Table 1 (`tab:split_protocol`) | DS1/DS2 record-level split, no patient overlap | 🟡 | Mechanism correct, matches `split.py` — but see contradiction #4 above re: which pipeline it actually describes. |
| Table 2 (`tab:class_dist`) | Class counts (N/S/V/F/Q × train/val/test) | 🟡 | Internally consistent with `verify_manuscript.py`'s hardcoded reference values — but `data/` is absent from this checkout, so the underlying `.npy` files cannot currently be re-counted to confirm. No contradiction found; just currently unverifiable. |
| Table 3 (`tab:baselines`) | Clean single-variable ablation ladder (morphology / rhythm / training strategy) | 🔴 | The rows differ in more than the labeled variable — different optimizer, epoch count, curriculum phasing between arms. **H6 / H20.** |
| Table 4 (`tab:ablation`) | Wavelet-basis and structural ablation, "across multiple random seeds" | 🔴 | 5 of 8 rows are single-seed (n=1), from a curriculum the codebase's own later script calls "the original broken curriculum from v1." **C9, C13.** |
| Table 5 (`tab:gap`) | Val/test generalization gap (0.06→0.00) | 🔴 | No code anywhere persists both val and test metrics in a form that could produce this specific comparison. **H7 (training cluster), C10.** |
| Table 6 (`tab:stats`) | 20-seed Wilcoxon comparison, V-Recall 0.871 vs 0.898, p<0.05 | 🔴 | No training script anywhere produces the matching 20-seed Baseline+WavKAN-CL checkpoint pair this table would require. **C5.** |
| Fig. 3 (`fig:seed_stability`) | 5-model, n=5-seed comparison w/ significance stars, Cohen's d, Levene's test | 🔴 | Rendered PDF shows 1 of 3 labeled categories populated, no stats annotations at all. **C1 — verified by directly rendering the PDF.** |
| Fig. 4 (`fig:convergence`) | Baseline-vs-curriculum, 50 epochs, phase transition at epoch 15 | 🔴 | Rendered PDF shows one model, ~37 epochs, transition at ~epoch 26. **C1 — verified by directly rendering the PDF.** |
| Table 7 (`tab:per_class`) | "Best-seed" per-class metrics, V-Recall 0.858 | 🔴 | Disclosed as "best-seed" but the actual generating artifact (`results/thesis_fusion_metrics.json`) is an 11-checkpoint, 2-architecture, hand-weighted ensemble, not one seed. **C2.** |
| Fig. 5 (`fig:confusion`) | Normalized confusion matrix, DS2 test | 🔴 | No generating script conclusively identified across any of the six Phase-2 clusters; the one same-named candidate script was hash-confirmed to produce a *different* file. **H21.** |
| Table 8 (`tab:ptbxl_zero_shot`, referenced inline as PTB-XL results) | Zero-shot PTB-XL, Macro-F1 0.233, per-class numbers | 🟢 | Confirmed: `eval_ptbxl.py` is genuinely zero-shot (no `.fit`/`.train`/`.backward` calls), and its real output matches the manuscript's numbers exactly. **C8 (positive half).** |
| Table 9 (`tab:multidataset`) | Zero-shot INCART/SVDB, >300% avg Macro-F1 improvement | 🟡 | Mechanism confirmed genuinely zero-shot (`eval_multidataset.py`, no train calls) — but no `results/multidataset_table/` JSON exists in this checkout to confirm these exact published per-dataset numbers. |
| Fig. 6 (`fig:rr_ablation`) | Leave-one-out RR-interval ablation, $t-2$ dominant | 🔴 | Real generating path (`<results_base>/rr_ablation/rr_ablation_figure.pdf`) doesn't exist anywhere in current `results/`; a hardcoded-literal decoy script targeting the same filename was demonstrably run at some point instead. **H17, H18.** |
| Table 10 (`tab:sota_comparison`) | Comparison with 5 recent studies + proposed model | 🟡 / 🔴 | Table itself is honest about intra- vs inter-patient protocol distinction (positive) — but see citation issues below (`zhao_mak-net_2025` metadata wrong, C14) and the same-section internal contradiction (#1 above, C11) about which parameter count/V-Recall belongs to the row. |
| Fig. 7 (`fig:wavelets`) | Learned Mexican Hat wavelets | 🟢 (currently) | Embedded version traces correctly to a real checkpoint (`wavelet_alignment_score.py`) — but a same-purpose, same-filename **fabricated** decoy also exists and was demonstrably run once (root-level file hash mismatch confirms this). Currently fine; live landmine for a future re-run. |
| Discussion — WAS score (Macro-WAS 0.971) | Wavelet-ECG Alignment Score | 🟢 | Verified directly against `results/was_scores.json` (0.9713 → rounds to published 0.971; per-component figures match too). One of the few claims in the entire manuscript with a clean, currently-checkable paper trail. |
| Discussion — Clinical operating point (AUPRC 0.7399, AUROC 0.9503, threshold 0.032, 85.18% specificity) | AHA Class I sensitivity analysis | 🟡 | Generating code (`src/clinical_story.py`) is legitimate — real `sklearn.roc_curve`/`average_precision_score` on real arrays, not hardcoded — but no output JSON exists in this checkout to confirm these exact published figures, and the script's own docstring illustrative example differs substantially (AUPRC 0.923, specificity 94.0%). |
| Discussion — E3C compliance, "mere 4.1% of studies" (`:405`) | Clinical evaluation rigor claim | 🔴 | Checker mechanism hardcodes PASS for 3 of 4 criteria regardless of input (**C6**); underlying statistic is real (arXiv:2503.07276) but its citation key resolves to wrong author metadata in both candidate bib files (**C7, C14**). |
| Discussion — ECE (0.022 overall, 0.862 V-class) | Calibration claim | 🟡 | `temperature_scaling.py` exists and computes ECE genuinely (not checked in exhaustive detail this pass) — no output JSON confirms the exact published figures in this checkout. Flagged for Phase 4/5 spot-check, not a confirmed contradiction. |
| Data/Code Availability (`:469-471`) | "Source code and pre-trained weights are available... to ensure full reproducibility" (cover letter) / GitHub link (manuscript) | 🔴 | Materially overstated as currently committed — data pipeline requires PhysioNet regeneration, headline-checkpoint training code doesn't exist in the repo, two figures and the bibliography are missing from the submission folder, `verify_manuscript.py` cannot execute. **H22.** |

---

## 4. Statistical rigor and baseline fairness — explicit checks

| Claim | Verdict | Detail |
|---|---|---|
| "Wilcoxon signed-rank test with 20 random seeds" (`:253`, Table 6) | 🔴 untraceable | No script produces the matching 20-seed pair (C5); pairing-by-seed-identity not confirmed for this specific table. |
| Fig. 3's Wilcoxon + Cohen's d + Levene's test apparatus | 🔴 not present in the actual figure | See §2, contradiction #5. |
| Paired-test seed-alignment risk elsewhere in the codebase | 🔴 confirmed bug (not in the specific live table's path, but a real, demonstrated failure mode in sibling code) | `generate_all_figures.py`'s Wilcoxon call checks equal list *length*, not equal seed *identity* (H8a); dead-but-same-naming-convention `statistical_validation.py` does the same with no ID check at all (H8b). |
| One-sided vs. two-sided test, same nominal comparison | 🔴 confirmed inconsistency | One live call site hardcodes a one-sided test favoring "ours > baseline"; the pipeline's actual Stage-4 call uses two-sided. H8d. |
| p<0.05 claimed at n=5 (`eval_multidataset.py`'s own caption) | 🔴 mathematically impossible | Minimum achievable exact two-sided p-value at n=5, no ties, is ≈0.0625. H8c. |
| Baseline vs. WavKAN-CL hyperparameter parity (Table 3's implicit claim) | 🔴 not matched | Different optimizer, epoch count, curriculum phasing between arms. H6/H20. |
| "The B-Spline KAN is evaluated under identical curriculum settings" (Fig. 3 caption, `:317`) | 🟡 unconfirmed | Plausible given `run_ablation_study.py`'s shared `train_one_config` path for the *internal wavelet-type* ablation specifically (this one path was checked clean in Phase 2) — but Fig. 3 itself doesn't render the B-Spline data at all (C1), so the claim is currently unverifiable from the artifact meant to demonstrate it. |
| Within-run checkpoint selection (early stopping) | 🟢 clean | Validation-based, not test-based, across all 10 training scripts audited. Positive finding, Phase 2 register. |

---

## 5. Strong-claim check ("first," "state of the art," ">Nx faster")

| Claim | Verdict |
|---|---|
| "We do not claim overall classification superiority on this metric" (`:407`) | 🟢 Responsible, explicit hedge — the manuscript does **not** overclaim aggregate SOTA. Positive finding, worth preserving in any rewrite. |
| "First KAN-based architecture with wavelet edge functions under the strict inter-patient DS1/DS2 protocol" (`:61`) | 🟡 Plausible given the related-work review, not independently fact-checked against the full KAN-ECG literature — lower priority than the fabricated-citation findings above. |
| ">30x faster than MAK-Net (24 ms/beat)" (`:375`) | 🔴 Not a same-hardware comparison — WavKAN's number is measured on CPU in this repo; MAK-Net's 24ms figure is that paper's own reported number, likely on different hardware, and per the citation sweep (C14) the MAK-Net bib entry itself has wrong metadata, so even the reference point needs re-confirming against the real paper. H9. |
| ">95% parameter reduction vs. MAK-Net (6.1M params)" | 🟡 Directionally very likely true regardless of the exact real MAK-Net parameter count, but the real paper's actual reported parameter count has not been re-confirmed given C14's finding that the citation metadata was wrong. |
| "Only 4.1% of studies" satisfy E3C | 🔴 See E3C row in §3 — real underlying statistic, broken checker, wrong citation metadata. |

---

## 6. Citation integrity — summary

Full detail in `AUDIT_FINDINGS.md` **C14**. Headline: of 21 citation keys used in the manuscript, **9 involve fabricated or substantially wrong bibliographic content** (not typos), and a further 4 have minor key/label bugs. Three of the fabricated ones attach a real-format DOI to a real but completely unrelated paper — the single hardest fabrication pattern to catch by inspection, and the exact failure mode `check_citations.py` (H4) is structurally incapable of detecting (it only checks that a `\cite{}` key has *some* matching `.bib` entry, never that the entry is truthful). The foundational `chazal_automatic_2004` citation is fully verified correct — the paper's core methodological grounding is not in question, only large parts of its related-work and comparative framing.

---

## 7. Phase 3 sign-off checklist

Per the audit's own ground rules (rule 5: stop and get sign-off between phases), this is what needs a decision before Phase 4 (fixing) begins:

- [ ] **Model identity (C3/C11):** is "WavKAN-CL" (95,189 params) or "WavKAN-v2" (154,325 params) the model being submitted? This determines which numbers throughout the paper are corrected vs. removed.
- [ ] **SMOTE contradiction (C12):** which number is right — S-Recall drops to 0.245, or rises to ≈0.46? This needs the actual experiment re-run or the original log recovered; it cannot be resolved by reasoning about the text alone.
- [ ] **Fig. 3 / Fig. 4 / Fig. 6 (C1, H17):** given their generating code paths are confirmed broken or untraceable, do these get regenerated from a real run, or does the underlying 5-seed/20-seed experiment need to be redone first? This is a Phase 4/5 sequencing question, not a Phase 3 one, but it's the single biggest driver of how much re-training this revision needs.
- [ ] **Citation list (C14):** confirm the correction plan — for the 6 "Candidate B is real" citations, adopt `paper/IEEE PAPER/references.bib`'s versions; for `zhao_mak-net_2025` and `mousavi_inter-_2019`, correct the metadata to the real paper found; for `xiao_deep_2023` and `akan_ecgformer_2024`, find a real replacement citation for the claim being supported, or remove the claim if none exists.
- [ ] **Reproducibility framing (H22):** decide how much of the "fully reproducible on GitHub" framing (cover letter, Data/Code Availability) should be softened versus how much Phase 4/5 effort goes into actually making it true before resubmission.

No numbers have been changed. No files other than `AUDIT_FINDINGS.md` and this document have been modified. Awaiting sign-off to proceed to Phase 4.
