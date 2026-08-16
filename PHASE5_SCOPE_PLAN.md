# Phase 5 Scope Plan — Data Regeneration & Retraining

**Status: ready to execute.** Every script referenced below has been fixed and is currently believed correct as of 2026-08-12 (see `CHANGELOG.md`). This document is the exact command sequence to run on your GPU infrastructure. Bring the resulting `results/` directory (or just the JSON/PDF outputs) back and I'll verify and integrate it into Phase 5's re-verification and Phase 6's manuscript update.

**Decided (project owner, 2026-08-12):** the canonical model is **WavKAN_v2** (154,325 params, PCWI+PWAM+RR-self-attention). Everything below targets that architecture.

**All of this is already committed and pushed to `origin/main`** (verified 2026-08-12: 0 uncommitted changes, 0 commits ahead/behind `origin/main`). You do not need to manually carry anything over from this checkout — a fresh `git clone` on your GPU machine gets everything below. `data/` and any prior local `results/...` subfolders from earlier experimentation are either gitignored or predate this audit's fixes — nothing there needs to be preserved; Sections 1-2 below regenerate/produce it all under new, non-colliding directory names.

---

## Quick start (fresh machine, one command to a full 20-seed run)

```bash
# 1. Fresh clone -- do this instead of reusing an old checkout, it's zero-risk
git clone git@github-vrhsr:vrhsr/WavKAN-CL.git
cd WavKAN-CL

# 2. Python environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install neurokit2 pytest     # both missing from requirements.txt (real gaps found during Phase 5 scoping --
                                  # neurokit2 is a runtime dependency of the data pipeline, pytest is needed
                                  # for run_gpu_pipeline.sh's own preflight/sanity checks)

# 3. Run everything: GPU check -> sanity tests -> data regen -> smoke test -> full
#    20-seed x 2-arm training -> aggregation. See run_gpu_pipeline.sh for exactly
#    what each step does and why; it pauses 10s after the smoke test so you can
#    Ctrl+C before committing to the full run.
#
#    Launched via the supervisor (recommended), not run_gpu_pipeline.sh directly --
#    the supervisor bounded-retries the whole pipeline (up to 5x) if the orchestrator
#    itself dies unexpectedly. This is safe/cheap because of two resume layers added
#    2026-08-13: train_pca.py checkpoints every epoch and auto-resumes mid-seed, and
#    run_gpu_pipeline.sh skips seeds that are already complete -- a retry re-does only
#    the fast setup steps, not all 40 runs. See run_gpu_pipeline_supervisor.sh.
chmod +x run_gpu_pipeline.sh run_gpu_pipeline_supervisor.sh pipeline_status.sh
nohup bash run_gpu_pipeline_supervisor.sh 2>&1 | tee supervisor_log.txt &

# 4. Monitor from a separate pane/session anytime -- read-only, doesn't touch the run
bash pipeline_status.sh
```

That single script (`run_gpu_pipeline.sh`, repo root) is the executable version of everything in Sections 1-3 below. Sections 1-3 are the narrative explanation of what it does and why; Sections 4-5 cover what it does **not** yet do (PTB-XL/INCART/SVDB, Fig. 3/4/6 regeneration) — those still need to be run/scripted separately, see below.

**Resilience (added 2026-08-13, after two real crashes during this session):** three independent layers, each covering a different failure scope --
1. **Per-epoch** (`train_pca.py`): checkpoints after every epoch, auto-resumes on restart. A crash at epoch 60/100 costs at most 1 epoch, not the whole seed. See `tests/test_train_pca_resume.py` for the actual verified behavior (resume-from-injected-checkpoint, corrupted-checkpoint fallback, at-cap edge case).
2. **Per-seed** (`run_gpu_pipeline.sh`): a seed that fails no longer kills the other 39 -- it's logged, recorded, and the sweep continues. Failures are summarized at the end.
3. **Per-script** (`run_gpu_pipeline_supervisor.sh`): if the orchestrator itself dies (not an individual seed), bounded-retries up to 5 times with a 60s backoff, then gives up loudly rather than looping forever on a real, persistent bug.

None of this changes what gets reported -- a resumed/retried seed is still a real, honestly-trained result on real data, just not guaranteed bit-identical to what an uninterrupted run of the same seed would have produced (the RNG trajectory across an interruption differs). Don't read more precision into a resumed seed than that.

---

## 0. Environment setup (detail)

```bash
pip install -r requirements.txt
pip install neurokit2   # used by process_data.py / process_extended_rr.py / process_sequence.py,
pip install pytest      # used by run_gpu_pipeline.sh's own sanity checks --
                         # both missing from requirements.txt (real gaps found while scoping this;
                         # the pytest gap caused a real, confusingly-masked failure on the first live
                         # GPU run -- see CHANGELOG.md)
```

Confirm GPU is visible before starting anything long-running:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 1. Regenerate MIT-BIH data

*(This and Section 2 are what `run_gpu_pipeline.sh` runs automatically — read on if you want to run steps individually, debug one seed, or understand what the script is doing.)*

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw')"
python src/process_data.py
```

`process_data.py` now imports its train/val/test record lists from `src/split.py` (fixed 2026-08-12, was previously diverging — see `AUDIT_FINDINGS.md` H16), so this will produce the exact 18/4/22-record split the manuscript's Table 1 describes. Output: `data/processed_rr_history/{X,X_rr,y,ids}_{train,val,test}.npy`, `class_weights.npy`. This step is CPU-only and should take well under an hour (MIT-BIH is ~200MB raw).

**Sanity-check before training anything:**
```bash
python -m pytest tests/test_split_integrity.py -v
python -c "
import numpy as np
for split in ['train', 'val', 'test']:
    y = np.load(f'data/processed_rr_history/y_{split}.npy')
    print(split, len(y), np.bincount(y))
"
```
Expected counts (from the manuscript's Table 2 — flag me if these don't match, since Table 2's numbers haven't been independently re-verified against real data yet): train 40,726 (N=36,396 S=773 V=3,150 F=399 Q=8), val 10,266 (N=9,443 S=170 V=638 F=15 Q=0), test 49,684 (N=44,232 S=1,837 V=3,220 F=388 Q=7).

---

## 2. Train both arms, 20 seeds each

Using the same 20 seed values already used elsewhere in this repo for continuity (`src/fusion_engine.py`'s seed list) — there's nothing special about these values, reusing them just avoids introducing yet another arbitrary seed list into the project:

```
SEEDS="42 101 777 2026 9999 1234 2024 31415 27182 7 11 13 888 555 333 99 1001 5050 8080 1998"
```

**Curriculum arm** (WavKAN_v2 + Progressive Curriculum Anchoring):
```bash
for seed in $SEEDS; do
  python src/train_pca.py --seed $seed --epochs 100 \
      --output-dir results/wavkan_v2_curriculum/seed_$seed \
      --data-dir data/processed_rr_history
done
```

**Baseline arm** (identical architecture/optimizer/schedule/epochs/augmentation, differing ONLY in the curriculum/sampling strategy — this is the `--no-curriculum` flag added 2026-08-12 specifically to make this a fair, single-variable comparison; see `AUDIT_FINDINGS.md` H6/H20):
```bash
for seed in $SEEDS; do
  python src/train_pca.py --seed $seed --epochs 100 --no-curriculum \
      --output-dir results/wavkan_v2_baseline/seed_$seed \
      --data-dir data/processed_rr_history
done
```

Both arms checkpoint on validation Macro-F1 only (never touches test labels until the final, single test-set evaluation after training — verified in `train_pca.py`'s code, not just its comments). Expect early stopping well before 100 epochs for most seeds (`--patience 15` default).

**Runtime estimate:** unknown without knowing your GPU, but `train_full_pipeline.py`'s own docstring estimates ~6-12 CPU-hours for 3 seeds/100 epochs on a similar-scale model — a single modern GPU should bring a single seed/100-epoch run down to well under an hour typically for a model this small (154K params, 360-sample input). 40 total runs (20 seeds × 2 arms). Recommend running a single seed of each arm first as a smoke test before committing to all 40.

---

## 3. Aggregate honestly

```bash
python src/aggregate_seed_results.py \
    --baseline-dir results/wavkan_v2_baseline \
    --curriculum-dir results/wavkan_v2_curriculum \
    --output results/wavkan_v2_20seed_comparison.json
```

This is a new script (`src/aggregate_seed_results.py`, written and regression-tested 2026-08-12 — see `tests/test_aggregate_seed_results.py`) written specifically to avoid the seed-pairing bugs found elsewhere in the codebase (H8): it pairs by seed *value*, not list position or length, reports any seed present in one arm but not the other instead of silently dropping it, and always runs a two-sided (not directionally-biased) paired Wilcoxon test, skipping it cleanly rather than fabricating a p-value if fewer than 6 seed-pairs exist.

**Send this JSON back to me** — it's the real replacement for the manuscript's Table 5/6.

---

## 4. Fig. 3 / Fig. 4 / Fig. 6 — status as of 2026-08-13

Section 2's real run completed (20/20 seeds, both arms — `results/wavkan_v2_20seed_comparison.json`). Since then:

- **Fig. 4 (convergence): DONE.** `src/generate_fig4_convergence.py` reads the real per-seed `training_history.json` and produces a genuine two-curve comparison — tested against real seed-42 data, output verified by direct rendering. Re-run anytime with `python src/generate_fig4_convergence.py`.
- **Fig. 3 (seed stability): script ready, data not yet generated.** `baselines_extended.py`'s protocol now reasonably matches `train_pca.py`'s real hyperparameters (H6/H20 — see `AUDIT_FINDINGS.md`), so the fairness blocker is resolved. `src/generate_fig3_seed_stability.py` (new) reads real per-seed data with correct seed-identity pairing (regression-tested). Needs the 4 baseline models actually trained first — see `run_gpu_pipeline_phase5b.sh` below.
- **Fig. 6 (RR-ablation): script confirmed ready, not yet run.** `src/rr_ablation.py` was already correctly built (real `WavKAN_v2`, real checkpoints, proper stats) — just needs to actually run against the new checkpoints. One command, in `run_gpu_pipeline_phase5b.sh`.

**Run everything left in one pass:**
```bash
chmod +x run_gpu_pipeline_phase5b.sh
nohup bash run_gpu_pipeline_phase5b.sh 2>&1 | tee phase5b_log.txt &
```
This covers Fig. 3's 4 baseline models (5 seeds each), Table 4's ablation matrix (6 configs × 5 seeds, via `train_pca.py`'s own flags — not the old C9-flagged ablation pipeline), Fig. 6, and PTB-XL zero-shot (if `data/processed_ptbxl` already exists). Same resilience pattern as the main run: skip-if-complete per job, log-and-continue on a single job's failure, summary at the end. See the script's own header comment for exact detail.

---

## 5. Cross-dataset data (PTB-XL / INCART / SVDB)

`src/eval_ptbxl.py` was fixed 2026-08-13 (H25) — it hardcoded the old 95K architecture, which would have failed loading a real `WavKAN_v2` checkpoint. Now fixed and verified against a real downloaded checkpoint locally.

**`--sampling-rate 100` is required, not optional** (added 2026-08-16): the manuscript's own text describes this evaluation as "the PTB-XL database (100 Hz...)", but `process_ptbxl.py --sampling-rate` defaults to `500` (both variants get resampled to the same 360-sample MIT-BIH-matching window internally, so this isn't a correctness bug either way — but running with the default would silently create a mismatch between what the manuscript claims was done and what was actually run, the exact class of gap this whole audit exists to catch).

```bash
python -c "import wfdb; wfdb.dl_database('ptb-xl', 'data/ptbxl')"   # large: ~1.7GB
python src/process_ptbxl.py --data-dir data/ptbxl --out-dir data/processed_ptbxl --sampling-rate 100
python src/eval_ptbxl.py --data-dir data/processed_ptbxl --model-path results/wavkan_v2_curriculum/seed_42/best_model.pth
```

`run_gpu_pipeline_phase5b.sh` will pick up PTB-XL eval automatically if `data/processed_ptbxl` already exists by the time it runs. INCART/SVDB (`eval_multidataset.py`, already correctly using `WavKAN_v2`) — commands available on request once PTB-XL is working.

---

## 6. What NOT to do yet

- Don't touch `Submission_JBHI/ieee_manuscript.tex` — that's Phase 6, after real numbers exist.
- Don't run anything in `landmine_scripts_do_not_use/` to "check" these results against a target — the whole point of this re-run is to report the honest distribution, not select a seed that matches a preconceived number.
- If any seed's run crashes or produces a clearly-broken result (e.g., NaN loss), don't silently exclude it and don't silently retry with a different seed — report it back to me as-is; `aggregate_seed_results.py` already surfaces missing seeds explicitly rather than hiding them.
