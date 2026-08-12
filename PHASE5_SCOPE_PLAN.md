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
chmod +x run_gpu_pipeline.sh
nohup bash run_gpu_pipeline.sh 2>&1 | tee pipeline_log.txt &
```

That single script (`run_gpu_pipeline.sh`, repo root) is the executable version of everything in Sections 1-3 below. Sections 1-3 are the narrative explanation of what it does and why; Sections 4-5 cover what it does **not** yet do (PTB-XL/INCART/SVDB, Fig. 3/4/6 regeneration) — those still need to be run/scripted separately, see below.

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

## 4. Fig. 3 / Fig. 4 / Fig. 6 — what they need

These three are the ones C1/H17 found to be currently broken or untraceable in the live manuscript. Given the compute is now happening for real, they can be regenerated honestly:

- **Fig. 3 (seed stability)**: needs all 5 comparator models (WavKAN-v2, ResNet1D, Transformer, CNN+Focal Loss, B-Spline KAN) trained with a matched protocol, not just WavKAN-v2. `src/baselines_extended.py` has ResNet1D/Transformer/B-Spline KAN training code — **not yet audited for whether its protocol now matches `train_pca.py`'s** (same epochs/optimizer/schedule). Flag this back to me before trusting a re-generated Fig. 3 — this needs a fairness check I haven't done yet, separate from the H6 fix already applied to the baseline/curriculum pair above.
- **Fig. 4 (convergence)**: `training_history.json` is already saved by `train_pca.py` for every seed/arm (Section 2 above) — plotting baseline-vs-curriculum training loss/Macro-F1/V-recall/S-recall across epochs from any one matched seed pair (e.g. seed 42 of each arm) directly gives the real two-curve comparison the caption describes. No new training needed once Section 2 is done — just a plotting script, which I can write once real `training_history.json` files exist to test against.
- **Fig. 6 (RR-ablation)**: needs a real leave-one-out run (`src/rr_ablation.py` — confirmed in Phase 2 as "the methodologically cleanest script in this cluster," consumes pretrained checkpoints only, no retraining confound) against the new WavKAN_v2 checkpoints from Section 2, across at least 5 seeds. I can prepare the exact command once Section 2's checkpoints exist.

I've deliberately not scripted these three yet — they depend on Section 2's checkpoints existing first, and Fig. 3 specifically needs a fairness audit of `baselines_extended.py` I haven't done. Flag back to me once Section 2 is done and I'll scope these three concretely.

---

## 5. Cross-dataset data (PTB-XL / INCART / SVDB) — lower priority, do after Section 2-3

Only needed for Table 9 (multi-dataset) and the PTB-XL zero-shot table (already confirmed honest and working — see `AUDIT_FINDINGS.md` C8 positive finding — it just needs real data and the retrained checkpoint to run against):

```bash
python -c "import wfdb; wfdb.dl_database('ptb-xl', 'data/ptbxl')"   # large: ~1.7GB
python src/process_ptbxl.py --data-dir data/ptbxl --out-dir data/processed_ptbxl
python src/eval_ptbxl.py --data-dir data/processed_ptbxl --model-path results/wavkan_v2_curriculum/seed_42/best_model.pth
```

INCART/SVDB are smaller and lower-priority; commands available on request once the above is working.

---

## 6. What NOT to do yet

- Don't touch `Submission_JBHI/ieee_manuscript.tex` — that's Phase 6, after real numbers exist.
- Don't run anything in `landmine_scripts_do_not_use/` to "check" these results against a target — the whole point of this re-run is to report the honest distribution, not select a seed that matches a preconceived number.
- If any seed's run crashes or produces a clearly-broken result (e.g., NaN loss), don't silently exclude it and don't silently retry with a different seed — report it back to me as-is; `aggregate_seed_results.py` already surfaces missing seeds explicitly rather than hiding them.
