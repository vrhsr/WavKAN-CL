#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# WavKAN-CL: Unified GPU Pipeline (Smoke Test → Full Training → Aggregation)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phase 1: SMOKE TEST   — seed=42 only, 5 epochs, both arms, verify checkpoints save
# Phase 2: FULL TRAINING — 20 seeds x 2 arms (curriculum vs. fair baseline), up to
#                          100 epochs each (early-stop patience=15)
# Phase 3: AGGREGATION  — seed-paired stats, no H8 pairing bugs (src/aggregate_seed_results.py)
#
# Model: WavKAN_v2 (154,325 params, PCWI+PWAM+RR-attn) -- decided as the canonical
# submission model 2026-08-12, AUDIT_FINDINGS.md C3. See PHASE5_SCOPE_PLAN.md for
# the full narrative version of this same plan and what's still NOT covered here
# (PTB-XL/INCART/SVDB cross-dataset eval, Fig. 3/4/6 regeneration -- Fig. 3
# specifically needs a fairness audit of the other baseline models that hasn't
# been done yet, so it's deliberately not scripted below).
#
# Usage:
#   chmod +x run_gpu_pipeline.sh
#   nohup bash run_gpu_pipeline.sh 2>&1 | tee pipeline_log.txt &
#
# Lives at repo root, not experiments/ -- experiments/ is in .gitignore (meant for
# run *output*, not checked-in scripts), so anything placed there would silently
# never be committed and would be missing after a fresh clone.
#
# Resumability (updated 2026-08-13 after two real crashes during this Phase 5 session):
#   - Per-EPOCH: train_pca.py now saves resume_checkpoint.pth after every epoch and
#     auto-resumes from it (model/optimizer/scheduler/history/best-so-far), so a crash
#     at epoch 60/100 costs at most one epoch of progress, not the whole seed. Written
#     via temp-file + atomic rename, and a corrupt checkpoint is caught and logged
#     rather than crashing the seed again. See train_pca.py's own comments and
#     tests/test_train_pca_resume.py.
#   - Per-SEED (this script): a seed whose test_metrics.json already exists is skipped
#     entirely (unchanged). NEW: a seed that FAILS (nonzero exit, or exits 0 but never
#     produces test_metrics.json) no longer kills the rest of the 40-run sweep -- it's
#     logged, recorded, and the script moves on to the next seed. A failed seed's
#     resume_checkpoint.pth (if any) is left in place, so simply re-running this whole
#     script later picks that seed back up from wherever it got to, not from scratch.
#   - Per-SCRIPT: if run_gpu_pipeline.sh itself dies (not an individual seed, the whole
#     process), that's what run_gpu_pipeline_supervisor.sh is for -- a bounded-retry
#     wrapper. See that file.
# Honesty note (also in train_pca.py): a resumed seed is not guaranteed bit-identical
# to an uninterrupted run of the same seed value -- it's still a real, honestly-trained
# result, just not promised to be bitwise-reproducible across an interruption.
# ═══════════════════════════════════════════════════════════════════════════════

set -e          # Exit on any error
set -o pipefail # Fixed 2026-08-12, found live on the real GPU box: without this,
                # `cmd | tee file` masks cmd's exit status with tee's (which is
                # almost always 0) -- `set -e` alone never sees cmd fail. This is
                # exactly what happened on the first pytest run below: it failed
                # with "No module named pytest", but because it was piped into
                # tee, the script printed "Test suite passed" anyway and kept
                # going. With pipefail, any command's failure anywhere in a pipe
                # now actually stops the script, for every `| tee` used below.

# ─── Preflight: dependencies this script assumes are installed ──────────────
echo "[Preflight] Checking required Python packages..."
MISSING_PKGS=""
for pkg in pytest wfdb neurokit2 torch numpy sklearn scipy; do
    python3 -c "import ${pkg}" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} ${pkg}"
done
if [ -n "$MISSING_PKGS" ]; then
    echo "  ✗ Missing packages:${MISSING_PKGS}" >&2
    echo "  Run: pip install -r requirements.txt pytest neurokit2" >&2
    echo "  (both pytest and neurokit2 are real gaps in requirements.txt -- see" >&2
    echo "  CHANGELOG.md / PHASE5_SCOPE_PLAN.md -- install them explicitly for now)" >&2
    exit 1
fi
echo "  ✓ All required packages importable"

# ─── Configuration ───────────────────────────────────────────────────────────
export PYTHONPATH="$PWD"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

DATA_DIR="data/processed_rr_history"
EPOCHS_FULL=100
SEEDS=(42 101 777 2026 9999 1234 2024 31415 27182 7 11 13 888 555 333 99 1001 5050 8080 1998)
# Reused from src/fusion_engine.py for continuity across the project -- there is
# nothing special about these specific 20 values, and no seed here was chosen by
# looking at any result (that's the exact thing AUDIT_FINDINGS.md's quarantined
# find_best_seed.py/check_seed_42.py did wrong).
FAILED_RUNS=()  # populated by run_arm() below; reported at the end, not hidden

echo "================================================================"
echo "WavKAN-CL: Unified GPU Pipeline"
echo "Started at: $(date)"
echo "================================================================"

# ─── GPU Check ───────────────────────────────────────────────────────────────
echo ""
echo "[0/5] Checking GPU availability..."
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  [WARNING] No GPU detected via torch.cuda.is_available()!"
    echo "  20 seeds x 2 arms x up to 100 epochs will be very slow on CPU."
    echo "  Press Ctrl+C in 10s to abort..."
    sleep 10
else
    python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
fi

# ─── Sanity: confirm this checkout actually has the Phase 4 fixes ────────────
echo ""
echo "[Sanity] Running test suite (confirms this checkout is the fixed one, not a stale copy)..."
python3 -m pytest tests/ test_ablation.py -q 2>&1 | tee "$LOG_DIR/sanity_pytest.log"
echo "  ✓ Test suite passed/skipped cleanly"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[1/5] Processing MIT-BIH..."

if [ ! -f "${DATA_DIR}/y_train.npy" ]; then
    if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
        echo "  Downloading MIT-BIH from PhysioNet..."
        python3 -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw')" 2>&1 | tee "$LOG_DIR/data_download.log"
    fi
    echo "  Running process_data.py (imports the canonical split from src/split.py --"
    echo "  AUDIT_FINDINGS.md H16, fixed 2026-08-12; previously diverged silently)..."
    python3 src/process_data.py 2>&1 | tee "$LOG_DIR/data_process.log"
else
    echo "  ${DATA_DIR} already populated. ✓"
fi

python3 -m pytest tests/test_split_integrity.py -q
echo "  ✓ Split-integrity tests re-confirmed"

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: run one arm (curriculum or fair-baseline) across all seeds, skipping
# any seed that already has a complete result.
# ═══════════════════════════════════════════════════════════════════════════════
run_arm () {
    local arm_name="$1"      # "curriculum" or "baseline"
    local extra_flag="$2"    # "" or "--no-curriculum"
    local epochs="$3"
    local out_root="results/wavkan_v2_${arm_name}"

    for seed in "${SEEDS[@]}"; do
        local out_dir="${out_root}/seed_${seed}"
        if [ -f "${out_dir}/test_metrics.json" ]; then
            echo "  [$arm_name] seed=$seed already complete. Skipping. ✓"
            continue
        fi
        if [ -f "${out_dir}/resume_checkpoint.pth" ]; then
            echo "  [$arm_name] seed=$seed has a resume checkpoint -- continuing from where it left off..."
        else
            echo "  [$arm_name] Training seed=$seed, epochs=$epochs (fresh start)..."
        fi

        # `if ! cmd; then ...` is deliberate: a command's failure inside an `if`
        # condition does NOT trigger `set -e`, even though `set -e` is on for the
        # rest of the script -- this is what lets ONE seed fail without taking down
        # the other 39. `tee -a` (append, not overwrite) so a later retry of this
        # same seed adds to its log instead of erasing what happened last time.
        if ! python3 src/train_pca.py \
            --seed "$seed" \
            --epochs "$epochs" \
            $extra_flag \
            --data-dir "$DATA_DIR" \
            --output-dir "$out_dir" \
            2>&1 | tee -a "$LOG_DIR/${arm_name}_seed${seed}.log"
        then
            echo "  ✗ [$arm_name] seed=$seed FAILED (nonzero exit) -- its resume_checkpoint.pth" >&2
            echo "    (if any) is preserved; re-running this script later will continue it," >&2
            echo "    not restart from epoch 1. Continuing to the next seed now." >&2
            FAILED_RUNS+=("${arm_name}/seed_${seed} (nonzero exit)")
            continue
        fi

        if [ -f "${out_dir}/test_metrics.json" ]; then
            echo "  ✓ [$arm_name] seed=$seed complete"
        else
            echo "  ✗ [$arm_name] seed=$seed exited 0 but produced no test_metrics.json -- treating as failed." >&2
            FAILED_RUNS+=("${arm_name}/seed_${seed} (exited 0, no test_metrics.json)")
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: SMOKE TEST (seed=42 only, 5 epochs, both arms)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "PHASE 1: SMOKE TEST (seed=42, 5 epochs, both arms)"
echo "================================================================"

rm -rf results/smoke_curriculum results/smoke_baseline

echo ""
echo "[2/5] Smoke test: curriculum arm..."
python3 src/train_pca.py --seed 42 --epochs 5 \
    --data-dir "$DATA_DIR" --output-dir results/smoke_curriculum \
    2>&1 | tee "$LOG_DIR/smoke_curriculum.log"
if [ ! -f results/smoke_curriculum/test_metrics.json ]; then
    echo "  ✗ ERROR: smoke curriculum run produced no test_metrics.json" >&2
    exit 1
fi
echo "  ✓ Smoke curriculum OK"

echo ""
echo "[3/5] Smoke test: baseline arm (--no-curriculum)..."
python3 src/train_pca.py --seed 42 --epochs 5 --no-curriculum \
    --data-dir "$DATA_DIR" --output-dir results/smoke_baseline \
    2>&1 | tee "$LOG_DIR/smoke_baseline.log"
if [ ! -f results/smoke_baseline/test_metrics.json ]; then
    echo "  ✗ ERROR: smoke baseline run produced no test_metrics.json" >&2
    exit 1
fi
echo "  ✓ Smoke baseline OK"

echo ""
echo "  Smoke test metrics (sanity numbers only -- 5 epochs, not meaningful results):"
python3 -c "
import json
for name in ['curriculum', 'baseline']:
    m = json.load(open(f'results/smoke_{name}/test_metrics.json'))
    print(f'    {name:12s} macro_f1={m[\"macro_f1\"]:.4f}  v_recall={m[\"v_recall\"]:.4f}  '
          f'params={m[\"model_params\"]:,}  epochs_run={m[\"epochs_run\"]}')
"
rm -rf results/smoke_curriculum results/smoke_baseline

echo ""
echo "================================================================"
echo "PHASE 1 COMPLETE: smoke test passed for both arms."
echo "Proceeding to the full 20-seed x 2-arm run in 10s -- Ctrl+C now to abort."
echo "================================================================"
sleep 10

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: FULL TRAINING (20 seeds x 2 arms, up to 100 epochs each)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "PHASE 2: FULL TRAINING (20 seeds x 2 arms, up to $EPOCHS_FULL epochs, patience=15)"
echo "Started at: $(date)"
echo "================================================================"

echo ""
echo "[4/5] Full run: curriculum arm (Progressive Curriculum Anchoring)..."
run_arm "curriculum" "" "$EPOCHS_FULL"

echo ""
echo "[4/5] Full run: baseline arm (--no-curriculum -- same architecture/optimizer/"
echo "  schedule/epochs as the curriculum arm, differing ONLY in sampling strategy --"
echo "  AUDIT_FINDINGS.md H6/H20)..."
run_arm "baseline" "--no-curriculum" "$EPOCHS_FULL"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[5/5] Aggregating (seed-paired by identity, not list position -- src/aggregate_seed_results.py,"
echo "  written to avoid AUDIT_FINDINGS.md H8's pairing bugs)..."
python3 src/aggregate_seed_results.py \
    --baseline-dir results/wavkan_v2_baseline \
    --curriculum-dir results/wavkan_v2_curriculum \
    --output results/wavkan_v2_20seed_comparison.json \
    2>&1 | tee "$LOG_DIR/aggregate.log"

# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "WavKAN-CL: PIPELINE COMPLETE -- all 40 runs succeeded."
else
    echo "WavKAN-CL: PIPELINE FINISHED WITH ${#FAILED_RUNS[@]} FAILED RUN(S):"
    for r in "${FAILED_RUNS[@]}"; do echo "    - $r"; done
    echo "  Their resume_checkpoint.pth (if any) is preserved. Re-running this script"
    echo "  will pick each of them back up from where it left off, not from scratch --"
    echo "  it will also re-run Phase 3 aggregation with whatever seeds ARE complete"
    echo "  in the meantime, which is what you're about to see below (partial, not final)."
fi
echo "Finished at: $(date)"
echo "================================================================"
echo ""
echo "Results:"
echo "  Curriculum checkpoints: results/wavkan_v2_curriculum/seed_*/best_model.pth"
echo "  Baseline checkpoints:   results/wavkan_v2_baseline/seed_*/best_model.pth"
echo "  Aggregated comparison:  results/wavkan_v2_20seed_comparison.json"
echo "  Per-seed logs:          $LOG_DIR/"
echo ""
echo "NOT covered by this script (see PHASE5_SCOPE_PLAN.md sections 4-5):"
echo "  - Fig. 3 (seed stability, needs the other 4 baseline models trained under a"
echo "    matched protocol -- not yet fairness-audited)"
echo "  - Fig. 4 (convergence) and Fig. 6 (RR-ablation) -- plotting scripts to be"
echo "    written once real training_history.json files exist to test against"
echo "  - PTB-XL/INCART/SVDB cross-dataset regeneration"
echo ""
echo "Next: bring results/wavkan_v2_20seed_comparison.json (and the checkpoints, if"
echo "you want Fig. 4/6 regenerated) back for Phase 5 verification."
