#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# WavKAN-CL: Phase 9 -- does removing default augmentation improve the paper's
# own final-configuration headline Macro-F1? (AUDIT_FINDINGS.md H36)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Background: the paper's adopted final-configuration headline number is
# Table tab:family_ablation's "No RR self-attention" row (Macro-F1=0.357,
# results/ablation_no_rr_attn/) -- trained with train_pca.py's DEFAULT
# use_augment=True (Combined-family augmentation: Gaussian noise + baseline
# wander + amplitude scaling). The completed Phase 7 augmentation study
# (results/noise_augmentation_final/, its own isolated 60-epoch,
# no-curriculum recipe) found this same augmentation family costs real
# Macro-F1 under the final configuration with no significant S-Recall
# benefit anymore -- but that study never used the actual headline training
# recipe (100 epochs, PCA curriculum, early stopping). This script tests the
# real headline recipe directly.
#
# Two new arms, 20 real seeds each, both on the final (--no-rr-attn)
# configuration:
#   [1] Curriculum ON,  augmentation OFF -- results/ablation_no_rr_attn_no_augment/
#       Directly comparable to the EXISTING results/ablation_no_rr_attn/ (same
#       config, only --no-augment added) -- this is the primary H36 test.
#   [2] Curriculum OFF, augmentation OFF -- results/ablation_no_rr_attn_no_curriculum_no_augment/
#       For completeness: mirrors tab:curriculum_stats's curriculum-vs-baseline
#       comparison, but on the final config instead of the initial one (no
#       such comparison currently exists for the final config at all).
#
# 40 real, full training runs (100 epochs, early stopping patience=15) -- the
# actual expensive headline recipe, not the augmentation study's cheap
# isolated one. Same resilience pattern as Phase 6/7/8: skip-if-complete,
# log-and-continue on a single seed's failure, summary at the end. Nothing
# here looks at test labels to decide whether to skip or which seed to keep.
#
# This is real, new training that could change a previously-reported
# headline number if the hypothesis holds -- flagged in AUDIT_FINDINGS.md H36
# as needing project-owner sign-off before running, which has been given.
# Whatever the result (better, worse, or no different), it gets reported
# honestly; this script does not pick a favorable seed or arm.
#
# Usage:
#   chmod +x run_gpu_pipeline_phase9.sh
#   nohup bash run_gpu_pipeline_phase9.sh 2>&1 | tee phase9_log.txt &
# ═══════════════════════════════════════════════════════════════════════════════

set -e
set -o pipefail

echo "[Preflight] Checking required Python packages..."
MISSING_PKGS=""
for pkg in torch numpy sklearn scipy; do
    python3 -c "import ${pkg}" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} ${pkg}"
done
if [ -n "$MISSING_PKGS" ]; then
    echo "  Missing packages:${MISSING_PKGS}" >&2
    exit 1
fi
if [ ! -d "data/processed_rr_history" ]; then
    echo "  data/processed_rr_history not found -- this pipeline needs the same" >&2
    echo "  processed MIT-BIH data every other train_pca.py run in this repo uses." >&2
    exit 1
fi
echo "  OK"

export PYTHONPATH="$PWD"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/phase9_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

FAILED_RUNS=()

echo "================================================================"
echo "WavKAN-CL Phase 9 | Started at: $(date)"
echo "================================================================"
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(none)')
"

SEEDS_20=(42 101 777 2026 9999 1234 2024 31415 27182 7 11 13 888 555 333 99 1001 5050 8080 1998)

# ─── Generic runner: skip-if-complete, continue-on-failure ──────────────────────
run_job () {
    local label="$1"; local done_check="$2"; shift 2
    if eval "$done_check"; then
        echo "  [$label] already complete. Skipping. OK"
        return 0
    fi
    echo "  [$label] running..."
    if ! "$@" 2>&1 | tee -a "$LOG_DIR/$(echo "$label" | tr ' /' '__').log"; then
        echo "  [$label] FAILED -- continuing to next job." >&2
        FAILED_RUNS+=("$label")
        return 1
    fi
    if ! eval "$done_check"; then
        echo "  [$label] exited 0 but expected output still missing -- treating as failed." >&2
        FAILED_RUNS+=("$label (no output)")
        return 1
    fi
    echo "  [$label] OK"
}

# ═══════════════════════════════════════════════════════════════════════════════
# [1] Final config, curriculum ON, augmentation OFF -- the primary H36 test
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[1/2] Final config, curriculum ON, augmentation OFF (20 seeds)..."
echo "  Directly comparable to existing results/ablation_no_rr_attn/"
echo "  (same config, only --no-augment added)."
echo "================================================================"

for seed in "${SEEDS_20[@]}"; do
    out_dir="results/ablation_no_rr_attn_no_augment/seed_${seed}"
    run_job "h36_curriculum_no_augment/seed_${seed}" "[ -f ${out_dir}/test_metrics.json ]" \
        python3 src/train_pca.py --seed "$seed" --epochs 100 \
            --no-rr-attn --no-augment \
            --data-dir data/processed_rr_history --output-dir "$out_dir" || true
done

# ═══════════════════════════════════════════════════════════════════════════════
# [2] Final config, curriculum OFF, augmentation OFF -- for completeness
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[2/2] Final config, curriculum OFF, augmentation OFF (20 seeds)..."
echo "  No existing result for this exact combination -- fills a real gap."
echo "================================================================"

for seed in "${SEEDS_20[@]}"; do
    out_dir="results/ablation_no_rr_attn_no_curriculum_no_augment/seed_${seed}"
    run_job "h36_no_curriculum_no_augment/seed_${seed}" "[ -f ${out_dir}/test_metrics.json ]" \
        python3 src/train_pca.py --seed "$seed" --epochs 100 \
            --no-rr-attn --no-augment --no-curriculum \
            --data-dir data/processed_rr_history --output-dir "$out_dir" || true
done

# ═══════════════════════════════════════════════════════════════════════════════
# [3] Analysis -- real, honest, paired stats against the existing headline arm
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[3/3] Aggregating and testing significance (forward-pass-free, cheap)..."
echo "================================================================"
python3 src/analyze_no_augment_ablation.py 2>&1 | tee -a "$LOG_DIR/analysis.log" || \
    FAILED_RUNS+=("analysis")

echo ""
echo "================================================================"
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "Phase 9 COMPLETE -- all jobs succeeded."
else
    echo "Phase 9 finished with ${#FAILED_RUNS[@]} failed/skipped job(s):"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
    echo "Re-running this script will retry only what's missing (skip-if-complete)."
fi
echo "Finished at: $(date)"
echo "================================================================"
echo ""
echo "Copy back once complete:"
echo "  - results/h36_no_augment_ablation_report.json   (the real answer to H36)"
echo "  - results/ablation_no_rr_attn_no_augment/seed_*/test_metrics.json"
echo "  - results/ablation_no_rr_attn_no_curriculum_no_augment/seed_*/test_metrics.json"
