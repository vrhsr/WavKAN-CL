#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# WavKAN-CL: Phase 5b -- Fig. 3 baselines, Table 4 ablation matrix, Fig. 6, cross-dataset
# ═══════════════════════════════════════════════════════════════════════════════
#
# Everything remaining after the main 20-seed baseline-vs-curriculum comparison
# (which is already done -- results/wavkan_v2_20seed_comparison.json). This script:
#   [1] Fig. 3 baselines: ResNet1D / Transformer / CNN+Focal / B-Spline KAN, 20 seeds each
#       (upgraded from the original manuscript's n=5 -- unlimited compute, no reason to
#       give a top-tier reviewer fewer seeds here than in the main comparison)
#   [2] Table 4 ablation matrix: 3 wavelet types (morlet/dog/b_spline) + 3 structural
#       ablations (no-pcwi/no-pwam/no-rr-attn), 20 seeds each, via train_pca.py's own
#       flags -- NOT the old run_ablation_study.py/run_ablation_v2.py (AUDIT_FINDINGS.md
#       C9's "broken curriculum" pipeline). mexican_hat/"full" is NOT re-run here --
#       the already-completed 20-seed curriculum arm IS that configuration.
#   [3] Fig. 6 RR-ablation: one command, against the real 20 curriculum checkpoints
#   [4] Cross-dataset: PTB-XL (zero-shot, via the just-fixed eval_ptbxl.py) + INCART/SVDB
#
# All comparisons in this script now use Holm-Bonferroni-corrected p-values + Cohen's d
# (src/metrics_full.statistical_comparison) wherever aggregation happens downstream, not
# just raw p<0.05 -- matches the same upgrade applied to the main 20-seed comparison.
#
# Same resilience approach as run_gpu_pipeline.sh: skip-if-complete per job, log-and-
# continue on a single job's failure (not a scary fabrication tool -- see
# AUDIT_FINDINGS.md rule 3 -- this only ever skips based on a job's OWN prior
# completion, never on any test-set metric), summary of failures at the end.
#
# Usage:
#   chmod +x run_gpu_pipeline_phase5b.sh
#   nohup bash run_gpu_pipeline_phase5b.sh 2>&1 | tee phase5b_log.txt &
# ═══════════════════════════════════════════════════════════════════════════════

set -e
set -o pipefail

echo "[Preflight] Checking required Python packages..."
MISSING_PKGS=""
for pkg in pytest wfdb neurokit2 torch numpy sklearn scipy; do
    python3 -c "import ${pkg}" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} ${pkg}"
done
if [ -n "$MISSING_PKGS" ]; then
    echo "  Missing packages:${MISSING_PKGS}" >&2
    exit 1
fi
echo "  OK"

export PYTHONPATH="$PWD"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/phase5b_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

DATA_DIR="data/processed_rr_history"
FAILED_RUNS=()

echo "================================================================"
echo "WavKAN-CL Phase 5b | Started at: $(date)"
echo "================================================================"
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
"

# Upgraded 2026-08-13: originally n=5 for Fig 3/Table 4 (matching the OLD
# manuscript caption's "n=5 seeds"), scaled to the full 20 now that compute isn't
# the constraint (project owner has unlimited GPU access, targeting a top-tier
# venue) -- no reason to give a reviewer an easy "why fewer seeds here than in
# the main comparison" question when there's no cost to matching it. Same list as
# run_gpu_pipeline.sh/the main comparison, for consistency across the whole paper.
SEEDS_20=(42 101 777 2026 9999 1234 2024 31415 27182 7 11 13 888 555 333 99 1001 5050 8080 1998)

# ─── Generic runner: skip-if-complete, continue-on-failure ──────────────────────
run_job () {
    local label="$1"; local out_dir="$2"; shift 2
    if [ -f "${out_dir}/test_metrics.json" ]; then
        echo "  [$label] already complete. Skipping. OK"
        return 0
    fi
    echo "  [$label] running..."
    if ! "$@" 2>&1 | tee -a "$LOG_DIR/$(echo "$label" | tr ' /' '__').log"; then
        echo "  [$label] FAILED -- continuing to next job." >&2
        FAILED_RUNS+=("$label")
        return 1
    fi
    if [ ! -f "${out_dir}/test_metrics.json" ]; then
        echo "  [$label] exited 0 but no test_metrics.json -- treating as failed." >&2
        FAILED_RUNS+=("$label (no output)")
        return 1
    fi
    echo "  [$label] OK"
}

# ═══════════════════════════════════════════════════════════════════════════════
# [1] Fig. 3 baselines
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[1/4] Fig. 3 baselines (4 models x 20 seeds)..."
for model in resnet1d transformer cnn_focal bspline_kan; do
    for seed in "${SEEDS_20[@]}"; do
        out_dir="results/baseline_${model}/seed_${seed}"
        run_job "fig3/${model}/seed_${seed}" "$out_dir" \
            python3 src/baselines_extended.py --model "$model" --seeds "$seed" \
                --data-dir "$DATA_DIR" || true
        # baselines_extended.py's own --seeds loop writes to
        # results/baseline_<model>/seed_<seed>/ when invoked with a single seed here.
    done
done

# ═══════════════════════════════════════════════════════════════════════════════
# [2] Table 4 ablation matrix (via train_pca.py's own flags)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[2/4] Table 4 ablation matrix (6 configs x 20 seeds)..."
# name | extra train_pca.py flags
ABLATIONS=(
    "wavelet_morlet|--wavelet morlet"
    "wavelet_dog|--wavelet dog"
    "wavelet_bspline|--wavelet b_spline"
    "no_pcwi|--no-pcwi"
    "no_pwam|--no-pwam"
    "no_rr_attn|--no-rr-attn"
)
for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r name flags <<< "$entry"
    for seed in "${SEEDS_20[@]}"; do
        out_dir="results/ablation_${name}/seed_${seed}"
        run_job "ablation/${name}/seed_${seed}" "$out_dir" \
            python3 src/train_pca.py --seed "$seed" --epochs 100 $flags \
                --data-dir "$DATA_DIR" --output-dir "$out_dir" || true
    done
done

# ═══════════════════════════════════════════════════════════════════════════════
# [3] Fig. 6 RR-ablation (real checkpoints, all 20 curriculum seeds)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[3/4] Fig. 6 RR-ablation..."
# Skip-if-complete added 2026-08-16: this step previously had no completion check at
# all (unlike [1]/[2]'s per-seed run_job() calls), so every re-invocation of this
# script -- e.g. to pick up [4]'s PTB-XL step after regenerating data -- silently
# redid the full 20-seed x 5-position leave-one-out ablation from scratch, wasting
# real GPU time on already-correct, already-saved results.
if [ -f "results/rr_ablation_real/rr_ablation_report.json" ]; then
    echo "  [fig6_rr_ablation] already complete. Skipping. OK"
else
    python3 src/rr_ablation.py --model-dir results/wavkan_v2_curriculum \
        --seeds "${SEEDS_20[@]}" --data-dir "$DATA_DIR" \
        --output-dir results/rr_ablation_real \
        2>&1 | tee -a "$LOG_DIR/fig6_rr_ablation.log" || FAILED_RUNS+=("fig6_rr_ablation")
fi

# ═══════════════════════════════════════════════════════════════════════════════
# [4] Cross-dataset (PTB-XL zero-shot, INCART/SVDB zero-shot)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[4/4] Cross-dataset evaluation..."
echo "  NOTE: needs PTB-XL/INCART/SVDB raw data downloaded first -- see PHASE5_SCOPE_PLAN.md"
echo "  section 5. Not auto-downloaded here (PTB-XL alone is ~1.7GB) -- run manually first:"
echo "    python3 -c \"import wfdb; wfdb.dl_database('ptb-xl', 'data/ptbxl')\""
echo "    python3 src/process_ptbxl.py --data-dir data/ptbxl --out-dir data/processed_ptbxl --sampling-rate 100"
if [ -f "data/processed_ptbxl/X_test.npy" ]; then
    python3 src/eval_ptbxl.py \
        --model-path results/wavkan_v2_curriculum/seed_42/best_model.pth \
        --data-dir data/processed_ptbxl \
        --out results/ptbxl_zero_shot_metrics_real.json \
        2>&1 | tee -a "$LOG_DIR/ptbxl_zeroshot.log" || FAILED_RUNS+=("ptbxl_zeroshot")
else
    echo "  Skipping PTB-XL eval -- data/processed_ptbxl not present yet."
fi

echo ""
echo "================================================================"
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "Phase 5b COMPLETE -- all jobs succeeded."
else
    echo "Phase 5b finished with ${#FAILED_RUNS[@]} failed/skipped job(s):"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
    echo "Re-running this script will retry only what's missing (skip-if-complete)."
fi
echo "Finished at: $(date)"
echo "================================================================"
echo ""
echo "Next: python3 src/generate_fig3_seed_stability.py   (once [1] has real data)"
echo "      python3 src/generate_fig4_convergence.py      (already works -- done)"
echo "      results/rr_ablation_real/rr_ablation_figure.pdf is Fig. 6, already generated by [3]"
