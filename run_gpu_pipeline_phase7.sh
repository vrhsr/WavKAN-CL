#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# WavKAN-CL: Phase 7 -- strengthen the paper further (project owner: "unlimited
# GPU time, all I need is to publish this for sure")
# ═══════════════════════════════════════════════════════════════════════════════
#
# Everything the main 20-seed comparison, Fig.3 baselines, the internal
# ablation, the RR-position ablation, and Phase 6's cross-dataset re-run
# (INCART/SVDB/PTB-XL on the final configuration) ALREADY established is not
# repeated here. What this script adds, all real, all newly designed this
# session:
#
#   [1] Multi-seed Wavelet-ECG Alignment Score (WAS) -- the manuscript
#       currently reports WAS from a SINGLE checkpoint (seed 42) per
#       configuration. This computes it across all 20 real seeds for BOTH
#       configurations and reports mean+-std, matching this paper's own
#       statistical standard everywhere else. Cheap: WAS only inspects
#       trained wavelet parameters directly (mu, gamma), no data/forward pass
#       needed at all -- this is the fastest job in this script by far.
#
#   [2] SVDB/INCART confusion-matrix diagnostic (src/svdb_confusion_diagnostic.py,
#       new). AUDIT_FINDINGS.md C16 established THAT the wavelet-KAN family
#       (WavKAN-v2 final + B-Spline KAN) is significantly weaker than 3 of 4
#       baselines on SVDB Macro-F1 -- this computes real per-class confusion
#       matrices (summed over 20 seeds) for all 6 models on SVDB and INCART,
#       to show WHICH class pairs the wavelet-KAN family specifically
#       confuses that the baselines don't. Forward-pass only, no new
#       training. Directly targets the manuscript's own Future Work item (i).
#
#   [3] Augmentation study, re-run on the FINAL (use_rr_attn=False) headline
#       configuration. AUDIT_FINDINGS.md H29 (found this session): the
#       existing augmentation study (Table tab:augmentation) was trained on
#       the superseded INITIAL (RR-self-attention) configuration the whole
#       time -- src/noise_augmentation.py never had a --no-rr-attn flag until
#       today. This is real, new training (5 strategies x 20 seeds, up to 60
#       epochs each, real GPU cost, NOT a forward pass) -- the one expensive
#       job in this script. Could plausibly change the reported numbers, and
#       might also finally answer Future Work item (iii) (why augmentation
#       costs Macro-F1 under this recipe) if the final config behaves
#       differently than the initial one did.
#
# Same resilience pattern as run_gpu_pipeline_phase6.sh: skip-if-complete per
# job, log-and-continue on a single job's failure, summary at the end.
# Nothing in this script looks at test labels to decide whether to skip or
# retry a job -- only each job's own prior output file existing or not.
#
# Prerequisite: run_gpu_pipeline_phase6.sh must have completed (needs
# data/processed_rr_history for [3], and the ablation_no_rr_attn /
# wavkan_v2_curriculum / baseline_* checkpoint directories for [1]/[2], all of
# which should already exist from earlier phases).
#
# Usage:
#   chmod +x run_gpu_pipeline_phase7.sh
#   nohup bash run_gpu_pipeline_phase7.sh 2>&1 | tee phase7_log.txt &
# ═══════════════════════════════════════════════════════════════════════════════

set -e
set -o pipefail

echo "[Preflight] Checking required Python packages..."
MISSING_PKGS=""
for pkg in torch numpy sklearn; do
    python3 -c "import ${pkg}" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} ${pkg}"
done
if [ -n "$MISSING_PKGS" ]; then
    echo "  Missing packages:${MISSING_PKGS}" >&2
    exit 1
fi
echo "  OK"

export PYTHONPATH="$PWD"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/phase7_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

FAILED_RUNS=()

echo "================================================================"
echo "WavKAN-CL Phase 7 | Started at: $(date)"
echo "================================================================"
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(none)')
"

# Same real 20-seed list used everywhere else in this project.
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
# [1] Multi-seed WAS, both configurations
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[1/3] Multi-seed Wavelet-ECG Alignment Score (WAS), both configurations..."
echo "================================================================"

for seed in "${SEEDS_20[@]}"; do
    ckpt="results/ablation_no_rr_attn/seed_${seed}/best_model.pth"
    out="results/was_multiseed_final/seed_${seed}/was_scores.json"
    if [ -f "$ckpt" ]; then
        run_job "was_final/seed_${seed}" "[ -f ${out} ]" \
            python3 src/wavelet_alignment_score.py \
                --checkpoint "$ckpt" \
                --output-dir "results/was_multiseed_final/seed_${seed}" \
                --no-rr-attn
    else
        echo "  [was_final/seed_${seed}] Skipping -- $ckpt not found."
        FAILED_RUNS+=("was_final/seed_${seed} (missing checkpoint)")
    fi
done

for seed in "${SEEDS_20[@]}"; do
    ckpt="results/wavkan_v2_curriculum/seed_${seed}/best_model.pth"
    out="results/was_multiseed_initial/seed_${seed}/was_scores.json"
    if [ -f "$ckpt" ]; then
        run_job "was_initial/seed_${seed}" "[ -f ${out} ]" \
            python3 src/wavelet_alignment_score.py \
                --checkpoint "$ckpt" \
                --output-dir "results/was_multiseed_initial/seed_${seed}"
    else
        echo "  [was_initial/seed_${seed}] Skipping -- $ckpt not found."
        FAILED_RUNS+=("was_initial/seed_${seed} (missing checkpoint)")
    fi
done

echo "  Aggregating WAS across seeds (both configurations)..."
python3 -c "
import json, glob

def aggregate(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'  {label}: no per-seed WAS results found.')
        return None
    scores = [json.load(open(f)) for f in files]
    import statistics as st
    def agg(key_path):
        vals = scores
        for k in key_path:
            vals = [v[k] for v in vals]
        return {'mean': st.mean(vals), 'std': st.stdev(vals) if len(vals) > 1 else 0.0, 'n': len(vals)}
    summary = {
        'n_seeds': len(scores),
        'macro_was': agg(['macro_was']),
        'qrs_was': agg(['QRS', 'was']),
        'p_was': agg(['P', 'was']),
        't_was': agg(['T', 'was']),
    }
    print(f\"  {label}: Macro WAS = {summary['macro_was']['mean']:.4f} +- {summary['macro_was']['std']:.4f} (n={summary['macro_was']['n']})\")
    return summary

final_summary = aggregate('results/was_multiseed_final/seed_*/was_scores.json', 'Final (use_rr_attn=False)')
initial_summary = aggregate('results/was_multiseed_initial/seed_*/was_scores.json', 'Initial (use_rr_attn=True)')

with open('results/was_multiseed_summary.json', 'w') as f:
    json.dump({'final_configuration': final_summary, 'initial_configuration': initial_summary}, f, indent=2)
print('  Saved -> results/was_multiseed_summary.json')
"

# ═══════════════════════════════════════════════════════════════════════════════
# [2] SVDB/INCART confusion-matrix diagnostic
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[2/3] SVDB/INCART confusion-matrix diagnostic..."
echo "================================================================"
if [ -f data/svdb_processed/X_test.npy ] || [ -f data/incart_processed/X_test.npy ]; then
    run_job "svdb_incart_diagnostic" '[ -f results/svdb_diagnostic/svdb_incart_confusion_diagnostic.json ]' \
        python3 src/svdb_confusion_diagnostic.py \
            --results-base results \
            --output-dir results/svdb_diagnostic \
            --datasets SVDB INCART
else
    echo "  Skipping -- neither data/svdb_processed nor data/incart_processed present."
    FAILED_RUNS+=("svdb_incart_diagnostic (no data)")
fi

# ═══════════════════════════════════════════════════════════════════════════════
# [3] Augmentation study, re-run on the final (use_rr_attn=False) configuration
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[3/3] Augmentation study, final configuration (real training, ~5x20 runs)..."
echo "================================================================"
if [ -d data/processed_rr_history ]; then
    run_job "augmentation_final" '[ -f results/noise_augmentation_final/augmentation_comparison.json ]' \
        python3 src/noise_augmentation.py \
            --strategies none gaussian baseline_wander combined smote \
            --seeds "${SEEDS_20[@]}" \
            --output-dir results/noise_augmentation_final \
            --no-rr-attn
else
    echo "  Skipping -- data/processed_rr_history not present."
    FAILED_RUNS+=("augmentation_final (no data)")
fi

echo ""
echo "================================================================"
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "Phase 7 COMPLETE -- all jobs succeeded."
else
    echo "Phase 7 finished with ${#FAILED_RUNS[@]} failed/skipped/blocked job(s):"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
    echo "Re-running this script will retry only what's missing (skip-if-complete)."
fi
echo "Finished at: $(date)"
echo "================================================================"
echo ""
echo "Copy back once complete:"
echo "  - results/was_multiseed_summary.json"
echo "  - results/svdb_diagnostic/svdb_incart_confusion_diagnostic.json"
echo "  - results/noise_augmentation_final/ (augmentation_comparison.json, augmentation_figure.pdf, augmentation_table.tex)"
