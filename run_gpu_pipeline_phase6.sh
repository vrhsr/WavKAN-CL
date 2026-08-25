#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# WavKAN-CL: Phase 6 -- close the remaining honest gaps in ieee_manuscript_v2.tex
# ═══════════════════════════════════════════════════════════════════════════════
#
# Everything the new manuscript draft (Submission_JBHI/ieee_manuscript_v2.tex)
# lists in its own Limitations/Future Work sections and can actually be closed
# with compute, not a new architecture. All of the big training runs (the main
# 20-seed curriculum-vs-baseline comparison, the 4-baseline Fig.3 comparison,
# the internal wavelet/PCWI/PWAM/RR-attn ablation, the RR-position ablation)
# are ALREADY DONE -- this script does not repeat any of them. What's left:
#
#   [1] Regenerate INCART + SVDB processed data (download via wfdb if absent)
#   [2] Multi-dataset zero-shot table: WavKAN-v2 + 4 baselines, all 20 seeds
#       each, on MIT-BIH (self) / INCART (zero-shot) / SVDB (zero-shot), via
#       src/eval_multidataset.py -- fixed 2026-08-25 (AUDIT_FINDINGS.md C18):
#       it used to unconditionally bold/tag the "wavkan_v2" row and assert an
#       uncomputed "p<0.05" in its caption regardless of the real numbers.
#       Report whatever this produces, however it lands -- do not re-add any
#       identity-based highlighting or an unearned significance claim on top
#       of its now-honest output.
#   [3] Multi-seed PTB-XL zero-shot: the manuscript currently reports only
#       seed 42 (Table tab:ptbxl_zero_shot) -- this re-runs all 20 curriculum
#       seeds and aggregates mean+-std, matching this paper's own statistical
#       standard everywhere else. Uses src/eval_ptbxl.py only (the honest,
#       already-fixed zero-shot script) -- deliberately NEVER
#       src/eval_fewshot_ptbxl.py, which AUDIT_FINDINGS.md C8 documents has a
#       silent numeric-fabrication fallback and a stated intent to swap out
#       an honest result that "hurt the narrative." Do not substitute it in.
#   [4] Re-run the 4-strategy augmentation study (none/gaussian/baseline_
#       wander/combined/smote) with real per-seed logging across the full
#       20-seed list -- closes manuscript Limitation (6): the current
#       augmentation numbers are read off a figure because the underlying
#       per-seed JSON did not survive in this checkout, not because the
#       script itself doesn't save one (it does -- src/noise_augmentation.py
#       already writes augmentation_comparison.json with both a summary and
#       the raw per-seed values; it just needs to actually be run again).
#
# Same resilience pattern as run_gpu_pipeline_phase5b.sh: skip-if-complete per
# job, log-and-continue on a single job's failure, summary at the end. Nothing
# in this script ever looks at DS2/test labels to decide whether to skip or
# retry a job -- only each job's own prior output file existing or not.
#
# NOT in this script (deliberately -- these are not GPU/data problems):
#   - Real on-device (ARM/microcontroller) latency benchmarking -- needs that
#     actual hardware, not a GPU box. Manuscript Limitation (5), still open.
#   - Adopting the plain-MLP RR encoding as the reported default instead of
#     the self-attention variant -- this needs NO new training (Table
#     tab:family_ablation's "no_rr_attn" row already IS this, already
#     20-seed-evaluated) -- it's a manuscript rewrite decision, not a job
#     for this script. See ieee_manuscript_v2.tex \S\ref{sec:discussion_rr}.
#
# Usage:
#   chmod +x run_gpu_pipeline_phase6.sh
#   nohup bash run_gpu_pipeline_phase6.sh 2>&1 | tee phase6_log.txt &
#   bash pipeline_status.sh     # check progress from another pane (Phase 6 section)
# ═══════════════════════════════════════════════════════════════════════════════

set -e
set -o pipefail

echo "[Preflight] Checking required Python packages..."
MISSING_PKGS=""
for pkg in wfdb torch numpy sklearn scipy; do
    python3 -c "import ${pkg}" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} ${pkg}"
done
if [ -n "$MISSING_PKGS" ]; then
    echo "  Missing packages:${MISSING_PKGS}" >&2
    exit 1
fi
echo "  OK"

export PYTHONPATH="$PWD"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/phase6_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

FAILED_RUNS=()

echo "================================================================"
echo "WavKAN-CL Phase 6 | Started at: $(date)"
echo "================================================================"
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(none)')
"

# Same real 20-seed list used everywhere else in this project (main comparison,
# Fig.3 baselines, Table 4 ablations, RR-ablation) -- kept identical so every
# result in the paper is comparable seed-for-seed.
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
# [1] INCART + SVDB data regeneration
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[1/4] INCART + SVDB data regeneration..."
run_job "incart_data" '[ -f data/incart_processed/X_test.npy ]' \
    python3 src/process_incart.py --download --out-dir data/incart_processed || true
run_job "svdb_data" '[ -f data/svdb_processed/X_test.npy ]' \
    python3 src/process_svdb.py --download --out-dir data/svdb_processed || true

# ═══════════════════════════════════════════════════════════════════════════════
# [2] Multi-dataset zero-shot table (WavKAN-v2 + 4 baselines, 20 seeds, 3 datasets)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[2/4] Multi-dataset zero-shot evaluation..."
if [ -f "data/incart_processed/X_test.npy" ] || [ -f "data/svdb_processed/X_test.npy" ]; then
    run_job "multidataset_eval" '[ -f results/multidataset_table/multidataset_raw.json ]' \
        python3 src/eval_multidataset.py \
            --results-base results \
            --seeds "${SEEDS_20[@]}" \
            --output-dir results/multidataset_table || true
else
    echo "  Skipping -- neither INCART nor SVDB data is present (step [1] must complete first)."
    FAILED_RUNS+=("multidataset_eval (blocked on step 1)")
fi

# ═══════════════════════════════════════════════════════════════════════════════
# [3] Multi-seed PTB-XL zero-shot
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[3/4] Multi-seed PTB-XL zero-shot (20 seeds, honest eval_ptbxl.py only)..."
if [ -f "data/processed_ptbxl/X_test.npy" ]; then
    mkdir -p results/ptbxl_zeroshot_multiseed
    for seed in "${SEEDS_20[@]}"; do
        ckpt="results/wavkan_v2_curriculum/seed_${seed}/best_model.pth"
        out="results/ptbxl_zeroshot_multiseed/seed_${seed}.json"
        if [ ! -f "$ckpt" ]; then
            echo "  [ptbxl/seed_${seed}] no checkpoint at $ckpt -- skipping."
            continue
        fi
        run_job "ptbxl/seed_${seed}" "[ -f ${out} ]" \
            python3 src/eval_ptbxl.py --model-path "$ckpt" \
                --data-dir data/processed_ptbxl --out "$out" || true
    done

    echo "  Aggregating across seeds..."
    python3 -c "
import json, glob
import numpy as np

files = sorted(glob.glob('results/ptbxl_zeroshot_multiseed/seed_*.json'))
if not files:
    print('  No per-seed PTB-XL results found -- nothing to aggregate.')
else:
    per_seed = [json.load(open(f)) for f in files]

    def agg_top(key):
        vals = [d[key] for d in per_seed if d.get(key) is not None]
        if not vals:
            return None
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 'n': len(vals)}

    def agg_class_recall(cls):
        vals = [d.get('per_class_recall', {}).get(cls) for d in per_seed]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 'n': len(vals)}

    per_class = {}
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        result = agg_class_recall(cls)
        if result is not None:
            per_class[cls] = result

    summary = {
        'n_seeds': len(per_seed),
        'macro_f1': agg_top('macro_f1'),
        'per_class_recall': per_class,
        'note': 'A class is omitted from per_class_recall above if every seed reported it as null (e.g. zero support in this PTB-XL extraction).',
    }
    with open('results/ptbxl_zero_shot_metrics_multiseed.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Aggregated {len(per_seed)} seeds -> results/ptbxl_zero_shot_metrics_multiseed.json')
    print(f\"  Macro-F1: {summary['macro_f1']['mean']:.4f} +- {summary['macro_f1']['std']:.4f} (n={summary['macro_f1']['n']})\")
"
else
    echo "  Skipping -- data/processed_ptbxl not regenerated yet (see run_gpu_pipeline_phase5b.sh step [4] instructions)."
    FAILED_RUNS+=("ptbxl_multiseed (blocked on PTB-XL data)")
fi

# ═══════════════════════════════════════════════════════════════════════════════
# [4] Augmentation study re-run with real per-seed logging (20 seeds)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[4/4] Augmentation strategy comparison, real per-seed logging, 20 seeds..."
run_job "augmentation_study" '[ -f results/noise_augmentation/augmentation_comparison.json ]' \
    python3 src/noise_augmentation.py \
        --strategies none gaussian baseline_wander combined smote \
        --seeds "${SEEDS_20[@]}" \
        --epochs 60 \
        --data-dir data/processed_rr_history \
        --output-dir results/noise_augmentation || true

echo ""
echo "================================================================"
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "Phase 6 COMPLETE -- all jobs succeeded."
else
    echo "Phase 6 finished with ${#FAILED_RUNS[@]} failed/skipped/blocked job(s):"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
    echo "Re-running this script will retry only what's missing (skip-if-complete)."
fi
echo "Finished at: $(date)"
echo "================================================================"
echo ""
echo "Next steps once this completes:"
echo "  - Update Table tab:multidataset / Table tab:ptbxl_zero_shot in ieee_manuscript_v2.tex"
echo "    with results/multidataset_table/multidataset_raw.json and"
echo "    results/ptbxl_zero_shot_metrics_multiseed.json."
echo "  - If real significance vs. a baseline is wanted for the multi-dataset table, run"
echo "    src/metrics_full.statistical_comparison on the saved per-seed values yourself and"
echo "    report only what it actually computes -- eval_multidataset.py no longer asserts one."
echo "  - Re-derive the augmentation-strategy prose in the manuscript's Discussion from"
echo "    results/noise_augmentation/augmentation_comparison.json's real 'summary' block,"
echo "    replacing the current figure-read/approximate framing."
