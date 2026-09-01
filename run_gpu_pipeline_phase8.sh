#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# WavKAN-CL: Phase 8 -- honest few-shot cross-dataset adaptation
# ═══════════════════════════════════════════════════════════════════════════════
#
# Directly answers the AIIM rejection's PTB-XL-generalization comment
# (AIIM_REJECTION_ANALYSIS.md) with a real result instead of only a disclosed
# limitation: given a SMALL amount of real target-domain data, how much of
# the zero-shot gap closes? Reports the full k-shot curve (including k=0 and
# any k that doesn't help), never a cherry-picked favorable k. Uses ONLY the
# final (use_rr_attn=False) headline configuration.
#
# NOT src/eval_fewshot_ptbxl.py (AUDIT_FINDINGS.md C8: silent fabrication
# fallback, documented intent to swap out an honest result). This is a fresh
# design (src/fewshot_domain_adaptation.py, new) with a record/patient-
# disjoint adaptation-vs-eval split fixed BEFORE any model touches the data --
# see that script's own docstring for the full methodology and why each
# design choice was made.
#
#   [1] Reprocess INCART, SVDB, PTB-XL. Needed because the existing processed
#       data on this box predates this session's addition of per-beat
#       record_ids_test.npy / patient_ids_test.npy (process_incart.py /
#       process_svdb.py / process_ptbxl.py, all fixed this session) --
#       without those, there is no way to build a leakage-safe adaptation/
#       eval split. Assumes raw PhysioNet downloads already exist from
#       earlier phases (data/raw/incart, data/raw/svdb, data/raw/ptbxl); if
#       not, add --download to the relevant process_*.py invocation below.
#
#   [2] Run the few-shot study for all 3 target datasets, 20 seeds each,
#       k in {0, 10, 50, 100, 500} beats/class. This is real, new compute
#       (3 datasets x 20 seeds x 5 k-values = up to 300 fine-tune+eval runs,
#       each small -- at most 500 beats, 10 epochs) -- moderate cost, not a
#       forward-pass-only job like Phase 7's diagnostic step.
#
# Same resilience pattern as Phase 6/7: skip-if-complete, log-and-continue,
# summary at the end. Nothing here looks at target-domain labels to decide
# whether to re-run or which k to keep -- only whether the output file for a
# given (dataset, all-seeds) combination already exists.
#
# Usage:
#   chmod +x run_gpu_pipeline_phase8.sh
#   nohup bash run_gpu_pipeline_phase8.sh 2>&1 | tee phase8_log.txt &
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
LOG_DIR="logs/phase8_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

FAILED_RUNS=()

echo "================================================================"
echo "WavKAN-CL Phase 8 | Started at: $(date)"
echo "================================================================"
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(none)')
"

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
# [1] Reprocess INCART, SVDB, PTB-XL with per-beat record/patient IDs
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[1/2] Reprocessing INCART/SVDB/PTB-XL to add per-beat record/patient IDs..."
echo "================================================================"

run_job "reprocess_incart" '[ -f data/incart_processed/record_ids_test.npy ]' \
    python3 src/process_incart.py --out-dir data/incart_processed

run_job "reprocess_svdb" '[ -f data/svdb_processed/record_ids_test.npy ]' \
    python3 src/process_svdb.py --out-dir data/svdb_processed

run_job "reprocess_ptbxl" '[ -f data/processed_ptbxl/patient_ids_test.npy ]' \
    python3 src/process_ptbxl.py --data-dir data/ptbxl --out-dir data/processed_ptbxl

# ═══════════════════════════════════════════════════════════════════════════════
# [2] Few-shot adaptation study, all 3 target datasets, final configuration
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "[2/2] Few-shot domain adaptation (INCART, SVDB, PTB-XL), 20 seeds each..."
echo "================================================================"

declare -A DATA_CHECK=(
    ["INCART"]="data/incart_processed/record_ids_test.npy"
    ["SVDB"]="data/svdb_processed/record_ids_test.npy"
    ["PTB-XL"]="data/processed_ptbxl/patient_ids_test.npy"
)

for ds in INCART SVDB PTB-XL; do
    check_file="${DATA_CHECK[$ds]}"
    out_file="results/fewshot_adaptation/$(echo "$ds" | tr 'A-Z' 'a-z' | tr -d '-')/fewshot_$(echo "$ds" | tr 'A-Z' 'a-z' | tr -d '-')_report.json"
    if [ -f "$check_file" ]; then
        run_job "fewshot_${ds}" "[ -f ${out_file} ]" \
            python3 src/fewshot_domain_adaptation.py \
                --dataset "$ds" \
                --checkpoints-dir results/ablation_no_rr_attn \
                --output-dir "results/fewshot_adaptation/$(echo "$ds" | tr 'A-Z' 'a-z' | tr -d '-')"
    else
        echo "  [fewshot_${ds}] Skipping -- $check_file not found (step [1] must complete first)."
        FAILED_RUNS+=("fewshot_${ds} (no data)")
    fi
done

echo ""
echo "================================================================"
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "Phase 8 COMPLETE -- all jobs succeeded."
else
    echo "Phase 8 finished with ${#FAILED_RUNS[@]} failed/skipped/blocked job(s):"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
    echo "Re-running this script will retry only what's missing (skip-if-complete)."
fi
echo "Finished at: $(date)"
echo "================================================================"
echo ""
echo "Copy back once complete:"
echo "  - results/fewshot_adaptation/incart/fewshot_incart_report.json"
echo "  - results/fewshot_adaptation/svdb/fewshot_svdb_report.json"
echo "  - results/fewshot_adaptation/ptbxl/fewshot_ptbxl_report.json"
