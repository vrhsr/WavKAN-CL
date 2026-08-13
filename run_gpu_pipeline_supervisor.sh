#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_gpu_pipeline_supervisor.sh -- bounded-retry wrapper around a pipeline script
# ═══════════════════════════════════════════════════════════════════════════════
#
# What this is for: the wrapped pipeline script already survives an individual
# job's failure (each has its own FAILED_RUNS-style handling) and an individual
# seed's mid-training crash (train_pca.py's epoch-level resume checkpoint). What
# NEITHER of those cover is the orchestrator process itself dying -- e.g. the OOM
# killer hitting the top-level bash process (not just one python subprocess), a
# transient environment issue in the preflight check, or anything else that exits
# the target script with a nonzero code before it reaches its own end. This
# script retries the WHOLE pipeline a bounded number of times when that happens.
#
# This is cheap to do safely BECAUSE of the resume/skip-if-complete layers already
# in place in every pipeline script this wraps: re-running from scratch re-does
# only the fast setup/preflight steps and then picks up exactly where it left off
# per-job (skip-if-complete, resume-if-partial) -- it does NOT mean redoing
# everything already finished.
#
# Deliberately bounded, not infinite: if there's a real, persistent bug, retrying
# forever would just burn GPU-hours while hiding the problem -- exactly what this
# whole audit has been about NOT doing. After MAX_ATTEMPTS, this gives up loudly
# and asks for a human to look, rather than silently looping.
#
# Usage (defaults to run_gpu_pipeline.sh if no argument given, for backward compat):
#   chmod +x run_gpu_pipeline_supervisor.sh
#   nohup bash run_gpu_pipeline_supervisor.sh 2>&1 | tee supervisor_log.txt &
#   nohup bash run_gpu_pipeline_supervisor.sh run_gpu_pipeline_phase5b.sh 2>&1 | tee supervisor_phase5b_log.txt &
# ═══════════════════════════════════════════════════════════════════════════════

MAX_ATTEMPTS=5
SLEEP_BETWEEN_SEC=60
TARGET_SCRIPT="${1:-run_gpu_pipeline.sh}"

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "# Supervisor: target script '${TARGET_SCRIPT}' not found in $(pwd)." >&2
    exit 1
fi

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo ""
    echo "################################################################"
    echo "# Supervisor: attempt ${attempt}/${MAX_ATTEMPTS} at $(date) -- target: ${TARGET_SCRIPT}"
    echo "################################################################"

    bash "$TARGET_SCRIPT"
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        echo ""
        echo "################################################################"
        echo "# Supervisor: ${TARGET_SCRIPT} exited 0 -- pipeline finished. Done."
        echo "################################################################"
        exit 0
    fi

    echo ""
    echo "################################################################"
    echo "# Supervisor: ${TARGET_SCRIPT} exited ${exit_code} (attempt ${attempt}/${MAX_ATTEMPTS})"
    echo "################################################################"

    if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
        echo "# Supervisor: giving up after ${MAX_ATTEMPTS} attempts."
        echo "# This needs a human look, not another retry -- check the most recent"
        echo "# logs/ subdirectory and this supervisor's own output above for what failed."
        exit 1
    fi

    echo "# Supervisor: retrying in ${SLEEP_BETWEEN_SEC}s. Per-job progress is preserved"
    echo "# (skip-if-complete + resume-checkpoint), so this is not starting over."
    sleep "$SLEEP_BETWEEN_SEC"
    attempt=$((attempt + 1))
done
