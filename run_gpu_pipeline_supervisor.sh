#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_gpu_pipeline_supervisor.sh -- bounded-retry wrapper around run_gpu_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════════
#
# What this is for: run_gpu_pipeline.sh itself already survives an individual
# seed's failure (see its own FAILED_RUNS handling) and an individual seed's
# mid-training crash (train_pca.py's epoch-level resume checkpoint). What NEITHER
# of those cover is the orchestrator process itself dying -- e.g. the OOM killer
# hitting the top-level bash process (not just one python subprocess), a
# transient environment issue in the preflight check, or anything else that exits
# run_gpu_pipeline.sh with a nonzero code before it reaches its own end. This
# script retries the WHOLE pipeline a bounded number of times when that happens.
#
# This is cheap to do safely BECAUSE of the two resume layers already in place:
# re-running run_gpu_pipeline.sh from scratch re-does only the fast setup steps
# (data check is a no-op if already done, smoke test takes a couple minutes) and
# then Phase 2 picks up exactly where it left off per-seed (skip-if-complete,
# resume-if-partial) -- it does NOT mean redoing all 40 runs.
#
# Deliberately bounded, not infinite: if there's a real, persistent bug, retrying
# forever would just burn GPU-hours while hiding the problem -- exactly what this
# whole audit has been about NOT doing. After MAX_ATTEMPTS, this gives up loudly
# and asks for a human to look, rather than silently looping.
#
# Usage:
#   chmod +x run_gpu_pipeline_supervisor.sh
#   nohup bash run_gpu_pipeline_supervisor.sh 2>&1 | tee supervisor_log.txt &
# ═══════════════════════════════════════════════════════════════════════════════

MAX_ATTEMPTS=5
SLEEP_BETWEEN_SEC=60

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo ""
    echo "################################################################"
    echo "# Supervisor: attempt ${attempt}/${MAX_ATTEMPTS} at $(date)"
    echo "################################################################"

    bash run_gpu_pipeline.sh
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        echo ""
        echo "################################################################"
        echo "# Supervisor: run_gpu_pipeline.sh exited 0 -- pipeline finished. Done."
        echo "################################################################"
        exit 0
    fi

    echo ""
    echo "################################################################"
    echo "# Supervisor: run_gpu_pipeline.sh exited ${exit_code} (attempt ${attempt}/${MAX_ATTEMPTS})"
    echo "################################################################"

    if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
        echo "# Supervisor: giving up after ${MAX_ATTEMPTS} attempts."
        echo "# This needs a human look, not another retry -- check the most recent"
        echo "# logs/ subdirectory and this supervisor's own output above for what failed."
        exit 1
    fi

    echo "# Supervisor: retrying in ${SLEEP_BETWEEN_SEC}s. Per-seed progress is preserved"
    echo "# (skip-if-complete + resume-checkpoint), so this is not starting over."
    sleep "$SLEEP_BETWEEN_SEC"
    attempt=$((attempt + 1))
done
