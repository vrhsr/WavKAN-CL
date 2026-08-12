#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# pipeline_status.sh -- read-only snapshot of run_gpu_pipeline.sh's progress.
# Safe to run anytime, from a separate terminal/tmux pane, while the pipeline runs.
# Does not touch or interfere with the running pipeline in any way.
#
# Usage:
#   bash pipeline_status.sh          # one-shot snapshot
#   watch -n 30 bash pipeline_status.sh   # auto-refresh every 30s
# ═══════════════════════════════════════════════════════════════════════════════

echo "================================================================"
echo "WavKAN-CL Pipeline Status — $(date)"
echo "================================================================"

echo ""
echo "GPU:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits | awk -F', ' '{printf "  Util: %s%%   VRAM: %s/%s MiB   Temp: %s C\n", $1,$2,$3,$4}'
else
    echo "  nvidia-smi not available"
fi

echo ""
echo "Background job:"
# Tightened 2026-08-13: plain "run_gpu_pipeline\.sh" false-positived on an unrelated
# process from a different project during this Phase 5 session. Requiring "bash" (or
# "sh") immediately before the script name is still not foolproof (pgrep -f matches
# substrings of the whole command line, so this can't be made airtight without also
# checking cwd/ppid), but it's meaningfully narrower than before.
PATTERN="(bash|sh)[^|]*run_gpu_pipeline(_supervisor)?\.sh"
if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    PID=$(pgrep -f "$PATTERN" | head -1)
    echo "  Running (pid $PID) -- if this still looks wrong, cross-check with: ps -p $PID -f"
else
    echo "  NOT currently running -- check pipeline_log.txt for how/why it stopped"
fi

echo ""
echo "Smoke test (Phase 1 -- these dirs are deleted right after the smoke test finishes,"
echo "so seeing nothing here just means Phase 1 already completed or hasn't started):"
for arm in curriculum baseline; do
    dir="results/smoke_${arm}"
    if [ -f "${dir}/test_metrics.json" ]; then
        echo "  smoke_$arm: complete"
    elif [ -d "$dir" ]; then
        echo "  smoke_$arm: in progress (dir exists, no test_metrics.json yet)"
    else
        echo "  smoke_$arm: not present (either not started, or already finished+cleaned up)"
    fi
done

echo ""
echo "Seed completion, Phase 2 (each arm needs 20):"
for arm in curriculum baseline; do
    dir="results/wavkan_v2_${arm}"
    if [ -d "$dir" ]; then
        n_done=$(find "$dir" -mindepth 2 -maxdepth 2 -name "test_metrics.json" 2>/dev/null | wc -l | tr -d ' ')
        # A resume_checkpoint.pth with no test_metrics.json alongside it means that
        # seed is either currently training or previously failed mid-run (added
        # 2026-08-13 alongside train_pca.py's real epoch-level resume support) --
        # worth distinguishing from "never touched at all".
        n_resumable=0
        for d in "$dir"/seed_*/; do
            if [ -f "${d}resume_checkpoint.pth" ] && [ ! -f "${d}test_metrics.json" ]; then
                n_resumable=$((n_resumable + 1))
            fi
        done
        echo "  $arm: $n_done / 20 complete" $([ "$n_resumable" -gt 0 ] && echo "  (+$n_resumable in-progress/retryable via resume checkpoint)")
    else
        echo "  $arm: not started yet"
    fi
done

echo ""
echo "Aggregated comparison:"
if [ -f "results/wavkan_v2_20seed_comparison.json" ]; then
    echo "  ✓ results/wavkan_v2_20seed_comparison.json exists -- pipeline has reached Phase 3"
else
    echo "  Not yet produced (Phase 3 hasn't run)"
fi

echo ""
echo "Most recently modified log (tail):"
latest_log=$(ls -t logs/*/*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    log_age_sec=$(( $(date +%s) - $(stat -c %Y "$latest_log" 2>/dev/null || stat -f %m "$latest_log" 2>/dev/null || echo 0) ))
    echo "  $latest_log  (last modified ${log_age_sec}s ago)"
    if [ "$log_age_sec" -gt 120 ]; then
        echo "  *** STALE: no update in >2 min. If a job is 'Running' above but this log is old,"
        echo "  *** the current run likely isn't writing here -- check how it was launched."
    fi
    echo "  --------------------------------------------------------------"
    tail -n 12 "$latest_log" | sed 's/^/  /'
    echo "  --------------------------------------------------------------"
else
    echo "  No per-step logs found under logs/ yet"
fi

echo ""
echo "Overall pipeline_log.txt (last 8 lines, if it exists in the current directory):"
if [ -f "pipeline_log.txt" ]; then
    plog_age_sec=$(( $(date +%s) - $(stat -c %Y "pipeline_log.txt" 2>/dev/null || stat -f %m "pipeline_log.txt" 2>/dev/null || echo 0) ))
    echo "  (last modified ${plog_age_sec}s ago)"
    if [ "$plog_age_sec" -gt 120 ]; then
        echo "  *** STALE -- this is very likely leftover from a PREVIOUS run, not the current one."
        echo "  *** Only trust this if you launched the current run with:"
        echo "  ***   nohup bash run_gpu_pipeline.sh 2>&1 | tee pipeline_log.txt &"
        echo "  *** If you instead ran 'bash run_gpu_pipeline.sh' directly (e.g. inside tmux without"
        echo "  *** the tee wrapper), this file is simply not being updated -- check the tmux pane"
        echo "  *** itself, or the per-step logs above, instead."
    fi
    tail -n 8 pipeline_log.txt | sed 's/^/  /'
else
    echo "  pipeline_log.txt not found in $(pwd) -- run this from the same directory you launched the pipeline from"
fi

echo ""
echo "================================================================"
