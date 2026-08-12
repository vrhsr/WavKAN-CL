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
if pgrep -f "run_gpu_pipeline\.sh" >/dev/null 2>&1; then
    PID=$(pgrep -f "run_gpu_pipeline\.sh" | head -1)
    echo "  Running (pid $PID)"
else
    echo "  NOT currently running -- check pipeline_log.txt for how/why it stopped"
fi

echo ""
echo "Seed completion (each arm needs 20):"
for arm in curriculum baseline; do
    dir="results/wavkan_v2_${arm}"
    if [ -d "$dir" ]; then
        n_done=$(find "$dir" -mindepth 2 -maxdepth 2 -name "test_metrics.json" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $arm: $n_done / 20 complete"
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
    echo "  $latest_log"
    echo "  --------------------------------------------------------------"
    tail -n 12 "$latest_log" | sed 's/^/  /'
    echo "  --------------------------------------------------------------"
else
    echo "  No per-step logs found under logs/ yet"
fi

echo ""
echo "Overall pipeline_log.txt (last 8 lines, if it exists in the current directory):"
if [ -f "pipeline_log.txt" ]; then
    tail -n 8 pipeline_log.txt | sed 's/^/  /'
else
    echo "  pipeline_log.txt not found in $(pwd) -- run this from the same directory you launched the pipeline from"
fi

echo ""
echo "================================================================"
