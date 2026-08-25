#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# pipeline_status.sh -- read-only snapshot of run_gpu_pipeline.sh AND/OR
# run_gpu_pipeline_phase5b.sh progress (both sections always print -- whichever
# one hasn't been launched yet just shows all-zero/not-started, same as before).
# Safe to run anytime, from a separate terminal/tmux pane, while a pipeline runs.
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
# checking cwd/ppid), but it's meaningfully narrower than before. Extended same day
# to also recognize run_gpu_pipeline_phase5b.sh (previously only matched the main
# pipeline / its supervisor).
PATTERN="(bash|sh)[^|]*run_gpu_pipeline(_supervisor|_phase5b)?\.sh"
if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    for PID in $(pgrep -f "$PATTERN"); do
        CMD=$(ps -p "$PID" -o args= 2>/dev/null | sed 's/^ *//')
        echo "  Running (pid $PID): $CMD"
    done
    echo "  If any of the above still looks wrong, cross-check with: ps -p <pid> -f"
else
    echo "  NOT currently running -- check pipeline_log.txt / supervisor_*_log.txt for how/why it stopped"
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
echo "----------------------------------------------------------------"
echo "Phase 5b (run_gpu_pipeline_phase5b.sh) -- Fig.3 baselines, Table 4 ablations,"
echo "Fig.6 RR-ablation, cross-dataset:"
echo "----------------------------------------------------------------"

echo ""
echo "[1] Fig. 3 baselines (each needs 20 seeds):"
for model in resnet1d transformer cnn_focal bspline_kan; do
    dir="results/baseline_${model}"
    if [ -d "$dir" ]; then
        n_done=$(find "$dir" -mindepth 2 -maxdepth 2 -name "test_metrics.json" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $model: $n_done / 20 complete"
    else
        echo "  $model: not started yet"
    fi
done

echo ""
echo "[2] Table 4 ablation matrix (each config needs 20 seeds):"
for name in wavelet_morlet wavelet_dog wavelet_bspline no_pcwi no_pwam no_rr_attn; do
    dir="results/ablation_${name}"
    if [ -d "$dir" ]; then
        n_done=$(find "$dir" -mindepth 2 -maxdepth 2 -name "test_metrics.json" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $name: $n_done / 20 complete"
    else
        echo "  $name: not started yet"
    fi
done

echo ""
echo "[3] Fig. 6 RR-ablation:"
if [ -f "results/rr_ablation_real/rr_ablation_report.json" ] && [ -f "results/rr_ablation_real/rr_ablation_figure.pdf" ]; then
    echo "  complete (results/rr_ablation_real/rr_ablation_{report.json,figure.pdf})"
else
    echo "  not complete yet"
fi

echo ""
echo "[4] PTB-XL zero-shot cross-dataset eval:"
if [ -f "results/ptbxl_zero_shot_metrics_real.json" ]; then
    echo "  complete (results/ptbxl_zero_shot_metrics_real.json)"
elif [ -f "data/processed_ptbxl/X_test.npy" ]; then
    echo "  data present, eval not yet complete (or still running)"
else
    echo "  skipped by the pipeline -- data/processed_ptbxl not regenerated yet"
fi

echo ""
echo "----------------------------------------------------------------"
echo "Phase 6 (run_gpu_pipeline_phase6.sh) -- INCART/SVDB data + multi-dataset"
echo "table, multi-seed PTB-XL, augmentation-study re-run with real logging:"
echo "----------------------------------------------------------------"

echo ""
echo "[1] INCART / SVDB data regeneration:"
for ds in incart svdb; do
    if [ -f "data/${ds}_processed/X_test.npy" ]; then
        echo "  $ds: data present"
    else
        echo "  $ds: not regenerated yet"
    fi
done

echo ""
echo "[2] Multi-dataset zero-shot table (WavKAN-v2 + 4 baselines x 20 seeds x 3 datasets):"
if [ -f "results/multidataset_table/multidataset_raw.json" ]; then
    echo "  complete (results/multidataset_table/multidataset_raw.json, killer_table.tex, killer_table.txt)"
else
    echo "  not complete yet (needs step [1] first)"
fi

echo ""
echo "[3] Multi-seed PTB-XL zero-shot (20 seeds):"
if [ -d "results/ptbxl_zeroshot_multiseed" ]; then
    n_done=$(find "results/ptbxl_zeroshot_multiseed" -maxdepth 1 -name "seed_*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $n_done / 20 per-seed evals complete"
fi
if [ -f "results/ptbxl_zero_shot_metrics_multiseed.json" ]; then
    echo "  aggregated summary: results/ptbxl_zero_shot_metrics_multiseed.json"
else
    echo "  aggregated summary: not yet produced"
fi

echo ""
echo "[4] Augmentation study re-run (real per-seed logging, 20 seeds):"
if [ -f "results/noise_augmentation/augmentation_comparison.json" ]; then
    echo "  complete (results/noise_augmentation/augmentation_comparison.json -- has real per-seed 'raw' values this time)"
else
    echo "  not complete yet"
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
echo "Overall log file (last 8 lines, checking common names in the current directory):"
FOUND_OVERALL_LOG=0
for candidate in supervisor_phase5b_log.txt supervisor_log.txt pipeline_log.txt; do
    if [ -f "$candidate" ]; then
        FOUND_OVERALL_LOG=1
        log_age_sec=$(( $(date +%s) - $(stat -c %Y "$candidate" 2>/dev/null || stat -f %m "$candidate" 2>/dev/null || echo 0) ))
        echo "  $candidate (last modified ${log_age_sec}s ago)"
        if [ "$log_age_sec" -gt 120 ]; then
            echo "  *** STALE -- this may be leftover from a PREVIOUS run rather than one currently active."
            echo "  *** If you ran a pipeline script directly (e.g. inside tmux without the 'tee' wrapper),"
            echo "  *** this file simply isn't being updated -- check the tmux pane itself, or the"
            echo "  *** per-step logs above, instead."
        fi
        tail -n 8 "$candidate" | sed 's/^/  /'
        echo ""
    fi
done
if [ "$FOUND_OVERALL_LOG" -eq 0 ]; then
    echo "  None of supervisor_phase5b_log.txt / supervisor_log.txt / pipeline_log.txt found in $(pwd)"
    echo "  -- run this from the same directory you launched the pipeline from, or check the tmux pane directly."
fi

echo ""
echo "================================================================"
