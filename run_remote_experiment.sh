#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_remote_experiment.sh -- one-command entry point for the Phase 10
# component ablation on the adopted PC-WavKAN architecture.
#
# Copy the repository to the GPU machine and run:
#
#     bash run_remote_experiment.sh
#
# Everything else -- environment checks, data integrity, parameter counts,
# training, completeness verification, statistics, plots, provenance and the
# results README -- is handled by run_remote_experiment.py, which fails loudly
# and early if any required input is missing or wrong.
#
# Restart after an interruption (skips seeds that already finished, never
# overwrites a completed run):
#
#     bash run_remote_experiment.sh --allow-resume
#
# Check everything without spending GPU time:
#
#     bash run_remote_experiment.sh --preflight-only
# ---------------------------------------------------------------------------
set -Eeuo pipefail

CONFIG="${CONFIG:-configs/final_component_ablation.yaml}"
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")"

echo "==============================================================================="
echo " PC-WavKAN Phase 10 -- component ablation on the adopted architecture"
echo "==============================================================================="
echo " repo   : $(pwd)"
echo " config : ${CONFIG}"
echo " python : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo " started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 2
fi

# pyyaml is the only dependency the runner needs beyond the training stack.
if ! ${PYTHON} -c "import yaml" >/dev/null 2>&1; then
  echo "pyyaml is missing; installing it (the only extra dependency)..."
  ${PYTHON} -m pip install --quiet pyyaml || {
    echo "ERROR: could not install pyyaml. Install it and re-run." >&2; exit 2; }
fi

# nvidia-smi is informational only; the runner does the authoritative CUDA check.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
             --format=csv,noheader 2>/dev/null || true
  echo
fi

# Long job: keep it alive if the shell dies, but stay in the foreground so the
# operator sees progress. Use nohup/tmux yourself if you want to detach.
set +e
${PYTHON} run_remote_experiment.py --config "${CONFIG}" "$@"
RC=$?
set -e

echo
echo "==============================================================================="
if [[ ${RC} -eq 0 ]]; then
  OUT=$(${PYTHON} - <<'PY'
import yaml, os
cfg = yaml.safe_load(open(os.environ.get("CONFIG", "configs/final_component_ablation.yaml")))
print(cfg["experiment"]["output_root"])
PY
)
  echo " SUCCESS"
  echo
  echo " Copy this directory back to the local repository, same path:"
  echo "     ${OUT}/"
  echo
  echo " Then, locally:"
  echo "     python src/integrate_remote_results.py --dir ${OUT}"
  echo "     python src/verify_manuscript_numbers.py"
  echo
  echo " Read first: ${OUT}/RESULTS_README.md"
else
  echo " FAILED (exit ${RC})"
  echo
  echo "   2 = preflight failure  -- an input is missing or wrong; nothing was trained"
  echo "   3 = training failure"
  echo "   4 = postflight failure -- runs completed but verification found gaps"
  echo
  echo " The console transcript is in the run.log inside the output directory."
  echo " Preflight failures are safe to fix and re-run; no GPU time was spent."
fi
echo " finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==============================================================================="
exit ${RC}
