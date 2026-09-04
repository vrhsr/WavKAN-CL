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
# exported because the success block below reads it from a Python heredoc
export CONFIG
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

# On many multi-user boxes nvidia-smi works while CUDA cannot initialise,
# because the nvidia_uvm kernel module is not loaded: nvidia-smi needs only
# /dev/nvidiactl, but CUDA also needs Unified Memory. The CUDA runtime normally
# loads it on demand through nvidia-modprobe, which is installed setuid-root
# for exactly that purpose -- so an unprivileged user can perform this repair.
#
# It must happen before Python starts: torch caches its device count in a
# function-local static, so a process that once saw zero GPUs keeps seeing
# zero even after the module appears.
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! ${PYTHON} -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "--- CUDA is not initialising; attempting the no-root repair ---"
    if command -v nvidia-modprobe >/dev/null 2>&1; then
      nvidia-modprobe -u -c 0 >/dev/null 2>&1 || true
      nvidia-modprobe -c 0    >/dev/null 2>&1 || true
      if ${PYTHON} -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        echo "    nvidia-modprobe loaded the UVM module; CUDA is now available."
      else
        echo "    nvidia-modprobe did not resolve it. Run 'bash diagnose_gpu.sh'"
        echo "    for the specific cause; preflight below will also report it."
      fi
    else
      echo "    nvidia-modprobe is not installed, so the no-root repair is"
      echo "    unavailable. Run 'bash diagnose_gpu.sh' for the specific cause."
    fi
    echo
  fi
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

  # Package the results for transport. The integration script needs the
  # top-level artifacts plus every per-seed test_metrics.json; the checkpoints
  # and histories are included because they are small and let every number be
  # re-derived locally from source rather than trusted.
  STAMP=$(date -u +%Y%m%d-%H%M%SZ)
  TARBALL="final_component_ablation_${STAMP}.tar.gz"
  if tar -czf "${TARBALL}" "${OUT}" 2>/dev/null; then
    echo " Packaged for transport:"
    echo "     $(pwd)/${TARBALL}   ($(du -h "${TARBALL}" | cut -f1))"
    echo
    echo " Copy it to the local repository root and unpack there:"
    echo "     tar -xzf ${TARBALL}          # restores ${OUT}/ in place"
  else
    echo " Copy this directory back to the local repository, same path:"
    echo "     ${OUT}/"
  fi
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
