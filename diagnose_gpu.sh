#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# diagnose_gpu.sh -- work out why PyTorch cannot initialise CUDA on a machine
# where nvidia-smi works.
#
#     bash diagnose_gpu.sh
#
# Read-only: it inspects and reports, and never changes system state. The most
# likely fix is printed at the end with the evidence for it.
#
# Written 2026-09-04 after run_remote_experiment.sh's preflight correctly
# refused to train on a box where nvidia-smi reported an RTX A4000 but
# torch.cuda.is_available() was False with "CUDA unknown error".
# ---------------------------------------------------------------------------
set -uo pipefail

PYTHON="${PYTHON:-python3}"
hr() { printf '%s\n' "-------------------------------------------------------------------------------"; }
sec() { echo; hr; echo " $1"; hr; }

VERDICTS=()
note() { VERDICTS+=("$1"); }

sec "1. nvidia-smi (userspace driver + GPU visibility)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi 2>&1 | head -12
  SMI_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  echo
  echo "  userspace driver version: ${SMI_DRIVER:-unknown}"
else
  echo "  nvidia-smi NOT FOUND -- no NVIDIA userspace driver installed"
  note "nvidia-smi is absent: install the NVIDIA driver."
  SMI_DRIVER=""
fi

sec "2. Kernel module version (must match userspace exactly)"
if [[ -r /proc/driver/nvidia/version ]]; then
  cat /proc/driver/nvidia/version
  KMOD_DRIVER=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version | head -1)
  echo
  echo "  kernel module version: ${KMOD_DRIVER:-unknown}"
  if [[ -n "${SMI_DRIVER}" && -n "${KMOD_DRIVER}" && "${SMI_DRIVER}" != "${KMOD_DRIVER}" ]]; then
    echo "  *** MISMATCH: userspace ${SMI_DRIVER} vs kernel module ${KMOD_DRIVER} ***"
    note "Driver version mismatch (userspace ${SMI_DRIVER}, kernel ${KMOD_DRIVER}). The driver was upgraded without reloading the kernel module. FIX: reboot, or unload/reload the modules."
  fi
else
  echo "  /proc/driver/nvidia/version not readable -- the nvidia kernel module may not be loaded"
  note "The nvidia kernel module does not appear to be loaded. FIX: sudo modprobe nvidia"
  KMOD_DRIVER=""
fi

sec "3. Loaded NVIDIA kernel modules"
if command -v lsmod >/dev/null 2>&1; then
  lsmod | grep -E '^nvidia' || echo "  (no nvidia modules listed)"
  echo
  for m in nvidia nvidia_uvm nvidia_modeset; do
    if lsmod 2>/dev/null | awk '{print $1}' | grep -qx "$m"; then
      echo "  ${m}: loaded"
    else
      echo "  ${m}: NOT LOADED"
      [[ "$m" == "nvidia_uvm" ]] && note "nvidia_uvm is not loaded. This is the single most common cause of 'CUDA unknown error' on a box where nvidia-smi works: nvidia-smi only needs /dev/nvidiactl, but CUDA needs Unified Memory. FIX: sudo modprobe nvidia_uvm"
    fi
  done
else
  echo "  lsmod unavailable"
fi

sec "4. Device nodes and permissions"
if ls /dev/nvidia* >/dev/null 2>&1; then
  ls -l /dev/nvidia* 2>&1
  echo
  for d in /dev/nvidiactl /dev/nvidia-uvm; do
    if [[ -e "$d" ]]; then
      if [[ -r "$d" && -w "$d" ]]; then
        echo "  ${d}: present, readable+writable by $(id -un)"
      else
        echo "  ${d}: present but NOT accessible by $(id -un)"
        note "${d} exists but $(id -un) cannot read/write it. FIX: add the user to the group owning it (often 'video' or 'render'): sudo usermod -aG video $(id -un), then log out and back in."
      fi
    else
      echo "  ${d}: MISSING"
      [[ "$d" == "/dev/nvidia-uvm" ]] && note "/dev/nvidia-uvm is missing, which CUDA requires. FIX: sudo modprobe nvidia_uvm  (it is normally created on demand; if that fails, reboot)."
    fi
  done
  echo
  echo "  current user: $(id -un)  groups: $(id -Gn)"
else
  echo "  no /dev/nvidia* device nodes at all"
  note "No /dev/nvidia* nodes exist. FIX: sudo modprobe nvidia nvidia_uvm, or reboot."
fi

sec "5. Environment variables that can hide the GPU"
for v in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES CUDA_DEVICE_ORDER \
         CUDA_HOME LD_LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF; do
  if [[ -n "${!v:-}" ]]; then
    echo "  ${v}=${!v}"
    if [[ "$v" == "CUDA_VISIBLE_DEVICES" ]]; then
      case "${!v}" in
        ""|"-1"|"none"|"NoDevFiles")
          note "CUDA_VISIBLE_DEVICES='${!v}' hides every GPU. FIX: unset CUDA_VISIBLE_DEVICES" ;;
      esac
    fi
  else
    echo "  ${v} unset"
  fi
done

sec "6. What PyTorch itself reports"
${PYTHON} - <<'PY' 2>&1
import ctypes, os, sys
try:
    import torch
except Exception as e:
    print("  cannot import torch:", e); sys.exit(0)
print(f"  torch            : {torch.__version__}")
print(f"  compiled for CUDA: {torch.version.cuda}")
print(f"  cuda.is_available: {torch.cuda.is_available()}")
try:
    print(f"  device_count     : {torch.cuda.device_count()}")
except Exception as e:
    print(f"  device_count     : raised {e!r}")

# The raw driver API error code is far more diagnostic than the warning text.
for lib in ("libcuda.so.1", "libcuda.so"):
    try:
        cuda = ctypes.CDLL(lib)
        break
    except OSError:
        cuda = None
if cuda is None:
    print("  libcuda.so.1     : NOT LOADABLE -- the driver library is missing from the linker path")
else:
    rc = cuda.cuInit(0)
    names = {0: "CUDA_SUCCESS", 100: "CUDA_ERROR_NO_DEVICE",
             101: "CUDA_ERROR_INVALID_DEVICE", 200: "CUDA_ERROR_INVALID_IMAGE",
             201: "CUDA_ERROR_INVALID_CONTEXT", 205: "CUDA_ERROR_MAP_FAILED",
             304: "CUDA_ERROR_OPERATING_SYSTEM", 802: "CUDA_ERROR_SYSTEM_NOT_READY",
             803: "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
             804: "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE",
             999: "CUDA_ERROR_UNKNOWN"}
    print(f"  raw cuInit(0)    : {rc}  ({names.get(rc, 'see CUDA driver API docs')})")
    hints = {
        803: "SYSTEM_DRIVER_MISMATCH: kernel module and userspace library versions differ -> reboot.",
        802: "SYSTEM_NOT_READY: driver is up but a component (often nvidia_uvm) is not -> sudo modprobe nvidia_uvm.",
        304: "OPERATING_SYSTEM: usually /dev/nvidia-uvm missing or permission-denied.",
        100: "NO_DEVICE: the GPU is not visible to this process -> check CUDA_VISIBLE_DEVICES.",
        999: "UNKNOWN: most often nvidia_uvm not loaded, a stale driver after upgrade, or device-node permissions.",
    }
    if rc in hints:
        print(f"  interpretation   : {hints[rc]}")
PY

sec "7. Compute mode / MIG / other processes holding the GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_mode,mig.mode.current,memory.used,persistence_mode \
             --format=csv 2>&1 | head -5
  echo
  echo "  processes currently on the GPU:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | head -8
  CM=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  case "$CM" in
    *Exclusive*|*Prohibited*)
      note "GPU compute mode is '${CM}', which can block a new process. FIX: sudo nvidia-smi -c 0 (DEFAULT mode)." ;;
  esac
fi

sec "VERDICT"
if ((${#VERDICTS[@]} == 0)); then
  cat <<'EOF'
  No specific cause identified by these checks.

  If section 6 shows cuda.is_available: True, the original failure was
  transient -- just re-run:

      bash run_remote_experiment.sh

  If it still shows False, try in this order (least disruptive first):

      sudo modprobe nvidia_uvm            # most common fix
      unset CUDA_VISIBLE_DEVICES
      sudo nvidia-smi -pm 1               # enable persistence mode
      sudo reboot                         # resolves driver-upgrade mismatches

  Then re-check cheaply, without spending GPU time:

      bash run_remote_experiment.sh --preflight-only
EOF
else
  echo "  Findings, most likely first:"
  echo
  i=1
  for v in "${VERDICTS[@]}"; do
    echo "  ${i}. ${v}"
    echo
    ((i++))
  done
  cat <<'EOF'
  After applying a fix, verify cheaply before committing GPU hours:

      bash run_remote_experiment.sh --preflight-only

  That runs every check (environment, data integrity, model construction,
  parameter counts, reference-arm completeness) and trains nothing.
EOF
fi
hr
