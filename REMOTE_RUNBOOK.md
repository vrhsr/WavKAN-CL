# Remote GPU runbook — Phase 10 component ablation

One experiment, run on the GPU box, brought back here for integration. It is the
only computation the manuscript still needs: PC-WavKAN's central architectural
contribution (PCWI) has never been ablated on the configuration the paper
actually adopts, and that gap cannot be closed from any result already on disk.

**Two arms × 20 seeds, roughly 8–14 GPU-hours.** Nothing here requires root.

---

## 0. Get the code onto the box

```bash
cd ~/projects/WavKAN-CL
git pull
```

Everything the run needs is tracked. Data and results are not (they are
gitignored), so §2 covers those separately.

## 1. Preflight — free, trains nothing

```bash
bash run_remote_experiment.sh --preflight-only
```

Six gates run before any GPU time is spent: environment and CUDA, data
integrity down to exact per-class beat counts, model construction with a real
forward pass and a parameter-count assertion per arm, reference-arm
completeness by seed identity, output writability, and the trainer's CLI
surface. It exits 2 and names the specific problem if anything is wrong.

Fix whatever it reports, then re-run it. Preflight is free; a failed 12-hour
job is not.

## 2. The three things preflight is most likely to stop on

### (a) CUDA cannot initialise

`nvidia-smi` working while `torch.cuda.is_available()` is `False` almost always
means the `nvidia_uvm` kernel module is not loaded: `nvidia-smi` needs only
`/dev/nvidiactl`, but CUDA also needs Unified Memory.

**This is fixable without root.** `nvidia-modprobe` is installed setuid-root
precisely so the CUDA runtime can load UVM on demand:

```bash
nvidia-modprobe -u -c 0
ls -l /dev/nvidia-uvm
python3 -c "import torch; print(torch.cuda.is_available())"
```

`run_remote_experiment.sh` now attempts this automatically before starting
Python — it has to happen there, because torch caches its device count in a
process-local static, so a process that once saw zero GPUs keeps seeing zero
even after the module appears.

If that does not resolve it:

```bash
bash diagnose_gpu.sh
```

Read-only. It checks the userspace/kernel driver versions against each other,
loaded modules, device-node permissions, GPU-hiding environment variables,
compute mode, and the raw `cuInit(0)` code — then prints a ranked verdict
naming the specific fix and its evidence, marking which fixes need an
administrator. If it concludes root is required, that is the answer to send to
whoever administers the box; there is no unprivileged workaround for a
driver/kernel version mismatch.

### (b) Training data absent

The run needs `data/processed_rr_history/`, which is gitignored.

```bash
ls data/processed_rr_history/ 2>/dev/null | head
```

If missing, regenerate it from the MIT-BIH records (`wfdb` downloads them; no
root needed):

```bash
python src/process_data.py            # reads data/raw/, writes data/processed_rr_history/
```

Preflight then verifies the output against the exact beat and per-class counts
recorded in `configs/final_component_ablation.yaml`, so a partial or
wrong-protocol regeneration is caught rather than trained on.

### (c) Reference arm absent

Both new arms are compared, seed by seed, against the already-published
adopted configuration in `results/ablation_no_rr_attn/`. It is **not**
retrained — that would waste GPU time and risk changing a published number.

```bash
ls results/ablation_no_rr_attn/seed_*/test_metrics.json 2>/dev/null | wc -l   # want 20
```

If that is not 20, upload `reference_arm_metrics.tar.gz` from this repository
(1.8 KB — the 20 metrics files, nothing else) and unpack it at the repo root:

```bash
# from the local machine
scp reference_arm_metrics.tar.gz USER@HOST:~/projects/WavKAN-CL/
# on the box
tar -xzf reference_arm_metrics.tar.gz
```

## 3. Run it

```bash
bash run_remote_experiment.sh
```

Long job — use `tmux` or `nohup` if the connection is unreliable:

```bash
tmux new -s pcwi 'bash run_remote_experiment.sh 2>&1 | tee run_console.log'
```

If it is interrupted, resume without redoing finished seeds:

```bash
bash run_remote_experiment.sh --allow-resume
```

Per-seed failures are isolated and logged rather than killing the other 39, and
the runner refuses to overwrite a completed run.

## 4. Bring the results back

On success the script packages everything itself and prints the path:

```
final_component_ablation_<UTC timestamp>.tar.gz
```

Copy it here and unpack at the repository root — it restores
`results/final_component_ablation/` in place:

```bash
# from the local machine
scp USER@HOST:~/projects/WavKAN-CL/final_component_ablation_*.tar.gz .
tar -xzf final_component_ablation_*.tar.gz
```

It contains the top-level artifacts (`results.json`, `provenance.json`,
`config_used.yaml`, `results.csv`, `per_seed.csv`, `RESULTS_README.md`) plus
every per-seed `test_metrics.json`, `training_history.json` and checkpoint —
the checkpoints included deliberately, because they let every reported number
be re-derived here from source rather than trusted.

If the job failed, send `results/final_component_ablation/run.log` and the
console transcript instead.

## 5. Integration (local, deterministic)

```bash
python src/integrate_remote_results.py --dir results/final_component_ablation
```

This verifies the artifacts, seed completeness and provenance — including that
the run's config snapshot matches this repository's — then **recomputes every
statistic locally from the per-seed files and diffs it against the remote
summary to 1e-9**, so transport corruption or a library-version difference
surfaces as a mismatch instead of being quietly accepted. It reports the
numbers and which manuscript claim each bears on, and deliberately does not
edit the manuscript.

It also states in advance what each of the three possible outcomes means for
the paper — including the outcome where PCWI does not replicate on the adopted
configuration and the central claim has to change.

Exit codes: `0` clean, `2` missing or inconsistent artifacts, `3` a statistic
that does not reproduce.

---

## Quick reference

| | |
|---|---|
| Check everything, free | `bash run_remote_experiment.sh --preflight-only` |
| Diagnose CUDA | `bash diagnose_gpu.sh` |
| No-root CUDA fix | `nvidia-modprobe -u -c 0` |
| Run | `bash run_remote_experiment.sh` |
| Resume | `bash run_remote_experiment.sh --allow-resume` |
| Regenerate data | `python src/process_data.py` |
| Integrate locally | `python src/integrate_remote_results.py --dir results/final_component_ablation` |
