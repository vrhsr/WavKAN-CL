# Remote GPU runbook — Phase 10 component ablation

One experiment, run on the GPU box, brought back here for integration. It is the
only computation the manuscript still needs: PC-WavKAN's central architectural
contribution (PCWI) has never been ablated on the configuration the paper
actually adopts, and that gap cannot be closed from any result already on disk.

**Four arms × 20 seeds, roughly 18 GPU-hours on an RTX A4000** (the first
attempt ran 60 runs in 13.7 h). Nothing here requires root.

> **Read this before starting — what went wrong on 2026-09-26.**
> The first run of this job trained all three arms on the wrong records and
> cannot be used (`AUDIT_FINDINGS.md` H49; kept for the record at
> `results/final_component_ablation_INVALID_pre_h16_split/`). The config's
> expected beat counts had been copied from a stale table and described an
> older split. So preflight rejected the box's **correct** extraction
> (40,177 training beats) for three weeks, and the box was then made to pass
> by regenerating the old split (40,726). Two things now prevent a repeat:
>
> 1. **Never edit `src/split.py`, `src/process_data.py` or the arrays to make
>    preflight pass.** If an *unmodified* regeneration disagrees with the
>    expected counts, stop and report it — the config may be what is wrong.
>    The expected counts are now derived from the PhysioNet annotations
>    (`configs/mitbih_split_counts.json`), and the runner refuses a config
>    that disagrees with them.
> 2. **Preflight 2b re-runs the 20 published checkpoints on your arrays** and
>    requires each to reproduce its own recorded validation and test score.
>    That checks the data itself, not just its labels. It is never to be
>    relaxed to get past it.
>
> The job also now re-trains the published configuration itself
> (`reference_replicate`), so every ablation is compared against a reference
> trained on the same machine, data and code.

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

Seven gates run before any GPU time is spent: environment and CUDA; data
integrity down to exact per-class beat counts (themselves checked against
`configs/mitbih_split_counts.json`); **data equivalence** — the 20 published
reference checkpoints re-evaluated on this box's arrays must reproduce their
own recorded scores (a few minutes on GPU); model construction with a real
forward pass and a parameter-count assertion per arm; reference-arm
completeness by seed identity; output writability; and the trainer's CLI
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

If missing — **or if it was ever produced by a modified `split.py` /
`process_data.py`, which is true of the arrays used on 2026-09-26** — delete it
and regenerate from the MIT-BIH records with the repository's code exactly as
checked out (`wfdb` downloads them; no root needed):

```bash
git status --short src/split.py src/process_data.py   # must print nothing
rm -rf data/processed_rr_history
python src/process_data.py            # reads data/raw/, writes data/processed_rr_history/
```

Expected result: **train 40,177 · val 10,815 · test 49,684 beats** (validation
= records 208, 209, 223, 230). Preflight checks these counts and then gate 2b
checks the arrays themselves. If an unmodified regeneration gives different
numbers, do not work around it: send `run.log`.

### (c) Reference arm absent

The published adopted configuration in `results/ablation_no_rr_attn/` is used
twice and never written to: gate 2b re-evaluates its checkpoints to prove the
data matches, and the drift check compares it with `reference_replicate`. Its
`best_model.pth`, `training_history.json` and `test_metrics.json` are all
git-tracked, so `git pull` normally suffices:

```bash
for f in best_model.pth training_history.json test_metrics.json; do
  echo "$f: $(ls results/ablation_no_rr_attn/seed_*/$f 2>/dev/null | wc -l)/20"; done
```

If any count is not 20, upload `reference_arm_for_equivalence.tar.gz` (about
13 MB — exactly those 60 files) and unpack it at the repo root:

```bash
# from the local machine
scp reference_arm_for_equivalence.tar.gz USER@HOST:~/projects/WavKAN-CL/
# on the box
tar -xzf reference_arm_for_equivalence.tar.gz
```

(The older `reference_arm_metrics.tar.gz` holds only the metrics files and is
not enough for gate 2b.)

## 3. Run it

```bash
bash run_remote_experiment.sh
```

Long job — use `tmux` or `nohup` if the connection is unreliable:

```bash
tmux new -s pcwi 'bash run_remote_experiment.sh 2>&1 | tee run_console.log'
```

If `results/final_component_ablation/` still holds the invalid 2026-09-26 run, move it aside first — never resume onto it:

```bash
mv results/final_component_ablation results/final_component_ablation_INVALID_pre_h16_split
```

If *this* job is interrupted, resume without redoing finished seeds. The runner writes a data fingerprint (`.run_fingerprint.json`) before the first seed and refuses `--allow-resume` unless it matches the current data, seeds and arms:

```bash
bash run_remote_experiment.sh --allow-resume
```

Per-seed failures are isolated and logged rather than killing the other 79, and
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
the run's config snapshot matches this repository's — and then
**comparability**: the run's training counts must equal the PhysioNet-derived
published split and its data-equivalence gate must have passed, or it is
refused outright. (The 2026-09-26 run passed every other check and was
reported CLEAN; this gate is what it lacked.) Only then does it **recompute
every statistic locally from the per-seed files and diff it against the remote
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
| Regenerate data | `rm -rf data/processed_rr_history && python src/process_data.py` (unmodified code) |
| Expected counts | train 40,177 · val 10,815 · test 49,684 |
| Integrate locally | `python src/integrate_remote_results.py --dir results/final_component_ablation` |
