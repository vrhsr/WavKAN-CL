#!/usr/bin/env python3
"""
run_remote_experiment.py -- self-validating remote-GPU runner for the Phase 10
component ablation on the adopted architecture.

    python run_remote_experiment.py --config configs/final_component_ablation.yaml

Design goals, in order:
  1. Fail loudly BEFORE spending GPU hours if any required input is wrong.
  2. Never silently overwrite an earlier run.
  3. Be restartable: a killed job resumes at the first unfinished seed.
  4. Produce everything the local machine needs to integrate the result,
     including provenance, so the GPU box can be released immediately after.

Everything about the recipe comes from the YAML config, which documents why each
value is what it is: the arms trained here must differ from the already-published
reference arm in exactly one flag, or the paired comparison is not a
single-variable ablation.

Stages
------
  preflight   environment, data integrity, model construction, parameter counts,
              reference-arm completeness, output directory, git revision
  train       one arm x one seed at a time, skipping completed seeds
  postflight  seed completeness, NaN screen, paired-seed alignment
  stats       paired Wilcoxon + Holm + Cohen's d + 95% CI vs the reference arm
  report      results.json, results.csv, RESULTS_README.md, plots, logs

Exit codes
----------
  0 success   2 preflight failure   3 training failure   4 postflight failure
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

EXIT_PREFLIGHT, EXIT_TRAIN, EXIT_POSTFLIGHT = 2, 3, 4


# ───────────────────────────────────────────────────────────── logging helpers
class Tee:
    """Duplicate stdout/stderr into the run log so the transcript is an artifact."""

    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def say(msg=""):
    print(msg, flush=True)


def head(title):
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def die(code, msg):
    say()
    say("!" * 78)
    say("ABORT: " + msg)
    say("!" * 78)
    sys.exit(code)


# ───────────────────────────────────────────────────────────────── preflight
def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        die(EXIT_PREFLIGHT, "pyyaml is not installed (pip install pyyaml)")
    if not path.exists():
        die(EXIT_PREFLIGHT, f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("experiment", "arms", "training", "seeds", "data", "statistics"):
        if key not in cfg:
            die(EXIT_PREFLIGHT, f"config is missing the '{key}' section")
    if len(cfg["seeds"]) != len(set(cfg["seeds"])):
        die(EXIT_PREFLIGHT, "config seed list contains duplicates")
    return cfg


def git_revision() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=REPO, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "describe": run("git", "describe", "--always", "--dirty"),
    }


def check_environment(cfg) -> dict:
    head("PREFLIGHT 1/6 -- environment")
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    say(f"  python    : {info['python']}")
    say(f"  platform  : {info['platform']}")
    say(f"  hostname  : {info['hostname']}")

    try:
        import torch
    except ImportError:
        die(EXIT_PREFLIGHT, "PyTorch is not installed")
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    say(f"  torch     : {torch.__version__}")

    if not torch.cuda.is_available():
        # Distinguish "there is no GPU here" from "there is a GPU but CUDA
        # cannot initialise". They look identical to torch.cuda.is_available()
        # and have completely different fixes, so report which one it is and
        # the raw driver-API error code, which is far more diagnostic than
        # torch's own warning text.
        smi, gpu_line = shutil.which("nvidia-smi"), ""
        if smi:
            try:
                gpu_line = subprocess.run(
                    [smi, "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=30).stdout.strip()
            except Exception:
                gpu_line = ""

        rc, rc_name = None, ""
        try:
            import ctypes
            for lib in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
                try:
                    rc = ctypes.CDLL(lib).cuInit(0)
                    break
                except OSError:
                    continue
            rc_name = {0: "CUDA_SUCCESS", 100: "CUDA_ERROR_NO_DEVICE",
                       304: "CUDA_ERROR_OPERATING_SYSTEM",
                       802: "CUDA_ERROR_SYSTEM_NOT_READY",
                       803: "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
                       999: "CUDA_ERROR_UNKNOWN"}.get(rc, "")
        except Exception:
            pass

        if gpu_line:
            # A GPU is physically present and the userspace driver works, so
            # this is an environment fault, not a wrong-machine mistake.
            msg = [
                "nvidia-smi reports a GPU but PyTorch cannot initialise CUDA.",
                "",
                f"       GPU (per nvidia-smi) : {gpu_line}",
                f"       torch                : {torch.__version__} "
                f"(built for CUDA {torch.version.cuda})",
            ]
            if rc is not None:
                msg.append(f"       raw cuInit(0)         : {rc}"
                           + (f"  ({rc_name})" if rc_name else ""))
            msg += [
                "",
                "       This is an environment fault on this machine, not a problem with",
                "       the job or the data. The usual causes, most common first:",
                "",
                "         1. nvidia_uvm kernel module not loaded. nvidia-smi only needs",
                "            /dev/nvidiactl, but CUDA also needs Unified Memory.",
                "              ls -l /dev/nvidia-uvm",
                "            NO ROOT NEEDED -- nvidia-modprobe is setuid-root for exactly",
                "            this, which is how CUDA normally loads UVM on demand:",
                "              nvidia-modprobe -u -c 0",
                "            With root, equivalently:  sudo modprobe nvidia_uvm",
                "         2. Driver upgraded without reloading the kernel module, so the",
                "            userspace and kernel versions differ.",
                "              cat /proc/driver/nvidia/version   (compare with nvidia-smi)",
                "              -> reboot",
                "         3. /dev/nvidia* nodes exist but this user cannot open them.",
                "              sudo usermod -aG video $(id -un)   (then re-login)",
                "         4. CUDA_VISIBLE_DEVICES set to '' or '-1'.",
                "              unset CUDA_VISIBLE_DEVICES",
                "",
                "       Run the bundled diagnostic, which checks all of these and prints",
                "       the specific fix with its evidence:",
                "",
                "           bash diagnose_gpu.sh",
                "",
                "       Then re-check for free, without training:",
                "",
                "           bash run_remote_experiment.sh --preflight-only",
            ]
            die(EXIT_PREFLIGHT, "\n".join(msg))

        die(EXIT_PREFLIGHT,
            "CUDA is not available and nvidia-smi reports no GPU, so this machine "
            "appears to have none.\n"
            "       This job trains 20 seeds x N arms and is not intended for CPU.\n"
            "       Run it on the GPU box, or pass --allow-cpu if you really mean CPU.")
    info["cuda"] = torch.version.cuda
    info["gpu_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
    say(f"  cuda      : {torch.version.cuda}")
    say(f"  gpu       : {info['gpu_name']} ({info['gpu_memory_gb']} GB)")
    if info["gpu_memory_gb"] < 3:
        die(EXIT_PREFLIGHT, f"GPU has only {info['gpu_memory_gb']} GB; need >= 3 GB")

    for pkg in ("numpy", "scipy", "sklearn"):
        try:
            __import__(pkg)
        except ImportError:
            die(EXIT_PREFLIGHT, f"required package missing: {pkg}")
    say("  packages  : numpy, scipy, scikit-learn present")

    info["git"] = git_revision()
    say(f"  git       : {info['git']['describe'] or 'n/a'} "
        f"(branch {info['git']['branch'] or '?'}"
        f"{', DIRTY' if info['git']['dirty'] else ''})")
    if info["git"]["dirty"]:
        say("  NOTE: working tree is dirty. The exact code state is recorded in "
            "provenance.json, but a clean checkout is preferable for a published run.")
    return info


def check_data(cfg) -> dict:
    head("PREFLIGHT 2/6 -- data integrity")
    import numpy as np

    d = Path(cfg["data"]["dir"])
    if not d.is_dir():
        die(EXIT_PREFLIGHT,
            f"data directory not found: {d}\n"
            f"       Regenerate it on this machine with:  python src/process_data.py\n"
            f"       (it needs the MIT-BIH raw records under data/raw/)")

    missing = [f for f in cfg["data"]["required_files"] if not (d / f).exists()]
    if missing:
        die(EXIT_PREFLIGHT, f"data directory {d} is missing: {', '.join(missing)}")
    say(f"  all {len(cfg['data']['required_files'])} required arrays present in {d}")

    out = {}
    for split, expect_n in cfg["data"]["expect_beats"].items():
        X = np.load(d / f"X_{split}.npy", mmap_mode="r")
        Xrr = np.load(d / f"X_rr_{split}.npy", mmap_mode="r")
        y = np.load(d / f"y_{split}.npy")
        if X.shape[0] != expect_n:
            die(EXIT_PREFLIGHT,
                f"{split}: expected {expect_n} beats, found {X.shape[0]}. The "
                f"preprocessing differs from the published arms, so results would "
                f"not be comparable. Re-run src/process_data.py unmodified.")
        if X.shape[1] != cfg["data"]["beat_length"]:
            die(EXIT_PREFLIGHT,
                f"{split}: beat length {X.shape[1]} != {cfg['data']['beat_length']}")
        if Xrr.shape[1] != cfg["data"]["rr_length"]:
            die(EXIT_PREFLIGHT,
                f"{split}: RR vector length {Xrr.shape[1]} != {cfg['data']['rr_length']}")
        if not (X.shape[0] == Xrr.shape[0] == y.shape[0]):
            die(EXIT_PREFLIGHT, f"{split}: X/X_rr/y row counts disagree")

        counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        expect_counts = {int(k): int(v)
                         for k, v in cfg["data"]["expect_class_counts"][split].items()}
        for cls, n in expect_counts.items():
            if counts.get(cls, 0) != n:
                die(EXIT_PREFLIGHT,
                    f"{split}: class {cls} has {counts.get(cls, 0)} beats, expected {n}. "
                    f"The AAMI mapping or split differs from the published arms.")
        say(f"  {split:5s}: {X.shape[0]:6d} beats  dist={counts}  OK")
        out[split] = {"n": int(X.shape[0]), "class_counts": counts}
    return out


def check_models(cfg) -> dict:
    head("PREFLIGHT 3/6 -- model construction and parameter counts")
    import torch
    from models.wavkan_v2 import WavKAN_v2

    def build(flags):
        kw = dict(use_pcwi=True, use_pwam=True, use_rr_attn=True,
                  wavelet_type=cfg["training"]["wavelet"])
        i = 0
        while i < len(flags):
            f = flags[i]
            if f == "--no-pcwi":
                kw["use_pcwi"] = False
            elif f == "--no-pwam":
                kw["use_pwam"] = False
            elif f == "--no-rr-attn":
                kw["use_rr_attn"] = False
            elif f == "--wavelet":
                kw["wavelet_type"] = flags[i + 1]
                i += 1
            i += 1
        return WavKAN_v2(**kw), kw

    out = {}
    for arm in selected_arms(cfg):
        m, kw = build(arm["flags"])
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        exp = arm.get("expect_params")
        flag = "OK" if (exp is None or n == exp) else f"MISMATCH (expected {exp})"
        say(f"  {arm['name']:16s} {n:>8,d} params  {kw}  {flag}")
        if exp is not None and n != exp:
            die(EXIT_PREFLIGHT,
                f"arm '{arm['name']}' builds with {n} parameters but the config "
                f"expects {exp}. Either the config or models/wavkan_v2.py changed; "
                f"resolve before training.")
        # a forward pass proves the graph runs before we commit GPU hours
        with torch.no_grad():
            y = m(torch.zeros(2, cfg["data"]["beat_length"]),
                  torch.zeros(2, cfg["data"]["rr_length"]))
        if tuple(y.shape) != (2, cfg["data"]["n_classes"]):
            die(EXIT_PREFLIGHT,
                f"arm '{arm['name']}' forward pass returned {tuple(y.shape)}, "
                f"expected (2, {cfg['data']['n_classes']})")
        out[arm["name"]] = {"n_params": int(n), "config": kw}
    say("  forward pass OK for every arm")
    return out


def check_reference_arm(cfg) -> dict:
    head("PREFLIGHT 4/6 -- reference arm (read-only)")
    ref = cfg["experiment"]["reference_arm"]
    d = REPO / ref["dir"]
    if not d.is_dir():
        die(EXIT_PREFLIGHT,
            f"reference arm directory not found: {ref['dir']}\n"
            f"       This job's whole purpose is to compare against it. Copy it to "
            f"this machine (it is git-tracked) before running.")
    seeds, params = {}, set()
    for s in cfg["seeds"]:
        f = d / f"seed_{s}" / "test_metrics.json"
        if not f.exists():
            die(EXIT_PREFLIGHT,
                f"reference arm is missing seed {s} ({f}). Paired statistics need "
                f"the same 20 seed identities, not merely the same count.")
        j = json.load(open(f))
        seeds[str(s)] = j
        if j.get("model_params"):
            params.add(j["model_params"])
    if ref.get("expect_params") and params != {ref["expect_params"]}:
        die(EXIT_PREFLIGHT,
            f"reference arm parameter counts {params} != expected "
            f"{ref['expect_params']}")
    say(f"  {ref['dir']}: {len(seeds)}/{len(cfg['seeds'])} seeds, "
        f"params={params.pop() if params else 'n/a'}  OK")
    say("  this directory is never written to by this job")
    return {"n_seeds": len(seeds), "dir": ref["dir"]}


def check_output(cfg, allow_resume: bool) -> Path:
    head("PREFLIGHT 5/6 -- output directory")
    root = REPO / cfg["experiment"]["output_root"]
    if root.exists():
        done = sorted(p.name for p in root.glob("*/seed_*/test_metrics.json"))
        if done and not allow_resume:
            die(EXIT_PREFLIGHT,
                f"{root} already contains completed runs.\n"
                f"       Refusing to overwrite an earlier run. Either move it aside, "
                f"or pass --allow-resume to skip the seeds that already finished.")
        if done:
            say(f"  {root} exists; --allow-resume given, finished seeds will be skipped")
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:
        die(EXIT_PREFLIGHT, f"cannot write to {root}: {e}")
    free_gb = shutil.disk_usage(root).free / 1024 ** 3
    say(f"  {root} writable, {free_gb:.1f} GB free")
    if free_gb < 2:
        die(EXIT_PREFLIGHT, f"only {free_gb:.1f} GB free; need >= 2 GB for checkpoints")
    return root


def check_trainer(cfg) -> None:
    head("PREFLIGHT 6/6 -- trainer entry point")
    t = REPO / "src" / "train_pca.py"
    if not t.exists():
        die(EXIT_PREFLIGHT, "src/train_pca.py not found")
    helptext = subprocess.run([sys.executable, str(t), "--help"],
                              cwd=REPO, capture_output=True, text=True).stdout
    need = ["--no-pcwi", "--no-pwam", "--no-rr-attn", "--seed", "--output-dir",
            "--data-dir", "--epochs", "--patience", "--batch-size", "--lr"]
    missing = [f for f in need if f not in helptext]
    if missing:
        die(EXIT_PREFLIGHT,
            f"src/train_pca.py does not expose: {', '.join(missing)}. "
            f"The config drives these flags; resolve before training.")
    say("  src/train_pca.py exposes every flag this config needs  OK")


# ───────────────────────────────────────────────────────────────── training
def selected_arms(cfg):
    arms = [a for a in cfg["arms"] if a.get("enabled", True)]
    if not _INCLUDE_OPTIONAL:
        arms = [a for a in arms if a.get("priority") != "optional"]
    return arms


def train_all(cfg, root: Path) -> dict:
    head("TRAIN")
    tr = cfg["training"]
    summary = {}
    total = len(selected_arms(cfg)) * len(cfg["seeds"])
    done_n = 0
    t_start = time.time()

    for arm in selected_arms(cfg):
        arm_dir = root / arm["name"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        say()
        say(f"--- arm '{arm['name']}'  flags={' '.join(arm['flags'])} ---")
        for seed in cfg["seeds"]:
            done_n += 1
            out_dir = arm_dir / f"seed_{seed}"
            marker = out_dir / "test_metrics.json"
            if marker.exists():
                say(f"  [{done_n}/{total}] seed {seed}: already complete, skipping")
                continue
            cmd = [sys.executable, str(REPO / "src" / "train_pca.py"),
                   "--seed", str(seed),
                   "--epochs", str(tr["epochs"]),
                   "--patience", str(tr["patience"]),
                   "--batch-size", str(tr["batch_size"]),
                   "--lr", str(tr["lr"]),
                   "--warmup", str(tr["warmup"]),
                   "--focal-gamma", str(tr["focal_gamma"]),
                   "--s-weight", str(tr["s_weight"]),
                   "--data-dir", cfg["data"]["dir"],
                   "--output-dir", str(out_dir)]
            if "--wavelet" not in arm["flags"]:
                cmd += ["--wavelet", tr["wavelet"]]
            if not tr.get("use_curriculum", True):
                cmd.append("--no-curriculum")
            if not tr.get("use_augment", True):
                cmd.append("--no-augment")
            cmd += arm["flags"]

            say(f"  [{done_n}/{total}] seed {seed}: training ...")
            log = out_dir.parent / f"seed_{seed}.log"
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(log, "w", encoding="utf-8") as lf:
                lf.write(" ".join(cmd) + "\n\n")
                lf.flush()
                rc = subprocess.run(cmd, cwd=REPO, stdout=lf,
                                    stderr=subprocess.STDOUT).returncode
            dt = time.time() - t0
            if rc != 0 or not marker.exists():
                # One seed failing must not lose the other 39.
                say(f"      FAILED (rc={rc}, {dt/60:.1f} min) -- see {log}")
                summary.setdefault("failures", []).append(
                    {"arm": arm["name"], "seed": seed, "returncode": rc,
                     "log": str(log.relative_to(REPO))})
            else:
                m = json.load(open(marker))
                say(f"      done in {dt/60:.1f} min  "
                    f"macro_f1={m.get('macro_f1', float('nan')):.4f}  "
                    f"epochs={m.get('epochs_run')}")
    say()
    say(f"training wall-clock: {(time.time()-t_start)/3600:.2f} h")
    if summary.get("failures"):
        say(f"WARNING: {len(summary['failures'])} run(s) failed; "
            f"postflight will report the gaps.")
    return summary


# ───────────────────────────────────────────────────────────────── postflight
def postflight(cfg, root: Path) -> dict:
    head("POSTFLIGHT -- completeness and sanity")
    import numpy as np

    metrics = cfg["statistics"]["metrics"]
    report, ok = {}, True
    for arm in selected_arms(cfg):
        arm_dir = root / arm["name"]
        found, missing, nan_hits, params = {}, [], [], set()
        for s in cfg["seeds"]:
            f = arm_dir / f"seed_{s}" / "test_metrics.json"
            if not f.exists():
                missing.append(s)
                continue
            j = json.load(open(f))
            for m in metrics:
                v = j.get(m)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    nan_hits.append((s, m, v))
            if not (arm_dir / f"seed_{s}" / "best_model.pth").exists():
                nan_hits.append((s, "best_model.pth", "missing"))
            params.add(j.get("model_params"))
            found[str(s)] = j
        say(f"  {arm['name']:16s} {len(found)}/{len(cfg['seeds'])} seeds"
            + (f"  MISSING={missing}" if missing else "")
            + (f"  BAD={nan_hits}" if nan_hits else "")
            + f"  params={params}")
        if missing or nan_hits:
            ok = False
        if arm.get("expect_params") and params - {None} not in ({arm["expect_params"]}, set()):
            say(f"      parameter-count mismatch: {params} != {arm['expect_params']}")
            ok = False
        report[arm["name"]] = {"n_seeds": len(found), "missing": missing,
                               "non_finite": nan_hits,
                               "params": sorted(x for x in params if x)}

    # paired-seed alignment against the reference arm
    ref_dir = REPO / cfg["experiment"]["reference_arm"]["dir"]
    ref_seeds = {p.name.replace("seed_", "")
                 for p in ref_dir.glob("seed_*") if (p / "test_metrics.json").exists()}
    for name, r in report.items():
        arm_seeds = {p.name.replace("seed_", "")
                     for p in (root / name).glob("seed_*")
                     if (p / "test_metrics.json").exists()}
        common = sorted(ref_seeds & arm_seeds, key=int)
        r["paired_with_reference"] = len(common)
        say(f"  {name:16s} pairs with reference on {len(common)} seeds")
        if len(common) < len(cfg["seeds"]):
            say(f"      NOTE only {len(common)} of {len(cfg['seeds'])} seeds pair; "
                f"statistics will use those and say so.")
    if not ok:
        say()
        say("POSTFLIGHT FOUND GAPS. Statistics will still be computed over the "
            "seeds that completed, and the gaps are recorded in results.json. "
            "Re-run with --allow-resume to fill them.")
    return report


# ───────────────────────────────────────────────────────────────── statistics
def compute_stats(cfg, root: Path) -> dict:
    head("STATISTICS -- paired vs the reference arm")
    from src.paired_stats import describe, holm_family, paired_compare

    ref_dir = REPO / cfg["experiment"]["reference_arm"]["dir"]
    metrics = cfg["statistics"]["metrics"]

    def load(d: Path, metric: str) -> dict:
        out = {}
        for f in d.glob("seed_*/test_metrics.json"):
            v = json.load(open(f)).get(metric)
            if v is not None:
                out[f.parent.name.replace("seed_", "")] = float(v)
        return out

    results = {"reference_arm": {}, "arms": {}}
    for m in metrics:
        results["reference_arm"][m] = describe(load(ref_dir, m).values())

    for arm in selected_arms(cfg):
        arm_dir = root / arm["name"]
        fam = {}
        for m in metrics:
            a, b = load(ref_dir, m), load(arm_dir, m)
            if not b:
                continue
            # sign convention from the config: reference minus variant, so a
            # positive d favours KEEPING the ablated component.
            fam[m] = paired_compare(a, b, metric_name=m, ci=cfg["statistics"]["ci"])
        if not fam:
            say(f"  {arm['name']}: no completed seeds, skipping")
            continue
        fam = holm_family(fam)
        results["arms"][arm["name"]] = {
            "flags": arm["flags"],
            "priority": arm.get("priority"),
            "protects_claim": arm.get("protects_claim"),
            "descriptives": {m: describe(load(arm_dir, m).values()) for m in metrics},
            "vs_reference": fam,
        }
        say()
        say(f"  arm '{arm['name']}'  (positive d favours keeping the component)")
        say(f"    {'metric':<10} {'reference':>17} {'variant':>17} "
            f"{'diff':>9} {'95% CI':>20} {'d':>7} {'Holm p':>10}")
        for m, r in fam.items():
            say(f"    {m:<10} {r['mean_a']:>9.4f}+-{r['std_a']:<6.4f} "
                f"{r['mean_b']:>9.4f}+-{r['std_b']:<6.4f} "
                f"{r['mean_diff']:>+9.4f} "
                f"[{r['ci_low']:>+8.4f},{r['ci_high']:>+8.4f}] "
                f"{r['cohens_d']:>+7.2f} {r['holm_p']:>10.2e}"
                + ("  *" if r["significant_after_holm"] else ""))
    return results


def make_plot(cfg, root: Path, stats: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        say(f"  (plot skipped: {e})")
        return
    metrics = cfg["statistics"]["metrics"]
    arms = list(stats["arms"])
    if not arms:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        labels = ["reference"] + arms
        means = [stats["reference_arm"][m]["mean"]] + \
                [stats["arms"][a]["descriptives"][m]["mean"] for a in arms]
        errs = [stats["reference_arm"][m]["std"]] + \
               [stats["arms"][a]["descriptives"][m]["std"] for a in arms]
        ax.bar(range(len(means)), means, yerr=errs, capsize=3,
               color=["#2166ac"] + ["#b2182b"] * len(arms), alpha=0.85)
        for i, a in enumerate(arms, start=1):
            r = stats["arms"][a]["vs_reference"].get(m)
            if r and r["significant_after_holm"]:
                ax.text(i, means[i] + errs[i], "*", ha="center", va="bottom",
                        fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
        ax.set_title(m, fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Component ablation on the adopted architecture "
                 "(20 seeds; * = Holm-significant vs reference)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = root / "component_ablation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    say(f"  plot: {out.relative_to(REPO)}")


# ─────────────────────────────────────────────────────────────────── reporting
def write_outputs(cfg, root: Path, env, data, models, ref, post, stats, failures):
    head("REPORT")
    started = _T_START.isoformat()
    finished = datetime.now(timezone.utc).isoformat()

    provenance = {
        "experiment": cfg["experiment"]["name"],
        "phase": cfg["experiment"].get("phase"),
        "started_utc": started,
        "finished_utc": finished,
        "environment": env,
        "data_verified": data,
        "models_verified": models,
        "reference_arm": ref,
        "config_snapshot": cfg,
        "seed_list": cfg["seeds"],
        "arms_run": [a["name"] for a in selected_arms(cfg)],
        "failures": failures.get("failures", []),
        "postflight": post,
    }
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    (root / "results.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8")
    shutil.copy2(_CONFIG_PATH, root / "config_used.yaml")

    # flat CSV: one row per arm x metric, for spreadsheet-level inspection
    with open(root / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "metric", "n_paired", "reference_mean", "reference_std",
                    "variant_mean", "variant_std", "mean_diff", "ci_low", "ci_high",
                    "cohens_d", "effect_size", "p_raw", "holm_p", "family_size",
                    "significant_after_holm"])
        for name, a in stats["arms"].items():
            for m, r in a["vs_reference"].items():
                w.writerow([name, m, r["n_paired"],
                            f"{r['mean_a']:.6f}", f"{r['std_a']:.6f}",
                            f"{r['mean_b']:.6f}", f"{r['std_b']:.6f}",
                            f"{r['mean_diff']:.6f}",
                            f"{r['ci_low']:.6f}", f"{r['ci_high']:.6f}",
                            f"{r['cohens_d']:.4f}", r["effect_size"],
                            f"{r['p_value']:.6g}", f"{r['holm_p']:.6g}",
                            r["family_size"], r["significant_after_holm"]])

    # per-seed CSV: the raw material for any re-analysis
    with open(root / "per_seed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed"] + cfg["statistics"]["metrics"] +
                   ["epochs_run", "model_params"])
        for arm in selected_arms(cfg):
            for s in cfg["seeds"]:
                p = root / arm["name"] / f"seed_{s}" / "test_metrics.json"
                if not p.exists():
                    continue
                j = json.load(open(p))
                w.writerow([arm["name"], s] +
                           [j.get(m) for m in cfg["statistics"]["metrics"]] +
                           [j.get("epochs_run"), j.get("model_params")])

    make_plot(cfg, root, stats)
    write_readme(cfg, root, env, stats, post, failures)

    say(f"  provenance.json  results.json  results.csv  per_seed.csv  "
        f"RESULTS_README.md  config_used.yaml")
    say(f"  all under {root.relative_to(REPO)}/")


def write_readme(cfg, root: Path, env, stats, post, failures):
    lines = []
    A = lines.append
    A(f"# {cfg['experiment']['name']} -- results")
    A("")
    A("## What was run and why")
    A("")
    A("Component ablation on the **adopted** PC-WavKAN architecture "
      "(`use_rr_attn=False`, 153,045 params).")
    A("")
    A("The manuscript's central architectural claim is that physiology-constrained")
    A("wavelet initialisation (PCWI) is the component that measurably helps. Before")
    A("this run that claim rested on a table whose every row varies one component of")
    A("the **base** configuration (`use_rr_attn=True`) -- the configuration the paper")
    A("does not adopt. No PCWI-off run existed for the adopted architecture. This job")
    A("closes that gap so the primary contribution is measured on the model actually")
    A("proposed.")
    A("")
    A("## Exact command")
    A("")
    A("```bash")
    A("bash run_remote_experiment.sh")
    A("#   or, equivalently:")
    A(f"python run_remote_experiment.py --config {Path(_CONFIG_PATH).as_posix()}")
    A("```")
    A("")
    A("Restart after an interruption (skips finished seeds, never overwrites):")
    A("")
    A("```bash")
    A(f"python run_remote_experiment.py --config {Path(_CONFIG_PATH).as_posix()} --allow-resume")
    A("```")
    A("")
    A("## Configuration")
    A("")
    A("Verbatim snapshot in `config_used.yaml`. Key values, all copied from the")
    A("reference arm so each comparison is single-variable:")
    A("")
    tr = cfg["training"]
    A(f"| setting | value |")
    A(f"|---|---|")
    for k in ("epochs", "patience", "batch_size", "lr", "weight_decay", "optimizer",
              "scheduler", "grad_clip_norm", "warmup", "focal_gamma", "s_weight",
              "wavelet", "use_curriculum", "use_augment", "checkpoint_rule"):
        if k in tr:
            A(f"| {k} | `{tr[k]}` |")
    A("")
    A(f"**Seeds ({len(cfg['seeds'])})**: `{cfg['seeds']}`")
    A("")
    A("These are the exact seed identities of every published arm. Paired statistics")
    A("require the identities, not merely the count.")
    A("")
    A("**Arms**")
    A("")
    A("| arm | flags | priority |")
    A("|---|---|---|")
    for a in selected_arms(cfg):
        A(f"| `{a['name']}` | `{' '.join(a['flags'])}` | {a.get('priority')} |")
    A("")
    A(f"**Reference arm** (not retrained, read-only): "
      f"`{cfg['experiment']['reference_arm']['dir']}`")
    A("")
    A("## Files")
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `results.json` | aggregate + full paired statistics, machine-readable |")
    A("| `results.csv` | one row per arm x metric |")
    A("| `per_seed.csv` | one row per arm x seed -- the raw material for re-analysis |")
    A("| `provenance.json` | environment, GPU, git revision, data integrity, config |")
    A("| `config_used.yaml` | verbatim config snapshot |")
    A("| `component_ablation.pdf` | per-metric bar chart with significance markers |")
    A("| `<arm>/seed_<N>/test_metrics.json` | per-seed metrics |")
    A("| `<arm>/seed_<N>/best_model.pth` | checkpoint |")
    A("| `<arm>/seed_<N>/training_history.json` | per-epoch validation curve |")
    A("| `<arm>/seed_<N>.log` | full training transcript |")
    A("| `run.log` | this run's console transcript |")
    A("")
    A("## Results")
    A("")
    A("Sign convention: **positive Cohen's _d_ favours the reference arm**, i.e.")
    A("favours *keeping* the ablated component. `*` marks Holm-significant within")
    A("that arm's five-metric family.")
    A("")
    for name, a in stats.get("arms", {}).items():
        A(f"### `{name}`  (`{' '.join(a['flags'])}`)")
        A("")
        A("| metric | reference | variant | diff | 95% CI | _d_ | Holm _p_ | sig |")
        A("|---|---|---|---|---|---|---|---|")
        for m, r in a["vs_reference"].items():
            A(f"| {m} | {r['mean_a']:.4f}±{r['std_a']:.4f} | "
              f"{r['mean_b']:.4f}±{r['std_b']:.4f} | {r['mean_diff']:+.4f} | "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['cohens_d']:+.2f} | "
              f"{r['holm_p']:.2e} | {'*' if r['significant_after_holm'] else ''} |")
        A("")
        A(f"_n_ paired = {list(a['vs_reference'].values())[0]['n_paired']}")
        A("")
    if failures.get("failures"):
        A("## Failures")
        A("")
        for f in failures["failures"]:
            A(f"- arm `{f['arm']}` seed {f['seed']}: rc={f['returncode']}, see `{f['log']}`")
        A("")
        A("Re-run with `--allow-resume` to fill these in.")
        A("")
    A("## Provenance")
    A("")
    A(f"- host: `{env.get('hostname')}`")
    A(f"- GPU: `{env.get('gpu_name')}` ({env.get('gpu_memory_gb')} GB)")
    A(f"- torch `{env.get('torch')}` / CUDA `{env.get('cuda')}` / "
      f"python `{env.get('python')}`")
    g = env.get("git", {})
    A(f"- git: `{g.get('commit','')[:12]}` on `{g.get('branch')}`"
      f"{' (DIRTY)' if g.get('dirty') else ''}")
    A("")
    A("## Reproducing the statistics locally")
    A("")
    A("Copy this whole directory into the local repo at the same path, then:")
    A("")
    A("```bash")
    A("# 1. verify the artifacts and recompute every statistic from per-seed files")
    A(f"python src/integrate_remote_results.py --dir {cfg['experiment']['output_root']}")
    A("")
    A("# 2. re-verify every number the manuscript reports")
    A("python src/verify_manuscript_numbers.py")
    A("```")
    A("")
    A("`integrate_remote_results.py` recomputes the paired statistics locally from")
    A("`per_seed.csv` and the per-seed JSONs and diffs them against `results.json`,")
    A("so a transport error or a version skew shows up as a mismatch rather than")
    A("being trusted. It does not write to the manuscript: it prints the exact")
    A("numbers to use, and the manuscript is updated only after the diff is clean.")
    (root / "RESULTS_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────── main
_T_START = datetime.now(timezone.utc)
_CONFIG_PATH = ""
_INCLUDE_OPTIONAL = False


def main():
    global _CONFIG_PATH, _INCLUDE_OPTIONAL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/final_component_ablation.yaml")
    ap.add_argument("--allow-resume", action="store_true",
                    help="skip seeds that already completed instead of refusing to run")
    ap.add_argument("--include-optional", action="store_true",
                    help="also train arms marked priority: optional")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit running without CUDA (not recommended)")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run every check and exit without training")
    args = ap.parse_args()

    _CONFIG_PATH = args.config
    _INCLUDE_OPTIONAL = args.include_optional

    cfg = load_config(Path(args.config))
    root = REPO / cfg["experiment"]["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(root / "run.log")

    say("=" * 78)
    say(f"{cfg['experiment']['name']}  --  started {_T_START.isoformat()}")
    say(f"config: {args.config}")
    say("=" * 78)

    if args.allow_cpu:
        import torch
        if not torch.cuda.is_available():
            say("\n  --allow-cpu given and no CUDA present: continuing on CPU.\n")
            _patch_cpu()

    env = check_environment(cfg)
    data = check_data(cfg)
    models = check_models(cfg)
    ref = check_reference_arm(cfg)
    check_output(cfg, args.allow_resume)
    check_trainer(cfg)

    say()
    say("PREFLIGHT PASSED -- every invariant holds, safe to train")
    n = len(selected_arms(cfg)) * len(cfg["seeds"])
    say(f"  {len(selected_arms(cfg))} arm(s) x {len(cfg['seeds'])} seeds = {n} runs")
    say(f"  reference arm: {cfg['experiment']['reference_arm']['dir']} (read-only)")

    if args.preflight_only:
        say("\n--preflight-only given: exiting before training.")
        return 0

    failures = train_all(cfg, root)
    post = postflight(cfg, root)
    stats = compute_stats(cfg, root)
    write_outputs(cfg, root, env, data, models, ref, post, stats, failures)

    head("DONE")
    say(f"Copy this directory back to the local repo:")
    say(f"    {cfg['experiment']['output_root']}/")
    say(f"Then run:")
    say(f"    python src/integrate_remote_results.py "
        f"--dir {cfg['experiment']['output_root']}")
    say(f"See {cfg['experiment']['output_root']}/RESULTS_README.md")
    return 0


def _patch_cpu():
    """Allow the CUDA gate to pass when --allow-cpu is given."""
    import torch
    torch.cuda.is_available = lambda: True          # noqa: E731
    torch.cuda.get_device_name = lambda i=0: "CPU (--allow-cpu)"

    class _P:
        total_memory = 8 * 1024 ** 3
    torch.cuda.get_device_properties = lambda i=0: _P()
    torch.version.cuda = "n/a (CPU)"


if __name__ == "__main__":
    sys.exit(main())
