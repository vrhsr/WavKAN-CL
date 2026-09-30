"""
run_inference_jobs.py -- run eval_inference_sensitivity.py as many single-thread
(model, seed) jobs in parallel, then merge each dataset's parts into the report.

Added 2026-09-30. Forward pass only; resumable (finished parts are skipped).
The merged per_seed.json/report.json are identical to the serial path's
(verified on a DS2 slice before use).

Usage:
  python src/run_inference_jobs.py --workers 6 \
      --job data/incart_matched:results/external_matched/incart \
      --job data/svdb_matched:results/external_matched/svdb
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval_inference_sensitivity import MODELS, SEEDS  # noqa: E402

# Relative single-thread cost per beat, measured on this machine; used only to
# schedule the longest jobs first.
COST = {"Transformer": 10.0, "ResNet1D": 2.3, "B-Spline KAN": 1.4, "PC-WavKAN": 1.2, "CNN+Focal": 1.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True, help="DATA_DIR:OUT_DIR")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    specs = [j.split(":", 1) for j in a.job]
    jobs = [(dd, od, m, s) for dd, od in specs for m in MODELS for s in SEEDS]
    jobs.sort(key=lambda j: -COST[j[2]])

    def run(job):
        dd, od, m, s = job
        cmd = [sys.executable, "src/eval_inference_sensitivity.py", "--data-dir", dd, "--out", od,
               "--models", m, "--seeds", str(s), "--threads", "1", "--parts-dir", os.path.join(od, "parts")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return job, r.returncode, r.stderr[-400:]

    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            job, rc, err = fut.result()
            done += 1
            if rc != 0:
                failed.append((job, err))
            if done % 10 == 0 or rc != 0:
                print(f"[{time.time() - t0:7.0f}s] {done}/{len(jobs)} jobs done, {len(failed)} failed"
                      + (f"  FAILED {job}: {err}" if rc != 0 else ""), flush=True)
    if failed:
        print(f"{len(failed)} jobs failed; not merging", flush=True)
        sys.exit(1)
    for dd, od in specs:
        subprocess.run([sys.executable, "src/eval_inference_sensitivity.py", "--data-dir", dd, "--out", od,
                        "--parts-dir", os.path.join(od, "parts"), "--merge-parts"], check=True)
        print(f"merged {od}", flush=True)
    print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
