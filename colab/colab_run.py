"""
colab_run.py -- run the final-audit inference-only evaluations on a Colab GPU.

Forward passes of the published checkpoints only: no training, no tuning.
Produces the same files the local CPU path would (results/rr_sensitivity/* and
results/external_matched/*), plus an input-fingerprint check against the arrays
extracted locally and an environment record. Everything is zipped into
final_audit_colab_results.zip for download.

Run from the unzipped bundle root:  python colab/colab_run.py
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = "https://physionet-open.s3.amazonaws.com"
PY = sys.executable
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def sh(args):
    log("$ " + " ".join(args))
    subprocess.run(args, check=True)


# ---------------------------------------------------------------- 1. data
def records(db):
    with urllib.request.urlopen(f"{BASE}/{db}/1.0.0/RECORDS") as r:
        return [l.strip() for l in r.read().decode().splitlines() if l.strip()]


sys.path.insert(0, "src")
from split import TEST_RECORDS  # noqa: E402

jobs = []
for db, out, keep in [("mitdb", "data/raw", set(TEST_RECORDS)),
                      ("incartdb", "data/raw_incart", None),
                      ("svdb", "data/raw_svdb", None)]:
    os.makedirs(out, exist_ok=True)
    for rec in records(db):
        if keep is not None and rec not in keep:
            continue
        for ext in (".hea", ".dat", ".atr"):
            jobs.append((f"{BASE}/{db}/1.0.0/{rec}{ext}", os.path.join(out, rec + ext)))


def fetch(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    for _ in range(5):
        try:
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("download failed: " + url)


log(f"downloading {len(jobs)} files from the PhysioNet AWS mirror")
with ThreadPoolExecutor(16) as ex:
    list(ex.map(fetch, jobs))
log("download done")

# ---------------------------------------------------------------- 2. extraction
sh([PY, "src/extract_matched.py", "--dataset", "mitdb_test", "--out", "data/mitdb_test_published_rr"])
sh([PY, "src/extract_matched.py", "--dataset", "mitdb_test", "--rr-mode", "beats_only",
    "--out", "data/mitdb_test_beats_only_rr"])
sh([PY, "src/extract_matched.py", "--dataset", "incart", "--out", "data/incart_matched"])
sh([PY, "src/extract_matched.py", "--dataset", "svdb", "--out", "data/svdb_matched"])

# ---------------------------------------------------------------- 3. fingerprints
local = json.load(open("colab/local_fingerprints.json"))
check = {}
for ds, fp in local.items():
    d = f"data/{ds}/"
    X, R, y = np.load(d + "X_test.npy"), np.load(d + "X_rr_test.npy"), np.load(d + "y_test.npy")
    got = {"n": int(len(y)), "class_counts": np.bincount(y, minlength=5).tolist(),
           "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
           "rr_sha256": hashlib.sha256(R.tobytes()).hexdigest(),
           "X_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
           "X_round4_sha256": hashlib.sha256(np.round(X.astype(np.float64), 4).tobytes()).hexdigest()}
    check[ds] = {k: (got[k] == fp[k]) for k in got}
    log(f"fingerprint {ds}: " + ", ".join(f"{k}={'MATCH' if v else 'DIFFERS'}" for k, v in check[ds].items()))
os.makedirs("results", exist_ok=True)
json.dump(check, open("results/colab_fingerprint_check.json", "w"), indent=1)
for ds in local:
    if not (check[ds]["n"] and check[ds]["y_sha256"] and check[ds]["rr_sha256"]):
        raise SystemExit(f"{ds}: beat set, labels or RR features differ from the local extraction; stopping")

# ---------------------------------------------------------------- 4. evaluation
import torch  # noqa: E402
dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device: {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")
common = ["--device", dev, "--batch", "256"]
sh([PY, "src/eval_inference_sensitivity.py", "--data-dir", "data/mitdb_test_published_rr",
    "--out", "results/rr_sensitivity/published_rr", "--check-published"] + common)
sh([PY, "src/eval_inference_sensitivity.py", "--data-dir", "data/mitdb_test_published_rr",
    "--rr-override", "data/mitdb_test_beats_only_rr/X_rr_test.npy",
    "--out", "results/rr_sensitivity/beats_only_rr"] + common)
sh([PY, "src/eval_inference_sensitivity.py", "--data-dir", "data/incart_matched",
    "--out", "results/external_matched/incart"] + common)
sh([PY, "src/eval_inference_sensitivity.py", "--data-dir", "data/svdb_matched",
    "--out", "results/external_matched/svdb"] + common)

# ---------------------------------------------------------------- 5. environment + zip
import scipy  # noqa: E402
import neurokit2  # noqa: E402
import wfdb  # noqa: E402
env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "numpy": np.__version__,
       "scipy": scipy.__version__, "neurokit2": neurokit2.__version__, "wfdb": wfdb.__version__,
       "tf32_disabled": True, "batch": 256, "runtime_s": round(time.time() - T0)}
json.dump(env, open("results/colab_environment.json", "w"), indent=1)
sh(["zip", "-qr", "final_audit_colab_results.zip", "results/rr_sensitivity", "results/external_matched",
    "results/colab_fingerprint_check.json", "results/colab_environment.json"])
log("DONE -> final_audit_colab_results.zip")
