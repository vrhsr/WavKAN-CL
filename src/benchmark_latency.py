"""
WavKAN-CL Inference Latency & Efficiency Benchmark
====================================================
Reports:
  - Inference latency (ms/beat) on CPU (worst-case for edge deployment)
  - Model file size in KB
  - Parameter count
  - GFLOPs / MACs (via thop, optional)

Usage:
    python src/benchmark_latency.py
    python src/benchmark_latency.py --n-beats 5000 --warmup 200

Output:
    Prints table to stdout + saves results/latency_report.json
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HybridWavKAN_RR(nn.Module):
    def __init__(self, input_size=360, num_classes=5):
        super().__init__()
        self.kan    = WavKANLinear(input_size, 64, wavelet_type='mexican_hat')
        self.ln     = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.0)          # disabled for latency eval
        self.bigru  = nn.GRU(64, 32, 1, batch_first=True, bidirectional=True)
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, num_classes)

    def forward(self, x, xr):
        x  = self.kan(x)
        x  = self.ln(x)
        x  = x.unsqueeze(1)
        x, _ = self.bigru(x)
        x  = x.squeeze(1)
        xr = self.rr_mlp(xr)
        x  = torch.cat((x, xr), dim=1)
        return self.fc2(torch.relu(self.fc1(x)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_model_size_kb(model):
    """Estimate model size in KB via state_dict."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024.0


def measure_latency(model, n_beats, warmup, device):
    """Measure per-beat inference latency in ms (CPU, single-beat mode)."""
    model.eval()
    model.to(device)

    x_single  = torch.randn(1, 360).to(device)
    xr_single = torch.randn(1, 5).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_single, xr_single)

    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    latencies = []
    with torch.no_grad():
        for _ in range(n_beats):
            t0 = time.perf_counter()
            _ = model(x_single, xr_single)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms":   float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms":    float(np.std(latencies)),
        "p95_ms":    float(np.percentile(latencies, 95)),
        "min_ms":    float(np.min(latencies)),
        "max_ms":    float(np.max(latencies)),
    }


def measure_batch_throughput(model, batch_size, n_batches, device):
    """Measure throughput in beats/second for batch processing."""
    model.eval()
    x  = torch.randn(batch_size, 360).to(device)
    xr = torch.randn(batch_size, 5).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x, xr)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_batches):
            _ = model(x, xr)
    t1 = time.perf_counter()

    total_beats = batch_size * n_batches
    elapsed_sec = t1 - t0
    return float(total_beats / elapsed_sec)


def try_compute_macs(model, device):
    """Use thop (pip install thop) to compute MACs and params."""
    try:
        from thop import profile
        x  = torch.randn(1, 360).to(device)
        xr = torch.randn(1, 5).to(device)
        macs, params = profile(model, inputs=(x, xr), verbose=False)
        return {"macs": float(macs), "macs_M": float(macs / 1e6),
                "thop_params": float(params)}
    except ImportError:
        return {"note": "Install thop for MACs: pip install thop"}
    except Exception as e:
        return {"error": str(e)}


def run_benchmark(n_beats=2000, warmup=100, batch_size=64, n_batches=100,
                  model_path=None, out_path="results/latency_report.json"):
    device = torch.device("cpu")   # CPU = worst-case edge scenario

    model = HybridWavKAN_RR()

    # Load checkpoint if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"✅ Loaded checkpoint: {model_path}")
    else:
        print("ℹ️  No checkpoint loaded — using random weights (structure benchmark only)")

    model.eval()

    # Core metrics
    n_params   = count_parameters(model)
    size_kb    = measure_model_size_kb(model)
    macs_info  = try_compute_macs(model, device)

    print("\n" + "="*55)
    print("  WavKAN-CL Efficiency Benchmark (CPU)")
    print("="*55)
    print(f"  Parameters:       {n_params:,}")
    print(f"  Model Size:       {size_kb:.1f} KB")
    if "macs_M" in macs_info:
        print(f"  MACs:             {macs_info['macs_M']:.3f} M")

    print(f"\n  Benchmarking single-beat latency ({n_beats} runs)...")
    latency = measure_latency(model, n_beats, warmup, device)

    print(f"  Mean latency:     {latency['mean_ms']:.3f} ms/beat")
    print(f"  Median latency:   {latency['median_ms']:.3f} ms/beat")
    print(f"  P95 latency:      {latency['p95_ms']:.3f} ms/beat")
    print(f"  Std deviation:    {latency['std_ms']:.3f} ms")

    throughput = measure_batch_throughput(model, batch_size, n_batches, device)
    print(f"\n  Batch throughput: {throughput:.0f} beats/sec  "
          f"(batch_size={batch_size})")
    print("="*55)

    report = {
        "device":          str(device),
        "model_path":      model_path,
        "n_parameters":    n_params,
        "size_kb":         round(size_kb, 2),
        "latency":         latency,
        "throughput_bps":  round(throughput, 1),
        "batch_size":      batch_size,
        "macs_info":       macs_info,
    }

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n✅ Report saved to: {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WavKAN-CL Latency Benchmark")
    parser.add_argument("--n-beats",    type=int,   default=2000,
                        help="Number of single-beat inferences to time")
    parser.add_argument("--warmup",     type=int,   default=100,
                        help="Number of warmup runs before timing")
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--n-batches",  type=int,   default=100)
    parser.add_argument("--model-path", type=str,
                        default="results/hybrid_rr_history_20_seeds/seed_42/best_hybrid_rr.pth")
    parser.add_argument("--out",        type=str,
                        default="results/latency_report.json")
    args = parser.parse_args()

    run_benchmark(
        n_beats    = args.n_beats,
        warmup     = args.warmup,
        batch_size = args.batch_size,
        n_batches  = args.n_batches,
        model_path = args.model_path,
        out_path   = args.out,
    )
