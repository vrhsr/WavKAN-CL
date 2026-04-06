import sys
import os
import psutil
import torch
import time
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.wavkan import WavKANClassifier

def measure_peak_ram(model_path):
    print("="*50)
    print(" PEAK RAM AND LATENCY BENCHMARK (G2, G1)")
    print("="*50)

    # 1. Baseline RAM
    process = psutil.Process(os.getpid())
    base_ram = process.memory_info().rss / (1024 * 1024)
    print(f"Base RAM usage: {base_ram:.2f} MB")

    # 2. Load Model
    device = torch.device('cpu')
    model = WavKANClassifier(wavelet_type='mexican_hat').to(device)
    # Using random weights since we just want memory footprints
    model.eval()
    
    loaded_ram = process.memory_info().rss / (1024 * 1024)
    print(f"RAM after model load: {loaded_ram:.2f} MB")
    print(f"Model footprint in RAM: {loaded_ram - base_ram:.2f} MB")

    # 3. Inference Simulation (Peak RAM spike check)
    dummy_input = torch.randn(64, 360).to(device)
    
    peak_ram = loaded_ram
    times = []
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)

    # Benchmark
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            
            current_ram = process.memory_info().rss / (1024 * 1024)
            if current_ram > peak_ram:
                peak_ram = current_ram

    mean_time_batch = np.mean(times) * 1000 # ms
    mean_time_beat = mean_time_batch / 64

    print(f"\nPeak RAM during inference: {peak_ram:.2f} MB")
    print(f"Active Memory Overhead: {peak_ram - loaded_ram:.2f} MB")
    print(f"Inference Latency: {mean_time_beat:.3f} ms/beat")
    print("="*50)

if __name__ == "__main__":
    measure_peak_ram("dummy")
