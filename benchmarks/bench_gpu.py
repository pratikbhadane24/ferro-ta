"""
GPU vs CPU benchmark for ferro_ta.gpu (SMA, EMA, RSI).

Requires:
    pip install "ferro-ta[gpu]"   # or pip install torch

Run:
    python benchmarks/bench_gpu.py

The script compares wall-clock time for 1M-element arrays and prints a
summary table.  If PyTorch is not installed or no GPU is found, GPU columns are skipped.
"""

from __future__ import annotations

import time

import numpy as np

# Try to import PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = None
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    DEVICE = None

from ferro_ta.gpu import ema, rsi, sma

N = 1_000_000
REPEATS = 10


def _time_fn(fn, *args, **kwargs) -> float:
    """Return minimum wall time (seconds) over REPEATS calls."""
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elif DEVICE == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    rng = np.random.default_rng(42)
    close_cpu = rng.uniform(100.0, 200.0, N)

    print(f"Array size: {N:,} elements")
    print(f"Repeats:    {REPEATS}")
    print(f"Device:     {DEVICE if DEVICE else 'CPU'}")
    print()

    header = f"{'Indicator':<20} {'CPU (ms)':>10}"
    if DEVICE:
        header += f" {'GPU (ms)':>10} {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for name, fn, kwargs in [
        ("sma(period=30)", sma, {"timeperiod": 30}),
        ("ema(period=30)", ema, {"timeperiod": 30}),
        ("rsi(period=14)", rsi, {"timeperiod": 14}),
    ]:
        cpu_time = _time_fn(fn, close_cpu, **kwargs) * 1000  # ms

        row = f"{name:<20} {cpu_time:>10.3f}"
        if DEVICE:
            dtype = torch.float32 if DEVICE == "mps" else torch.float64
            close_gpu = torch.tensor(close_cpu, dtype=dtype, device=DEVICE)
            # Warm-up
            fn(close_gpu, **kwargs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elif DEVICE == "mps":
                torch.mps.synchronize()
            gpu_time = _time_fn(fn, close_gpu, **kwargs) * 1000  # ms
            speedup = cpu_time / gpu_time
            row += f" {gpu_time:>10.3f} {speedup:>10.2f}×"
        print(row)

    if not TORCH_AVAILABLE:
        print()
        print("PyTorch not available — GPU columns skipped.")
        print("Install with: pip install 'ferro_ta[gpu]'")
    elif not DEVICE:
        print()
        print(
            "PyTorch found, but no CUDA or MPS device detected — GPU columns skipped."
        )


if __name__ == "__main__":
    main()
