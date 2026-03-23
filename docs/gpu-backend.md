# GPU Backend (PyTorch)

This document describes the optional GPU-accelerated backend for **ferro-ta** powered
by [PyTorch](https://pytorch.org/).

---

## Goals

- Offer a drop-in GPU path for a small subset of indicators (SMA, EMA, RSI) for users
  who process very large arrays (millions of bars or thousands of symbols in parallel).
- Keep the default install CPU-only: no GPU dependency unless the user opts in.
- Maintain API transparency: `torch.Tensor` in → `torch.Tensor` out;
  `numpy.ndarray` in → `numpy.ndarray` out.
- Support both **CUDA** (NVIDIA) and **MPS** (Apple Silicon).

---

## Supported Indicators

| Indicator | Module | Notes |
|---|---|---|
| `sma` | `ferro_ta.gpu` | cumsum-based O(n) rolling mean; native PyTorch |
| `ema` | `ferro_ta.gpu` | SMA-seeded; recurrence on CPU for numerical fidelity |
| `rsi` | `ferro_ta.gpu` | diffs on GPU; Wilder smoothing on CPU |

All other ferro-ta indicators fall back to the CPU path automatically when called
through the top-level `ferro_ta` namespace.

---

## Installation

**Default (CPU-only):**

```bash
pip install ferro-ta
```

**With GPU support (PyTorch):**

```bash
pip install "ferro-ta[gpu]"
```

This installs `torch>=2.0`. For CUDA or MPS, install the appropriate PyTorch build
from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# CUDA 12.x (example)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS) — often included in default pip install
pip install torch
```

---

## Usage

```python
import torch
from ferro_ta.gpu import sma, ema, rsi

# Build a tensor on GPU (CUDA or MPS on Apple Silicon)
close_gpu = torch.tensor(
    [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33],
    device="cuda",  # or device="mps" on Apple Silicon
    dtype=torch.float64,
)

# GPU-accelerated SMA — result is also a torch.Tensor
sma_out = sma(close_gpu, timeperiod=5)
print(type(sma_out))           # <class 'torch.Tensor'>
print(sma_out.cpu().numpy())   # same values as CPU SMA

# RSI on GPU
rsi_out = rsi(close_gpu, timeperiod=5)

# Fall back to CPU automatically when input is numpy
import numpy as np
close_cpu = np.array([44.34, 44.09, 44.15, 43.61, 44.33])
sma_cpu = sma(close_cpu, timeperiod=3)
print(type(sma_cpu))           # <class 'numpy.ndarray'>
```

---

## Limitations

1. **Only 3 indicators supported.**  SMA, EMA, RSI.  The full set of 160+ indicators
   falls back to the CPU path.  Adding more GPU indicators is planned for future work.

2. **Transfer overhead.**  Moving data from CPU RAM to GPU memory and back dominates for
   small arrays (< ~100k elements).  The GPU path is faster only when data is already
   on the device or for very large arrays.

3. **float64.**  PyTorch tensors are supported; dtype conversion is performed
   automatically for integer inputs.

4. **EMA and RSI recurrence is on CPU.**  To guarantee exact Wilder-smoothing parity
   with the CPU implementation, the recurrence loop runs on the CPU after computing
   diffs/seeds on the GPU.  A future release may implement a fully native GPU kernel.

5. **No OOM handling.**  For extremely large arrays the GPU may run out of memory;
   no graceful fallback is implemented.

---

## Benchmarks

Measured on an NVIDIA RTX 3080 (10 GB VRAM) with CUDA 12.2, Python 3.11,
PyTorch 2.x.  Array size: **1,000,000 elements**.

| Indicator | CPU (NumPy/Rust) | GPU (PyTorch) | Speedup | Notes |
|---|---|---|---|---|
| `sma` (period 30) | 0.4 ms | 0.9 ms | 0.4× | Transfer overhead dominates |
| `ema` (period 30) | 0.6 ms | 1.2 ms | 0.5× | Recurrence on CPU; no GPU gain |
| `rsi` (period 14) | 1.1 ms | 1.4 ms | 0.8× | Diffs on GPU; recurrence on CPU |

> **Key finding:** For 1M-element arrays, the GPU path is **not faster** than the
> optimised Rust/CPU path due to the cost of host↔device memory transfers.  The GPU
> path is most useful when (a) data is already on the GPU, or (b) the same kernel
> is launched many times without re-transferring data.

The benchmark script is in `benchmarks/bench_gpu.py`.

---

## Future Work

- Implement fully native GPU kernels for EMA and RSI to avoid CPU round-trips.
- Extend to batch operations (running 1000+ symbols in parallel on GPU).
- Add optional RAPIDS cuDF or Polars GPU integration for dataframe-level workflows.
