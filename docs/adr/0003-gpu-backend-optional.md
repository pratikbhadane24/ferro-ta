# ADR 0003: GPU backend (torch) is an optional extra

**Status:** Accepted
**Date:** 2026-04-14

## Context

ferro-ta's CPU path (Rust + runtime-dispatched SIMD via the `multiversion`
crate, with rayon for batch parallelism) is the primary performance story. A subset of batch
operations — notably large matrix DTW and vectorised Greek surfaces —
benefit from GPU acceleration, and ferro-ta includes a `torch`-backed
path for these.

PyTorch is a ~2 GB dependency with CUDA wheels. Making it a hard
requirement would balloon the install size and tie ferro-ta's release
cadence to PyTorch's.

## Decision

GPU support is gated behind the `[gpu]` extra:

```bash
pip install "ferro-ta[gpu]"    # installs torch>=2.0
```

Without the extra, GPU entry points raise `ModuleNotFoundError` with a
pointer to the install command. The CPU path remains the single source
of truth for correctness — GPU kernels are benchmarked against it.

## Scope

**In scope for the GPU backend:**
- Batch DTW over large matrices (N > 1000 rows)
- Vectorised Greeks across a strike/expiry grid
- Rolling statistics where the reduction dominates I/O

**Not in scope:**
- Streaming indicators (CPU latency is already sub-microsecond)
- Single-series indicators under 10k bars
- Anything TA-Lib has a hand-optimised C implementation for

## Consequences

- Default install stays lean.
- GPU contributors have a clear boundary: if a CPU kernel already wins,
  the GPU version doesn't ship.
- `docs/guides/gpu.md` documents when the GPU path is actually faster.
