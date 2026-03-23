#!/usr/bin/env python3
"""Generate the canonical OHLCV benchmark fixture.

This script creates benchmarks/fixtures/canonical_ohlcv.npz — a fixed,
deterministic dataset used by the benchmark suite for both numerical-regression
and performance tests.

Run once (or when you want to regenerate):
    python benchmarks/fixtures/generate_canonical.py

The fixture is checked into the repository so that CI does not need to
regenerate it every run.
"""

from __future__ import annotations

import pathlib

import numpy as np

SEED = 20240101
N = 2000  # number of bars

RNG = np.random.default_rng(SEED)

# Simulate a GBM-style price series
returns = RNG.normal(0, 0.01, N)
close = np.cumprod(1 + returns) * 100.0

open_ = close * RNG.uniform(0.998, 1.002, N)
high = np.maximum(close, open_) + np.abs(RNG.normal(0, 0.2, N))
low = np.minimum(close, open_) - np.abs(RNG.normal(0, 0.2, N))
volume = RNG.uniform(500_000, 2_000_000, N)

out_path = pathlib.Path(__file__).parent / "canonical_ohlcv.npz"
np.savez_compressed(
    out_path,
    open=open_,
    high=high,
    low=low,
    close=close,
    volume=volume,
)
print(f"Written {out_path}  (N={N}, seed={SEED})")
