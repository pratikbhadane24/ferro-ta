"""
Benchmark data generator for cross-library comparison.

Produces C-contiguous float64 NumPy arrays that work correctly with all
six libraries (ferro-ta, TA-Lib, pandas-ta, ta, Tulipy, finta).
Critical: every array is np.ascontiguousarray(..., dtype=np.float64) to
prevent memory segmentation faults in C-extension libraries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)


def generate_ohlcv(size: int = 10_000) -> dict[str, np.ndarray]:
    """Return a dict of C-contiguous float64 OHLCV arrays.

    Uses a geometric Brownian motion walk so values are realistic (no
    negatives, bounded intraday spread).  Every array satisfies:
      high >= close >= low > 0
      open > 0
      volume > 0
    """
    # Geometric random walk for close
    returns = _RNG.normal(0.0002, 0.01, size)
    close = 100.0 * np.exp(np.cumsum(returns))

    noise_hi = np.abs(_RNG.normal(0, 0.005, size)) * close
    noise_lo = np.abs(_RNG.normal(0, 0.005, size)) * close

    high = close + noise_hi
    low = np.maximum(close - noise_lo, 0.01)  # never negative
    open_ = low + _RNG.random(size) * (high - low)
    volume = _RNG.uniform(1e5, 1e7, size)

    def _c(arr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr, dtype=np.float64)

    return {
        "open": _c(open_),
        "high": _c(high),
        "low": _c(low),
        "close": _c(close),
        "volume": _c(volume),
    }


def get_pandas_ohlcv(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert an OHLCV dict to a DataFrame with a DatetimeIndex.

    pandas-ta and finta both require a datetime-indexed DataFrame with
    lowercase column names (open/high/low/close/volume).
    """
    idx = pd.date_range("2015-01-01", periods=len(data["close"]), freq="D")
    return pd.DataFrame(data, index=idx)


# Pre-built datasets at several scales so benchmarks can import them directly
SMALL = generate_ohlcv(1_000)
MEDIUM = generate_ohlcv(10_000)
LARGE = generate_ohlcv(100_000)

SMALL_DF = get_pandas_ohlcv(SMALL)
MEDIUM_DF = get_pandas_ohlcv(MEDIUM)
LARGE_DF = get_pandas_ohlcv(LARGE)
