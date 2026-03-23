"""
Shared pytest fixtures for all test modules.

This module provides session-scoped fixtures to avoid duplicated data setup
across multiple test files. All fixtures use seeded RNG for reproducibility.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def ohlcv_500():
    """500-bar seeded OHLCV data, always the same across all test files.

    Returns a dictionary with keys: open, high, low, close, volume.
    All arrays are numpy float64 arrays of length 500.

    Seeded with RNG seed=42 for reproducibility.
    """
    rng = np.random.default_rng(42)
    n = 500

    # Generate realistic price movement
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    open_ = close + rng.standard_normal(n) * 0.3
    volume = rng.uniform(500.0, 5000.0, n)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


@pytest.fixture(scope="session")
def ohlcv_real():
    """Real data from tests/fixtures/ohlcv_daily.csv (252 bars).

    Returns a dictionary with keys: open, high, low, close, volume.
    All arrays are numpy float64 arrays of length 252.

    This is real market data for integration testing.
    """
    fixture_path = pathlib.Path(__file__).parent / "fixtures" / "ohlcv_daily.csv"

    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")

    df = pd.read_csv(fixture_path)

    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            pytest.skip(f"Required column '{col}' not found in fixture")

    return {
        "open": df["open"].to_numpy(dtype=np.float64),
        "high": df["high"].to_numpy(dtype=np.float64),
        "low": df["low"].to_numpy(dtype=np.float64),
        "close": df["close"].to_numpy(dtype=np.float64),
        "volume": df["volume"].to_numpy(dtype=np.float64),
    }


@pytest.fixture(scope="session")
def ohlcv_100():
    """100-bar seeded OHLCV data for quick tests.

    Returns a dictionary with keys: open, high, low, close, volume.
    All arrays are numpy float64 arrays of length 100.

    Seeded with RNG seed=42 for reproducibility.
    """
    rng = np.random.default_rng(42)
    n = 100

    # Generate realistic price movement
    close = 44.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.standard_normal(n) * 0.2
    volume = rng.uniform(500.0, 2000.0, n)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
