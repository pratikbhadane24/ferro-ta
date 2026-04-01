"""Shared test helpers for ferro_ta unit tests.

This module consolidates common assertion patterns and data-generation
utilities that are duplicated across multiple test files.  Importing
from here keeps individual test modules DRY and makes it easier to
update assertion logic in one place.

Usage
-----
    from tests.unit.helpers import (
        nan_count, finite, assert_nan_warmup, assert_output_length,
        assert_finite_values, assert_range, make_ohlcv,
    )

Note: Each test file that already has inline helpers continues to work
unchanged.  These helpers are provided for *new* tests and for gradual
consolidation of existing ones.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Array inspection helpers
# ---------------------------------------------------------------------------


def nan_count(arr: np.ndarray) -> int:
    """Return the number of NaN entries in *arr*.

    Equivalent to the ``_nan_count`` functions duplicated in:
    - tests/unit/test_ferro_ta.py
    - tests/integration/test_vs_talib.py
    - tests/integration/test_vs_pandas_ta.py
    """
    return int(np.sum(np.isnan(arr)))


def finite(arr: np.ndarray) -> np.ndarray:
    """Return only the finite (non-NaN) elements of *arr*.

    Equivalent to the ``_finite`` helpers in:
    - tests/unit/test_ferro_ta.py
    - tests/unit/streaming/test_streaming.py
    """
    return arr[~np.isnan(arr)]


# ---------------------------------------------------------------------------
# Common assertion helpers
# ---------------------------------------------------------------------------


def assert_output_length(result: np.ndarray, expected_length: int) -> None:
    """Assert the indicator output has the expected length.

    This pattern (``assert len(result) == len(PRICES)``) appears 82+ times
    across the test suite.
    """
    assert len(result) == expected_length, (
        f"Expected output length {expected_length}, got {len(result)}"
    )


def assert_nan_warmup(result: np.ndarray, warmup: int) -> None:
    """Assert that the first *warmup* values are NaN and that at least
    one value after the warmup period is finite.

    This pattern (``assert np.all(np.isnan(result[:N]))``) appears 36+
    times in indicator tests.
    """
    assert np.all(np.isnan(result[:warmup])), (
        f"Expected first {warmup} values to be NaN"
    )
    if len(result) > warmup:
        assert np.any(np.isfinite(result[warmup:])), (
            f"Expected at least one finite value after warmup index {warmup}"
        )


def assert_finite_values(arr: np.ndarray) -> None:
    """Assert that *all* non-NaN values are finite (not +/-inf).

    The pattern ``np.all(np.isfinite(arr[~np.isnan(arr)]))`` appears
    60+ times across the test suite.
    """
    valid = arr[~np.isnan(arr)]
    assert np.all(np.isfinite(valid)), "Found non-finite (inf) values in output"


def assert_range(
    arr: np.ndarray,
    lo: float = 0.0,
    hi: float = 100.0,
    *,
    ignore_nan: bool = True,
) -> None:
    """Assert every (non-NaN) value in *arr* falls within [lo, hi].

    The ``valid >= 0 and valid <= 100`` pattern appears 10+ times for
    oscillator-type indicators (RSI, WILLR, CMO, etc.).
    """
    values = arr[~np.isnan(arr)] if ignore_nan else arr
    assert np.all(values >= lo), f"Found value below {lo}: {values.min()}"
    assert np.all(values <= hi), f"Found value above {hi}: {values.max()}"


def assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 0.0,
    ignore_nan: bool = True,
) -> None:
    """Assert element-wise closeness, optionally skipping NaN positions.

    Thin wrapper around ``np.testing.assert_allclose`` that mirrors the
    NaN-stripping pattern seen in integration tests.
    """
    if ignore_nan:
        mask = ~(np.isnan(actual) | np.isnan(expected))
        actual = actual[mask]
        expected = expected[mask]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def make_ohlcv(
    n: int = 100,
    seed: int = 42,
    base_price: float = 100.0,
) -> dict[str, np.ndarray]:
    """Generate reproducible synthetic OHLCV data.

    This pattern is duplicated across many test files with slight
    variations (different seeds, base prices, spread logic).  Using
    this helper ensures consistent generation logic.

    Returns a dict with keys: close, high, low, open, volume.
    """
    rng = np.random.default_rng(seed)
    close = base_price + np.cumsum(rng.normal(0, 0.5, n))
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.1, n)
    volume = rng.uniform(1000, 5000, n)
    return {
        "close": close,
        "high": high,
        "low": low,
        "open": open_,
        "volume": volume,
    }
