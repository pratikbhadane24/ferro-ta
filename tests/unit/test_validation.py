"""Tests for validation and error handling."""

import numpy as np
import pytest

from ferro_ta import (
    ATR,
    BBANDS,
    CDLDOJI,
    MACD,
    RSI,
    SMA,
    FerroTAInputError,
    FerroTAValueError,
)
from ferro_ta.core.exceptions import check_min_length, check_timeperiod

# ---------------------------------------------------------------------------
# Invalid timeperiod / period parameters → FerroTAValueError
# ---------------------------------------------------------------------------


class TestInvalidTimeperiod:
    """Invalid period parameters must raise FerroTAValueError."""

    def test_sma_timeperiod_zero(self):
        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 1"):
            SMA(np.array([1.0, 2.0, 3.0]), timeperiod=0)

    def test_sma_timeperiod_negative(self):
        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 1"):
            SMA(np.array([1.0, 2.0, 3.0]), timeperiod=-1)

    def test_rsi_timeperiod_zero(self):
        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 1"):
            RSI(np.array([1.0, 2.0, 3.0]), timeperiod=0)

    def test_macd_fast_slow_periods(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        with pytest.raises(FerroTAValueError):
            MACD(close, fastperiod=26, slowperiod=12)

    def test_bbands_timeperiod_zero(self):
        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 1"):
            BBANDS(np.array([1.0, 2.0, 3.0]), timeperiod=0)

    def test_atr_timeperiod_zero(self):
        h = np.array([1.0, 2.0, 3.0])
        low = np.array([0.5, 1.5, 2.5])
        c = np.array([0.8, 1.8, 2.8])
        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 1"):
            ATR(h, low, c, timeperiod=0)


# ---------------------------------------------------------------------------
# Mismatched array lengths → FerroTAInputError
# ---------------------------------------------------------------------------


class TestMismatchedLengths:
    """Mismatched OHLCV lengths must raise FerroTAInputError."""

    def test_atr_mismatched_lengths(self):
        h = np.array([1.0, 2.0, 3.0])
        low = np.array([0.5, 1.5])
        c = np.array([0.8, 1.8, 2.8])
        with pytest.raises(FerroTAInputError, match="same length"):
            ATR(h, low, c, timeperiod=2)

    def test_cdl_pattern_mismatched_lengths(self):
        open_ = np.array([1.0, 2.0, 3.0])
        high = np.array([1.1, 2.1])
        low = np.array([0.9, 1.9, 2.9])
        close = np.array([1.05, 2.05, 3.05])
        with pytest.raises(FerroTAInputError, match="same length"):
            CDLDOJI(open_, high, low, close)


# ---------------------------------------------------------------------------
# Empty and short arrays (defined behaviour or clear exception)
# ---------------------------------------------------------------------------


class TestEmptyAndShortArrays:
    """Empty or too-short arrays have defined behaviour or raise."""

    def test_sma_empty_array(self):
        # Empty array: _to_f64 returns shape (0,); Rust may return empty or raise.
        arr = np.array([], dtype=np.float64)
        result = SMA(arr, timeperiod=1)
        assert result.shape == (0,)

    def test_sma_single_element_timeperiod_one(self):
        arr = np.array([1.0])
        result = SMA(arr, timeperiod=1)
        assert len(result) == 1
        assert result[0] == 1.0

    def test_sma_short_array_timeperiod_larger_than_length(self):
        # len=3, timeperiod=5 → output is all NaN for warmup
        arr = np.array([1.0, 2.0, 3.0])
        result = SMA(arr, timeperiod=5)
        assert len(result) == 3
        assert np.all(np.isnan(result))

    def test_rsi_all_nan_input(self):
        # All-NaN input: output is all NaN (propagation)
        arr = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        result = RSI(arr, timeperiod=2)
        assert len(result) == 5
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# Validation helpers (check_timeperiod, check_min_length)
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    """Exported validation helpers behave as documented."""

    def test_check_timeperiod_ok(self):
        check_timeperiod(5)
        check_timeperiod(1)

    def test_check_timeperiod_raises(self):
        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 1"):
            check_timeperiod(0)
        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 1"):
            check_timeperiod(-1)

    def test_check_min_length_ok(self):
        check_min_length(np.array([1.0, 2.0, 3.0]), 2)
        check_min_length([1, 2, 3], 3)

    def test_check_min_length_raises(self):
        with pytest.raises(FerroTAInputError, match="at least 3 elements"):
            check_min_length(np.array([1.0, 2.0]), 3, name="input")


# ---------------------------------------------------------------------------
# Exception inheritance (ValueError still works)
# ---------------------------------------------------------------------------


class TestExceptionInheritance:
    """FerroTAValueError/FerroTAInputError are ValueErrors for backward compatibility."""

    def test_catch_value_error(self):
        with pytest.raises(ValueError, match="timeperiod must be >= 1"):
            SMA(np.array([1.0, 2.0, 3.0]), timeperiod=0)

    def test_catch_ferro_ta_value_error(self):
        with pytest.raises(FerroTAValueError):
            SMA(np.array([1.0, 2.0, 3.0]), timeperiod=0)
