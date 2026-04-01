"""Edge-case tests for ferro_ta indicators.

Covers NaN handling, empty arrays, single-element inputs, extreme values,
constant series, and dtype robustness.
"""

import numpy as np
import pytest

from ferro_ta import (
    ATR,
    BBANDS,
    EMA,
    MACD,
    MFI,
    OBV,
    RSI,
    SMA,
    STOCH,
    WMA,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_nan(arr):
    """True if every element is NaN."""
    return np.all(np.isnan(arr))


# ---------------------------------------------------------------------------
# Empty arrays
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """All indicators should return an empty array (not crash) for len-0 input."""

    def test_sma_empty(self):
        result = SMA(np.array([], dtype=np.float64), timeperiod=14)
        assert len(result) == 0

    def test_ema_empty(self):
        result = EMA(np.array([], dtype=np.float64), timeperiod=14)
        assert len(result) == 0

    def test_rsi_empty(self):
        result = RSI(np.array([], dtype=np.float64), timeperiod=14)
        assert len(result) == 0

    def test_bbands_empty(self):
        upper, mid, lower = BBANDS(np.array([], dtype=np.float64), timeperiod=5)
        assert len(upper) == 0
        assert len(mid) == 0
        assert len(lower) == 0

    def test_macd_empty(self):
        macd, sig, hist = MACD(np.array([], dtype=np.float64))
        assert len(macd) == 0

    def test_wma_empty(self):
        result = WMA(np.array([], dtype=np.float64), timeperiod=10)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Single-element arrays
# ---------------------------------------------------------------------------


class TestSingleElement:
    """Single-element inputs should produce NaN (insufficient data) without panic."""

    def test_sma_single(self):
        result = SMA(np.array([42.0]), timeperiod=14)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_ema_single(self):
        result = EMA(np.array([42.0]), timeperiod=14)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_rsi_single(self):
        result = RSI(np.array([42.0]), timeperiod=14)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_sma_period_1_single(self):
        """SMA(period=1) on a single element should return that element."""
        result = SMA(np.array([42.0]), timeperiod=1)
        assert len(result) == 1
        np.testing.assert_allclose(result[0], 42.0)


# ---------------------------------------------------------------------------
# All-NaN input
# ---------------------------------------------------------------------------


class TestAllNaN:
    """Indicators fed entirely NaN input should not crash and return all NaN."""

    @pytest.fixture()
    def nan_50(self):
        return np.full(50, np.nan)

    def test_sma_all_nan(self, nan_50):
        result = SMA(nan_50, timeperiod=14)
        assert len(result) == 50
        assert _all_nan(result)

    def test_ema_all_nan(self, nan_50):
        result = EMA(nan_50, timeperiod=14)
        assert len(result) == 50
        assert _all_nan(result)

    def test_rsi_all_nan(self, nan_50):
        result = RSI(nan_50, timeperiod=14)
        assert len(result) == 50
        assert _all_nan(result)


# ---------------------------------------------------------------------------
# NaN in the middle
# ---------------------------------------------------------------------------


class TestNaNInMiddle:
    """A single NaN in a valid series should propagate but not crash."""

    def test_sma_nan_mid(self):
        data = np.arange(1.0, 21.0)
        data[10] = np.nan
        result = SMA(data, timeperiod=5)
        assert len(result) == 20
        # Values around the NaN should be NaN
        for i in range(10, min(15, 20)):
            assert np.isnan(result[i])

    def test_rsi_nan_mid(self):
        data = np.arange(1.0, 31.0)
        data[15] = np.nan
        result = RSI(data, timeperiod=14)
        assert len(result) == 30


# ---------------------------------------------------------------------------
# Extreme values
# ---------------------------------------------------------------------------


class TestExtremeValues:
    """Indicators should not crash on very large or very small values."""

    def test_sma_large_values(self):
        data = np.full(50, 1e300)
        result = SMA(data, timeperiod=14)
        assert len(result) == 50
        # Non-NaN values should be ~1e300
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 1e300, rtol=1e-10)

    def test_sma_tiny_values(self):
        data = np.full(50, 1e-300)
        result = SMA(data, timeperiod=14)
        assert len(result) == 50
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 1e-300, rtol=1e-10)

    def test_rsi_large_monotone(self):
        """Monotonically increasing large values -> RSI should approach 100."""
        data = np.linspace(1e10, 2e10, 100)
        result = RSI(data, timeperiod=14)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] > 90.0  # strongly bullish

    def test_rsi_zero_change(self):
        """Constant series -> RSI should be 50 (or NaN in some implementations)."""
        data = np.full(100, 50.0)
        result = RSI(data, timeperiod=14)
        valid = result[~np.isnan(result)]
        # Constant series: no gains, no losses -> typically NaN or 50
        # Just verify no crash and valid range
        for v in valid:
            assert 0.0 <= v <= 100.0 or np.isnan(v)

    def test_bbands_constant_series(self):
        """Constant series -> upper == middle == lower (zero std dev)."""
        data = np.full(50, 100.0)
        upper, mid, lower = BBANDS(data, timeperiod=10)
        valid_mask = ~np.isnan(mid)
        np.testing.assert_allclose(upper[valid_mask], mid[valid_mask])
        np.testing.assert_allclose(lower[valid_mask], mid[valid_mask])


# ---------------------------------------------------------------------------
# Timeperiod edge cases
# ---------------------------------------------------------------------------


class TestTimePeriodEdge:
    """Boundary conditions for the timeperiod parameter."""

    def test_sma_period_equals_length(self):
        data = np.arange(1.0, 11.0)  # 10 elements
        result = SMA(data, timeperiod=10)
        assert len(result) == 10
        # Only last element should be valid
        assert not np.isnan(result[-1])
        np.testing.assert_allclose(result[-1], 5.5)

    def test_sma_period_exceeds_length(self):
        data = np.arange(1.0, 6.0)  # 5 elements
        result = SMA(data, timeperiod=10)
        assert len(result) == 5
        assert _all_nan(result)

    def test_ema_period_1(self):
        """EMA with period=1 should return the input itself."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = EMA(data, timeperiod=1)
        np.testing.assert_allclose(result, data)


# ---------------------------------------------------------------------------
# Multi-input indicator edge cases (OHLCV)
# ---------------------------------------------------------------------------


class TestOHLCVEdgeCases:
    """Edge cases for indicators requiring multiple price series."""

    def test_atr_empty(self):
        empty = np.array([], dtype=np.float64)
        result = ATR(empty, empty, empty, timeperiod=14)
        assert len(result) == 0

    def test_stoch_empty(self):
        empty = np.array([], dtype=np.float64)
        slowk, slowd = STOCH(empty, empty, empty)
        assert len(slowk) == 0
        assert len(slowd) == 0

    def test_obv_empty(self):
        empty = np.array([], dtype=np.float64)
        result = OBV(empty, empty)
        assert len(result) == 0

    def test_atr_single_bar(self):
        h = np.array([10.0])
        l = np.array([9.0])
        c = np.array([9.5])
        result = ATR(h, l, c, timeperiod=14)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_mfi_constant_price(self):
        """Constant price -> no money flow direction -> MFI should be well-defined."""
        n = 50
        h = np.full(n, 100.0)
        l = np.full(n, 100.0)
        c = np.full(n, 100.0)
        v = np.full(n, 1000.0)
        result = MFI(h, l, c, v, timeperiod=14)
        assert len(result) == n
        # Should not crash; values may be NaN or 50


# ---------------------------------------------------------------------------
# Dtype robustness
# ---------------------------------------------------------------------------


class TestDtypeRobustness:
    """Indicators should accept float32/int inputs and coerce to float64."""

    def test_sma_float32(self):
        data = np.arange(1.0, 51.0, dtype=np.float32)
        result = SMA(data, timeperiod=14)
        assert len(result) == 50

    def test_sma_int64(self):
        data = np.arange(1, 51, dtype=np.int64)
        result = SMA(data, timeperiod=14)
        assert len(result) == 50

    def test_rsi_float32(self):
        data = np.arange(1.0, 51.0, dtype=np.float32)
        result = RSI(data, timeperiod=14)
        assert len(result) == 50
