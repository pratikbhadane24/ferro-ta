"""Unit tests for ferro_ta.indicators.volatility"""

import numpy as np

from ferro_ta.indicators.volatility import ATR, NATR, TRANGE

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(3)
N = 100
_CLOSE = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_HIGH = _CLOSE + np.abs(RNG.normal(0, 0.3, N))
_LOW = _CLOSE - np.abs(RNG.normal(0, 0.3, N))

# Simple 5-bar data with constant range
SMALL_H = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
SMALL_L = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
SMALL_C = np.array([11.0, 12.0, 13.0, 14.0, 15.0])


# ---------------------------------------------------------------------------
# TRANGE
# ---------------------------------------------------------------------------


class TestTRANGE:
    def test_known_values_constant_range(self):
        result = TRANGE(SMALL_H, SMALL_L, SMALL_C)
        # First bar: only high-low = 3 (no prior close)
        np.testing.assert_allclose(result[0], 3.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], 3.0, rtol=1e-10)

    def test_no_nan(self):
        result = TRANGE(SMALL_H, SMALL_L, SMALL_C)
        assert np.all(np.isfinite(result))

    def test_always_positive(self):
        result = TRANGE(_HIGH, _LOW, _CLOSE)
        assert np.all(result > 0)

    def test_length(self):
        assert len(TRANGE(_HIGH, _LOW, _CLOSE)) == N

    def test_formula_first_bar(self):
        h = np.array([15.0, 16.0, 17.0])
        l = np.array([10.0, 11.0, 12.0])
        c = np.array([13.0, 14.0, 15.0])
        result = TRANGE(h, l, c)
        # bar 0: TRANGE = h[0] - l[0] = 5
        np.testing.assert_allclose(result[0], 5.0, rtol=1e-10)
        # bar 1: max(h[1]-l[1], |h[1]-c[0]|, |l[1]-c[0]|)
        #      = max(5, |16-13|, |11-13|) = max(5, 3, 2) = 5
        np.testing.assert_allclose(result[1], 5.0, rtol=1e-10)

    def test_with_gap(self):
        # Gap up: prev close=10, curr high=20, curr low=15
        h = np.array([10.0, 20.0])
        l = np.array([8.0, 15.0])
        c = np.array([10.0, 18.0])
        result = TRANGE(h, l, c)
        # bar 1: max(20-15, |20-10|, |15-10|) = max(5, 10, 5) = 10
        np.testing.assert_allclose(result[1], 10.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_timeperiod_1_equals_trange(self):
        atr = ATR(SMALL_H, SMALL_L, SMALL_C, timeperiod=1)
        trange = TRANGE(SMALL_H, SMALL_L, SMALL_C)
        # ATR(1) first bar is NaN, subsequent equal TRANGE
        np.testing.assert_allclose(atr[1:], trange[1:], rtol=1e-10)

    def test_nan_warmup(self):
        result = ATR(_HIGH, _LOW, _CLOSE, timeperiod=14)
        assert np.all(np.isnan(result[:14]))

    def test_length(self):
        assert len(ATR(_HIGH, _LOW, _CLOSE, 14)) == N

    def test_always_positive(self):
        result = ATR(_HIGH, _LOW, _CLOSE, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_constant_range_converges(self):
        # Constant TRANGE=3 → ATR should converge to 3
        h = np.full(100, 12.0) + np.arange(100) * 0.0
        l = np.full(100, 9.0) + np.arange(100) * 0.0
        c = np.full(100, 11.0) + np.arange(100) * 0.0
        result = ATR(h, l, c, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid[-1], 3.0, atol=0.01)


# ---------------------------------------------------------------------------
# NATR
# ---------------------------------------------------------------------------


class TestNATR:
    def test_nan_warmup(self):
        result = NATR(_HIGH, _LOW, _CLOSE, timeperiod=14)
        assert np.all(np.isnan(result[:14]))

    def test_length(self):
        assert len(NATR(_HIGH, _LOW, _CLOSE, 14)) == N

    def test_positive(self):
        result = NATR(_HIGH, _LOW, _CLOSE, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_relation_to_atr(self):
        # NATR = ATR / close * 100
        atr = ATR(_HIGH, _LOW, _CLOSE, 14)
        natr = NATR(_HIGH, _LOW, _CLOSE, 14)
        valid = ~np.isnan(atr) & ~np.isnan(natr)
        expected = atr[valid] / _CLOSE[valid] * 100
        np.testing.assert_allclose(natr[valid], expected, rtol=1e-5)
