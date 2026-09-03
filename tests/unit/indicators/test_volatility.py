"""Unit tests for ferro_ta.indicators.volatility"""

import numpy as np

from ferro_ta.indicators.overlap import BBANDS
from ferro_ta.indicators.volatility import (
    ATR,
    BBPERCENT,
    BBWIDTH,
    CHAIKIN_VOL,
    HISTORICAL_VOLATILITY,
    MASS,
    NATR,
    STARC,
    TRANGE,
    ULCER_INDEX,
)

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
        # Bar 0 is NaN (TA-Lib: no prior close). Subsequent bars are H-L = 3.
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 3.0, rtol=1e-10)

    def test_bar0_is_nan(self):
        result = TRANGE(SMALL_H, SMALL_L, SMALL_C)
        assert np.isnan(result[0])
        assert np.all(np.isfinite(result[1:]))

    def test_always_positive(self):
        result = TRANGE(_HIGH, _LOW, _CLOSE)
        assert np.all(result[1:] > 0)

    def test_length(self):
        assert len(TRANGE(_HIGH, _LOW, _CLOSE)) == N

    def test_formula_first_bar(self):
        h = np.array([15.0, 16.0, 17.0])
        l = np.array([10.0, 11.0, 12.0])
        c = np.array([13.0, 14.0, 15.0])
        result = TRANGE(h, l, c)
        # bar 0: NaN (TA-Lib; no previous close)
        assert np.isnan(result[0])
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


# ---------------------------------------------------------------------------
# CHAIKIN_VOL
# ---------------------------------------------------------------------------


class TestCHAIKIN_VOL:
    def test_length(self):
        assert len(CHAIKIN_VOL(_HIGH, _LOW, timeperiod=10, rocperiod=10)) == N

    def test_constant_range_is_zero(self):
        result = CHAIKIN_VOL(SMALL_H, SMALL_L, timeperiod=2, rocperiod=2)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# MASS
# ---------------------------------------------------------------------------


class TestMASS:
    def test_length(self):
        assert len(MASS(_HIGH, _LOW, timeperiod=9, sumperiod=25)) == N

    def test_constant_range_equals_sumperiod(self):
        h = np.arange(1.0, 21.0) + 1.0
        l = np.arange(1.0, 21.0) - 1.0
        result = MASS(h, l, timeperiod=3, sumperiod=3)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 3.0, atol=1e-10)


# ---------------------------------------------------------------------------
# BBPERCENT / BBWIDTH
# ---------------------------------------------------------------------------


class TestBBPERCENT:
    def test_matches_bbands(self):
        upper, middle, lower = BBANDS(_CLOSE, timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
        pct = BBPERCENT(_CLOSE, timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
        width = BBWIDTH(_CLOSE, timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
        valid = ~np.isnan(upper) & (upper != lower) & (middle != 0.0)
        np.testing.assert_allclose(
            pct[valid],
            (_CLOSE[valid] - lower[valid]) / (upper[valid] - lower[valid]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            width[valid], (upper[valid] - lower[valid]) / middle[valid], atol=1e-12
        )

    def test_warmup(self):
        pct = BBPERCENT(_CLOSE, timeperiod=5)
        assert np.all(np.isnan(pct[:4]))


# ---------------------------------------------------------------------------
# HISTORICAL_VOLATILITY
# ---------------------------------------------------------------------------


class TestHISTORICAL_VOLATILITY:
    def test_length(self):
        assert len(HISTORICAL_VOLATILITY(_CLOSE, timeperiod=10)) == N

    def test_constant_return_is_zero(self):
        close = 2.0 ** np.arange(8, dtype=np.float64)
        result = HISTORICAL_VOLATILITY(close, timeperiod=3, annual=252.0)
        assert np.all(np.isnan(result[:3]))
        np.testing.assert_allclose(result[3:], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# ULCER_INDEX
# ---------------------------------------------------------------------------


class TestULCER_INDEX:
    def test_length(self):
        assert len(ULCER_INDEX(_CLOSE, timeperiod=14)) == N

    def test_rising_series_is_zero(self):
        result = ULCER_INDEX(np.arange(1.0, 21.0), timeperiod=4)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# STARC
# ---------------------------------------------------------------------------


class TestSTARC:
    def test_returns_three_arrays(self):
        upper, middle, lower = STARC(_HIGH, _LOW, _CLOSE, timeperiod=15)
        assert len(upper) == len(middle) == len(lower) == N

    def test_linear_identity(self):
        close = np.arange(1.0, 11.0)
        high = close + 1.0
        low = close - 1.0
        upper, middle, lower = STARC(
            high, low, close, timeperiod=3, atr_period=3, multiplier=1.0
        )
        np.testing.assert_allclose(middle[3:], np.arange(3.0, 10.0), atol=1e-10)
        np.testing.assert_allclose(upper[3:], middle[3:] + 2.0, atol=1e-10)
        np.testing.assert_allclose(lower[3:], middle[3:] - 2.0, atol=1e-10)
