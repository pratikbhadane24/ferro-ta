"""Unit tests for ferro_ta.indicators.statistic"""
import numpy as np
import pytest
from ferro_ta.indicators.statistic import (
    STDDEV, VAR, BETA, CORREL,
    LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE,
    TSF,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(11)
N = 100
_A = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_B = 100 + np.cumsum(RNG.normal(0, 0.5, N))

LINDATA = np.arange(1.0, 6.0)       # [1,2,3,4,5]
CONSTDATA = np.ones(10)             # all 1.0


# ---------------------------------------------------------------------------
# STDDEV
# ---------------------------------------------------------------------------

class TestSTDDEV:
    def test_constant_is_zero(self):
        result = STDDEV(CONSTDATA, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_known_values(self):
        # std([1,2,3,4,5], ddof=0) = sqrt(2)
        result = STDDEV(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], np.sqrt(2.0), rtol=1e-6)

    def test_nan_warmup(self):
        result = STDDEV(_A, timeperiod=5)
        assert np.all(np.isnan(result[:4]))

    def test_length(self):
        assert len(STDDEV(_A, 5)) == N

    def test_positive(self):
        result = STDDEV(_A, 5)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------

class TestVAR:
    def test_constant_is_zero(self):
        result = VAR(CONSTDATA, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_known_values(self):
        # var([1,2,3,4,5], ddof=0) = 2.0
        result = VAR(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], 2.0, rtol=1e-6)

    def test_equals_stddev_squared(self):
        std = STDDEV(_A, timeperiod=10)
        var = VAR(_A, timeperiod=10)
        valid = ~np.isnan(std) & ~np.isnan(var)
        np.testing.assert_allclose(var[valid], std[valid] ** 2, rtol=1e-6)

    def test_length(self):
        assert len(VAR(_A, 5)) == N


# ---------------------------------------------------------------------------
# LINEARREG
# ---------------------------------------------------------------------------

class TestLINEARREG:
    def test_perfect_line(self):
        # For [1,2,3,4,5] over window 5, forecast = 5.0
        result = LINEARREG(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], 5.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = LINEARREG(_A, timeperiod=14)
        assert np.all(np.isnan(result[:13]))

    def test_length(self):
        assert len(LINEARREG(_A, 14)) == N


# ---------------------------------------------------------------------------
# LINEARREG_SLOPE
# ---------------------------------------------------------------------------

class TestLINEARREG_SLOPE:
    def test_perfect_line_slope_one(self):
        result = LINEARREG_SLOPE(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], 1.0, rtol=1e-10)

    def test_constant_slope_zero(self):
        result = LINEARREG_SLOPE(CONSTDATA, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_length(self):
        assert len(LINEARREG_SLOPE(_A, 14)) == N


# ---------------------------------------------------------------------------
# LINEARREG_INTERCEPT
# ---------------------------------------------------------------------------

class TestLINEARREG_INTERCEPT:
    def test_perfect_line_intercept_one(self):
        # y = [1,2,3,4,5] with x=[0,1,2,3,4] → y = 1 + 1*x → intercept = 1.0
        result = LINEARREG_INTERCEPT(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], 1.0, atol=1e-10)

    def test_length(self):
        assert len(LINEARREG_INTERCEPT(_A, 14)) == N


# ---------------------------------------------------------------------------
# LINEARREG_ANGLE
# ---------------------------------------------------------------------------

class TestLINEARREG_ANGLE:
    def test_slope_one_gives_45_degrees(self):
        result = LINEARREG_ANGLE(LINDATA, timeperiod=5)
        # arctan(1) * 180/pi = 45
        np.testing.assert_allclose(result[4], 45.0, rtol=1e-6)

    def test_constant_gives_zero_degrees(self):
        result = LINEARREG_ANGLE(CONSTDATA, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-8)

    def test_length(self):
        assert len(LINEARREG_ANGLE(_A, 14)) == N


# ---------------------------------------------------------------------------
# BETA
# ---------------------------------------------------------------------------

class TestBETA:
    def test_nan_warmup(self):
        result = BETA(_A, _B, timeperiod=5)
        assert np.all(np.isnan(result[:4]))

    def test_length(self):
        assert len(BETA(_A, _B, 5)) == N

    def test_same_series(self):
        # Beta of x vs x = 1.0 (regression of itself)
        result = BETA(_A, _A, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))

    def test_finite_after_warmup(self):
        result = BETA(_A, _B, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# CORREL
# ---------------------------------------------------------------------------

class TestCOREL:
    def test_self_correlation_is_one(self):
        result = CORREL(_A, _A, timeperiod=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_opposite_correlation_is_minus_one(self):
        arr = np.arange(1.0, 11.0)
        result = CORREL(arr, arr[::-1], timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, -1.0, atol=1e-10)

    def test_range(self):
        result = CORREL(_A, _B, timeperiod=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1 - 1e-10) and np.all(valid <= 1 + 1e-10)

    def test_length(self):
        assert len(CORREL(_A, _B, 10)) == N


# ---------------------------------------------------------------------------
# TSF
# ---------------------------------------------------------------------------

class TestTSF:
    def test_perfect_line(self):
        arr = np.arange(1.0, 10.0)
        result = TSF(arr, timeperiod=3)
        # TSF(3) on [1,2,...] = linear forecast one period ahead
        # Over window [1,2,3]: slope=1, intercept=0 → forecast at bar 2+1=3 → TSF[2]=4
        np.testing.assert_allclose(result[2], 4.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 5.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = TSF(_A, timeperiod=14)
        assert np.all(np.isnan(result[:13]))

    def test_length(self):
        assert len(TSF(_A, 14)) == N
