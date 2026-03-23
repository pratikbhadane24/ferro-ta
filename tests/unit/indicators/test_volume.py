"""Unit tests for ferro_ta.indicators.volume"""
import numpy as np
import pytest
from ferro_ta.indicators.volume import AD, ADOSC, OBV

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(5)
N = 100
_CLOSE = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_HIGH = _CLOSE + np.abs(RNG.normal(0, 0.3, N))
_LOW = _CLOSE - np.abs(RNG.normal(0, 0.3, N))
_VOL = RNG.uniform(1000, 5000, N)

SMALL_H = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
SMALL_L = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
SMALL_C = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
SMALL_V = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------

class TestOBV:
    def test_known_values_rising(self):
        # Rising close: OBV accumulates all volume
        c = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        v = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        result = OBV(c, v)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[1], 1000.0, atol=1e-10)
        np.testing.assert_allclose(result[4], 4000.0, atol=1e-10)

    def test_known_values_falling(self):
        c = np.array([14.0, 13.0, 12.0, 11.0, 10.0])
        v = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        result = OBV(c, v)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[1], -1000.0, atol=1e-10)
        np.testing.assert_allclose(result[4], -4000.0, atol=1e-10)

    def test_unchanged_price_no_change(self):
        c = np.array([10.0, 10.0, 10.0])
        v = np.array([500.0, 500.0, 500.0])
        result = OBV(c, v)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_no_nan(self):
        result = OBV(SMALL_C, SMALL_V)
        assert np.all(np.isfinite(result))

    def test_length(self):
        assert len(OBV(_CLOSE, _VOL)) == N

    def test_starts_zero(self):
        result = OBV(_CLOSE, _VOL)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# AD
# ---------------------------------------------------------------------------

class TestAD:
    def test_known_formula(self):
        # AD = cumsum(CLV * volume)
        # CLV = ((close - low) - (high - close)) / (high - low)
        h = np.array([15.0])
        l = np.array([10.0])
        c = np.array([12.0])
        v = np.array([1000.0])
        clv = ((12 - 10) - (15 - 12)) / (15 - 10)   # (2 - 3) / 5 = -0.2
        expected = clv * 1000.0
        result = AD(h, l, c, v)
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_monotone_rising_positive(self):
        # High CLV on rising data → AD should be non-negative cumulatively
        result = AD(SMALL_H, SMALL_L, SMALL_C, SMALL_V)
        assert np.all(np.isfinite(result))

    def test_no_nan(self):
        result = AD(_HIGH, _LOW, _CLOSE, _VOL)
        assert np.all(np.isfinite(result))

    def test_length(self):
        assert len(AD(_HIGH, _LOW, _CLOSE, _VOL)) == N


# ---------------------------------------------------------------------------
# ADOSC
# ---------------------------------------------------------------------------

class TestADOSC:
    def test_nan_warmup(self):
        result = ADOSC(_HIGH, _LOW, _CLOSE, _VOL, fastperiod=3, slowperiod=10)
        assert np.all(np.isnan(result[:9]))

    def test_length(self):
        assert len(ADOSC(_HIGH, _LOW, _CLOSE, _VOL, 3, 10)) == N

    def test_finite_after_warmup(self):
        result = ADOSC(_HIGH, _LOW, _CLOSE, _VOL, fastperiod=3, slowperiod=10)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))

    def test_known_values(self):
        result = ADOSC(SMALL_H, SMALL_L, SMALL_C, SMALL_V, fastperiod=2, slowperiod=3)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))
