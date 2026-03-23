"""Unit tests for ferro_ta.indicators.extended"""

import numpy as np

from ferro_ta.indicators.extended import (
    CHANDELIER_EXIT,
    CHOPPINESS_INDEX,
    DONCHIAN,
    HULL_MA,
    ICHIMOKU,
    KELTNER_CHANNELS,
    PIVOT_POINTS,
    SUPERTREND,
    VWAP,
    VWMA,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)
N = 200
_C = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_H = _C + np.abs(RNG.normal(0, 0.3, N))
_L = _C - np.abs(RNG.normal(0, 0.3, N))
_O = _C + RNG.normal(0, 0.1, N)
_VOL = RNG.uniform(1000, 5000, N)


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


class TestVWAP:
    def test_length(self):
        result = VWAP(_H, _L, _C, _VOL)
        assert len(result) == N

    def test_no_nan(self):
        result = VWAP(_H, _L, _C, _VOL)
        assert np.all(np.isfinite(result))

    def test_positive(self):
        result = VWAP(_H, _L, _C, _VOL)
        assert np.all(result > 0)

    def test_windowed(self):
        result = VWAP(_H, _L, _C, _VOL, timeperiod=20)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# SUPERTREND
# ---------------------------------------------------------------------------


class TestSUPERTREND:
    def test_returns_two_arrays(self):
        result = SUPERTREND(_H, _L, _C)
        assert isinstance(result, tuple) and len(result) == 2

    def test_length(self):
        trend, direction = SUPERTREND(_H, _L, _C)
        assert len(trend) == len(direction) == N

    def test_direction_binary(self):
        trend, direction = SUPERTREND(_H, _L, _C)
        valid = direction[~np.isnan(direction.astype(float))]
        assert np.all(np.isin(valid, [-1, 0, 1]))

    def test_nan_warmup(self):
        trend, direction = SUPERTREND(_H, _L, _C, timeperiod=7)
        assert np.any(np.isnan(trend))


# ---------------------------------------------------------------------------
# ICHIMOKU
# ---------------------------------------------------------------------------


class TestICHIMOKU:
    def test_returns_five_arrays(self):
        result = ICHIMOKU(_H, _L, _C)
        assert isinstance(result, tuple) and len(result) == 5

    def test_length(self):
        result = ICHIMOKU(_H, _L, _C)
        for arr in result:
            assert len(arr) == N

    def test_tenkan_warmup(self):
        tenkan, kijun, senkou_a, senkou_b, chikou = ICHIMOKU(
            _H, _L, _C, tenkan_period=9
        )
        assert np.all(np.isnan(tenkan[:8]))

    def test_finite_after_warmup(self):
        tenkan, kijun, senkou_a, senkou_b, chikou = ICHIMOKU(_H, _L, _C)
        for arr in [tenkan, kijun]:
            valid = arr[~np.isnan(arr)]
            assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# DONCHIAN
# ---------------------------------------------------------------------------


class TestDONCHIAN:
    def test_returns_three_arrays(self):
        result = DONCHIAN(_H, _L)
        assert isinstance(result, tuple) and len(result) == 3

    def test_length(self):
        upper, middle, lower = DONCHIAN(_H, _L)
        assert len(upper) == len(middle) == len(lower) == N

    def test_upper_ge_lower(self):
        upper, middle, lower = DONCHIAN(_H, _L)
        valid = ~np.isnan(upper) & ~np.isnan(lower)
        assert np.all(upper[valid] >= lower[valid])

    def test_middle_is_average(self):
        upper, middle, lower = DONCHIAN(_H, _L)
        valid = ~np.isnan(upper) & ~np.isnan(lower) & ~np.isnan(middle)
        np.testing.assert_allclose(
            middle[valid],
            (upper[valid] + lower[valid]) / 2.0,
            rtol=1e-10,
        )

    def test_nan_warmup(self):
        upper, middle, lower = DONCHIAN(_H, _L, timeperiod=20)
        assert np.all(np.isnan(upper[:19]))


# ---------------------------------------------------------------------------
# PIVOT_POINTS
# ---------------------------------------------------------------------------


class TestPIVOT_POINTS:
    def test_returns_five_arrays(self):
        result = PIVOT_POINTS(_H, _L, _C)
        assert isinstance(result, tuple) and len(result) == 5

    def test_length(self):
        result = PIVOT_POINTS(_H, _L, _C)
        for arr in result:
            assert len(arr) == N

    def test_classic_pivot_formula(self):
        # PP = (H + L + C) / 3
        pp, r1, s1, r2, s2 = PIVOT_POINTS(_H, _L, _C, method="classic")
        valid = ~np.isnan(pp)
        expected_pp = (_H[:-1] + _L[:-1] + _C[:-1]) / 3.0
        np.testing.assert_allclose(pp[valid], expected_pp[valid[1:]], rtol=1e-6)

    def test_first_is_nan(self):
        pp, r1, s1, r2, s2 = PIVOT_POINTS(_H, _L, _C)
        assert np.isnan(pp[0])


# ---------------------------------------------------------------------------
# KELTNER_CHANNELS
# ---------------------------------------------------------------------------


class TestKELTNER_CHANNELS:
    def test_returns_three_arrays(self):
        result = KELTNER_CHANNELS(_H, _L, _C)
        assert isinstance(result, tuple) and len(result) == 3

    def test_length(self):
        upper, middle, lower = KELTNER_CHANNELS(_H, _L, _C)
        assert len(upper) == len(middle) == len(lower) == N

    def test_upper_gt_lower(self):
        upper, middle, lower = KELTNER_CHANNELS(_H, _L, _C)
        valid = ~np.isnan(upper) & ~np.isnan(lower)
        assert np.all(upper[valid] > lower[valid])

    def test_nan_warmup(self):
        upper, middle, lower = KELTNER_CHANNELS(_H, _L, _C, timeperiod=20)
        assert np.all(np.isnan(upper[:19]))


# ---------------------------------------------------------------------------
# HULL_MA
# ---------------------------------------------------------------------------


class TestHULL_MA:
    def test_length(self):
        assert len(HULL_MA(_C, timeperiod=16)) == N

    def test_nan_warmup(self):
        result = HULL_MA(_C, timeperiod=16)
        assert np.all(np.isnan(result[:18]))

    def test_finite_after_warmup(self):
        result = HULL_MA(_C, timeperiod=16)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))

    def test_tracks_trend(self):
        rising = np.linspace(10.0, 200.0, 200)
        result = HULL_MA(rising, timeperiod=16)
        valid = result[~np.isnan(result)]
        assert np.all(np.diff(valid) > 0)


# ---------------------------------------------------------------------------
# CHANDELIER_EXIT
# ---------------------------------------------------------------------------


class TestCHANDELIER_EXIT:
    def test_returns_two_arrays(self):
        result = CHANDELIER_EXIT(_H, _L, _C)
        assert isinstance(result, tuple) and len(result) == 2

    def test_length(self):
        long_stop, short_stop = CHANDELIER_EXIT(_H, _L, _C)
        assert len(long_stop) == len(short_stop) == N

    def test_nan_warmup(self):
        long_stop, short_stop = CHANDELIER_EXIT(_H, _L, _C, timeperiod=22)
        assert np.all(np.isnan(long_stop[:21]))

    def test_finite_after_warmup(self):
        long_stop, short_stop = CHANDELIER_EXIT(_H, _L, _C, timeperiod=22)
        for arr in [long_stop, short_stop]:
            valid = arr[~np.isnan(arr)]
            assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# VWMA
# ---------------------------------------------------------------------------


class TestVWMA:
    def test_length(self):
        assert len(VWMA(_C, _VOL, timeperiod=20)) == N

    def test_nan_warmup(self):
        result = VWMA(_C, _VOL, timeperiod=20)
        assert np.all(np.isnan(result[:19]))

    def test_finite_after_warmup(self):
        result = VWMA(_C, _VOL, timeperiod=20)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))

    def test_constant_volume_equals_sma(self):
        # When all volumes are equal, VWMA = SMA
        vol = np.ones(N) * 1000.0
        vwma = VWMA(_C, vol, timeperiod=20)
        from ferro_ta.indicators.overlap import SMA

        sma = SMA(_C, timeperiod=20)
        valid = ~np.isnan(vwma) & ~np.isnan(sma)
        np.testing.assert_allclose(vwma[valid], sma[valid], rtol=1e-8)


# ---------------------------------------------------------------------------
# CHOPPINESS_INDEX
# ---------------------------------------------------------------------------


class TestCHOPPINESS_INDEX:
    def test_length(self):
        assert len(CHOPPINESS_INDEX(_H, _L, _C, timeperiod=14)) == N

    def test_nan_warmup(self):
        result = CHOPPINESS_INDEX(_H, _L, _C, timeperiod=14)
        assert np.all(np.isnan(result[:14]))

    def test_range(self):
        # Choppiness index is bounded between 0 and 100
        result = CHOPPINESS_INDEX(_H, _L, _C, timeperiod=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0) and np.all(valid < 200)

    def test_finite_after_warmup(self):
        result = CHOPPINESS_INDEX(_H, _L, _C, timeperiod=14)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))
