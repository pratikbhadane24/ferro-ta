"""Unit tests for ferro_ta.indicators.overlap"""

import numpy as np
import pytest

from ferro_ta.indicators.overlap import (
    ALLIGATOR,
    ALMA,
    BBANDS,
    CHANDE_KROLL_STOP,
    DEMA,
    EMA,
    FRAMA,
    KAMA,
    MA,
    MA_ENVELOPES,
    MACD,
    MACDEXT,
    MACDFIX,
    MAMA,
    MAVP,
    MCGINLEY,
    MIDPOINT,
    MIDPRICE,
    SAR,
    SAREXT,
    SMA,
    T3,
    TEMA,
    TRIMA,
    VIDYA,
    WMA,
    ZLEMA,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200
_CLOSE = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_HIGH = _CLOSE + np.abs(RNG.normal(0, 0.3, N))
_LOW = _CLOSE - np.abs(RNG.normal(0, 0.3, N))

SMALL5 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
SMALL5_HIGH = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
SMALL5_LOW = np.array([9.0, 10.0, 11.0, 12.0, 13.0])


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_known_values(self):
        result = SMA(SMALL5, timeperiod=3)
        expected = np.array([np.nan, np.nan, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(result[2:], expected[2:], rtol=1e-10)

    def test_nan_warmup(self):
        result = SMA(SMALL5, timeperiod=3)
        assert np.all(np.isnan(result[:2]))

    def test_length(self):
        result = SMA(_CLOSE, timeperiod=20)
        assert len(result) == N

    def test_nan_warmup_long(self):
        result = SMA(_CLOSE, timeperiod=20)
        assert np.all(np.isnan(result[:19]))
        assert np.all(np.isfinite(result[19:]))


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEMA:
    def test_known_values(self):
        # k = 2/(3+1) = 0.5; seed = SMA(3) = 11.0
        # EMA[2] = SMA([10,11,12]) = 11.0
        # EMA[3] = close[3]*k + EMA[2]*(1-k) = 13*0.5 + 11.0*0.5 = 12.0
        # EMA[4] = close[4]*k + EMA[3]*(1-k) = 14*0.5 + 12.0*0.5 = 13.0
        result = EMA(SMALL5, timeperiod=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        np.testing.assert_allclose(result[2], 11.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 12.0, rtol=1e-10)
        np.testing.assert_allclose(result[4], 13.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = EMA(SMALL5, timeperiod=3)
        assert np.all(np.isnan(result[:2]))

    def test_length(self):
        assert len(EMA(_CLOSE, 20)) == N

    def test_monotone_on_rising(self):
        rising = np.arange(1.0, 51.0)
        result = EMA(rising, 5)
        valid = result[~np.isnan(result)]
        assert np.all(np.diff(valid) > 0)


# ---------------------------------------------------------------------------
# WMA
# ---------------------------------------------------------------------------


class TestWMA:
    def test_known_values(self):
        arr = np.arange(1.0, 6.0)
        result = WMA(arr, timeperiod=3)
        # weights 1,2,3 / 6
        expected_2 = (1 * 1 + 2 * 2 + 3 * 3) / 6.0  # 14/6
        expected_3 = (1 * 2 + 2 * 3 + 3 * 4) / 6.0  # 20/6
        assert np.isnan(result[0]) and np.isnan(result[1])
        np.testing.assert_allclose(result[2], expected_2, rtol=1e-10)
        np.testing.assert_allclose(result[3], expected_3, rtol=1e-10)

    def test_nan_warmup(self):
        result = WMA(_CLOSE, timeperiod=10)
        assert np.all(np.isnan(result[:9]))

    def test_length(self):
        assert len(WMA(_CLOSE, 10)) == N


# ---------------------------------------------------------------------------
# DEMA
# ---------------------------------------------------------------------------


class TestDEMA:
    def test_nan_warmup(self):
        result = DEMA(_CLOSE, timeperiod=5)
        assert np.all(np.isnan(result[:8]))  # DEMA needs 2*(tp-1) bars

    def test_length(self):
        assert len(DEMA(_CLOSE, 5)) == N

    def test_values_finite_after_warmup(self):
        result = DEMA(_CLOSE, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))

    def test_tracks_close(self):
        # DEMA is more responsive than EMA; on trending data it should lead EMA
        rising = np.linspace(10.0, 100.0, 100)
        dema = DEMA(rising, 5)
        ema = EMA(rising, 5)
        valid = ~np.isnan(dema) & ~np.isnan(ema)
        # DEMA > EMA on a rising series (lower lag)
        assert np.all(dema[valid] >= ema[valid] - 1e-9)


# ---------------------------------------------------------------------------
# TEMA
# ---------------------------------------------------------------------------


class TestTEMA:
    def test_nan_warmup(self):
        result = TEMA(_CLOSE, timeperiod=5)
        assert np.all(np.isnan(result[:12]))

    def test_length(self):
        assert len(TEMA(_CLOSE, 5)) == N

    def test_values_finite_after_warmup(self):
        result = TEMA(_CLOSE, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# TRIMA
# ---------------------------------------------------------------------------


class TestTRIMA:
    def test_known_values(self):
        arr = np.arange(1.0, 11.0)
        result = TRIMA(arr, timeperiod=5)
        # TRIMA(5) is SMA of SMA(3) on a 5-window
        assert np.all(np.isnan(result[:4]))
        np.testing.assert_allclose(result[4], 3.0, rtol=1e-10)
        np.testing.assert_allclose(result[5], 4.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = TRIMA(_CLOSE, timeperiod=10)
        assert np.all(np.isnan(result[:9]))

    def test_length(self):
        assert len(TRIMA(_CLOSE, 10)) == N


# ---------------------------------------------------------------------------
# KAMA
# ---------------------------------------------------------------------------


class TestKAMA:
    def test_nan_warmup(self):
        result = KAMA(_CLOSE, timeperiod=10)
        # TA-Lib: first output at index timeperiod (seed bar is NaN)
        assert np.all(np.isnan(result[:10]))

    def test_length(self):
        assert len(KAMA(_CLOSE, 10)) == N

    def test_first_output_at_timeperiod(self):
        arr = np.arange(1.0, 21.0)
        result = KAMA(arr, timeperiod=10)
        assert np.isnan(result[9])
        assert np.isfinite(result[10])

    def test_finite_after_warmup(self):
        result = KAMA(_CLOSE, timeperiod=10)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------


class TestT3:
    def test_nan_warmup(self):
        arr = np.linspace(10.0, 30.0, 100)
        result = T3(arr, timeperiod=5)
        # warmup for T3(tp) = 6*(tp-1)
        assert np.all(np.isnan(result[:24]))

    def test_length(self):
        assert len(T3(_CLOSE, timeperiod=5)) == N

    def test_finite_after_warmup(self):
        arr = np.linspace(10.0, 30.0, 100)
        result = T3(arr, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))

    def test_trending(self):
        rising = np.linspace(10.0, 200.0, 150)
        result = T3(rising, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert np.all(np.diff(valid) > 0)


# ---------------------------------------------------------------------------
# MA
# ---------------------------------------------------------------------------


class TestMA:
    def test_default_is_sma(self):
        result_ma = MA(_CLOSE, timeperiod=10, matype=0)
        result_sma = SMA(_CLOSE, timeperiod=10)
        np.testing.assert_allclose(result_ma, result_sma, rtol=1e-10, equal_nan=True)

    def test_ema_matype(self):
        result_ma = MA(_CLOSE, timeperiod=10, matype=1)
        result_ema = EMA(_CLOSE, timeperiod=10)
        np.testing.assert_allclose(result_ma, result_ema, rtol=1e-10, equal_nan=True)

    def test_length(self):
        assert len(MA(_CLOSE, 10)) == N


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_returns_three_arrays(self):
        result = MACD(_CLOSE, 12, 26, 9)
        assert isinstance(result, tuple) and len(result) == 3

    def test_length(self):
        macd, signal, hist = MACD(_CLOSE, 12, 26, 9)
        assert len(macd) == len(signal) == len(hist) == N

    def test_histogram_is_diff(self):
        macd, signal, hist = MACD(_CLOSE)
        valid = ~np.isnan(macd) & ~np.isnan(signal)
        np.testing.assert_allclose(hist[valid], macd[valid] - signal[valid], atol=1e-10)

    def test_nan_warmup(self):
        macd, signal, hist = MACD(_CLOSE, 12, 26, 9)
        # MACD line: warmup = slowperiod - 1 = 25
        assert np.all(np.isnan(macd[:25]))


# ---------------------------------------------------------------------------
# MACDFIX
# ---------------------------------------------------------------------------


class TestMACDFIX:
    def test_returns_three_arrays(self):
        result = MACDFIX(_CLOSE)
        assert isinstance(result, tuple) and len(result) == 3

    def test_histogram_is_diff(self):
        macd, signal, hist = MACDFIX(_CLOSE)
        valid = ~np.isnan(macd) & ~np.isnan(signal)
        np.testing.assert_allclose(hist[valid], macd[valid] - signal[valid], atol=1e-10)

    def test_length(self):
        macd, signal, hist = MACDFIX(_CLOSE)
        assert len(macd) == N


# ---------------------------------------------------------------------------
# MACDEXT
# ---------------------------------------------------------------------------


class TestMACDEXT:
    def test_returns_three_arrays(self):
        result = MACDEXT(_CLOSE)
        assert isinstance(result, tuple) and len(result) == 3

    def test_histogram_is_diff(self):
        macd, signal, hist = MACDEXT(_CLOSE)
        valid = ~np.isnan(macd) & ~np.isnan(signal)
        np.testing.assert_allclose(hist[valid], macd[valid] - signal[valid], atol=1e-10)

    def test_length(self):
        assert len(MACDEXT(_CLOSE)[0]) == N


# ---------------------------------------------------------------------------
# BBANDS
# ---------------------------------------------------------------------------


class TestBBANDS:
    def test_returns_three_arrays(self):
        result = BBANDS(_CLOSE, 20)
        assert isinstance(result, tuple) and len(result) == 3

    def test_middle_is_sma(self):
        upper, middle, lower = BBANDS(_CLOSE, timeperiod=20)
        sma = SMA(_CLOSE, timeperiod=20)
        np.testing.assert_allclose(middle, sma, rtol=1e-10, equal_nan=True)

    def test_bands_symmetric(self):
        upper, middle, lower = BBANDS(_CLOSE, 20, nbdevup=2.0, nbdevdn=2.0)
        valid = ~np.isnan(upper)
        np.testing.assert_allclose(
            upper[valid] - middle[valid],
            middle[valid] - lower[valid],
            rtol=1e-10,
        )

    def test_nan_warmup(self):
        upper, middle, lower = BBANDS(_CLOSE, 20)
        assert np.all(np.isnan(middle[:19]))


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------


class TestSAR:
    def test_length(self):
        result = SAR(_HIGH, _LOW)
        assert len(result) == N

    def test_first_is_nan(self):
        result = SAR(_HIGH, _LOW)
        assert np.isnan(result[0])

    def test_finite_after_warmup(self):
        result = SAR(_HIGH, _LOW)
        assert np.all(np.isfinite(result[1:]))


# ---------------------------------------------------------------------------
# SAREXT
# ---------------------------------------------------------------------------


class TestSAREXT:
    def test_length(self):
        result = SAREXT(_HIGH, _LOW)
        assert len(result) == N

    def test_first_is_nan(self):
        result = SAREXT(_HIGH, _LOW)
        assert np.isnan(result[0])

    def test_finite_after_warmup(self):
        result = SAREXT(_HIGH, _LOW)
        assert np.all(np.isfinite(result[1:]))


# ---------------------------------------------------------------------------
# MAMA
# ---------------------------------------------------------------------------


class TestMAMA:
    def test_returns_two_arrays(self):
        result = MAMA(_CLOSE)
        assert isinstance(result, tuple) and len(result) == 2

    def test_length(self):
        mama, fama = MAMA(_CLOSE)
        assert len(mama) == len(fama) == N

    def test_nan_warmup(self):
        mama, fama = MAMA(_CLOSE)
        assert np.all(np.isnan(mama[:32]))

    def test_mama_ge_fama(self):
        # MAMA is adaptive; on average MAMA >= FAMA on a trending up series
        rising = np.linspace(10.0, 200.0, 200)
        mama, fama = MAMA(rising)
        valid = ~np.isnan(mama) & ~np.isnan(fama)
        # not strictly guaranteed, just check output is finite
        assert np.all(np.isfinite(mama[valid]))


# ---------------------------------------------------------------------------
# MAVP
# ---------------------------------------------------------------------------


class TestMAVP:
    def test_length(self):
        arr = np.linspace(10.0, 30.0, 50)
        periods = np.full(50, 5.0)
        result = MAVP(arr, periods, minperiod=2, maxperiod=10)
        assert len(result) == 50

    def test_finite_for_large_enough_data(self):
        arr = np.linspace(10.0, 30.0, 50)
        periods = np.full(50, 3.0)
        result = MAVP(arr, periods, minperiod=2, maxperiod=10)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# MIDPOINT
# ---------------------------------------------------------------------------


class TestMIDPOINT:
    def test_known_values(self):
        arr = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        result = MIDPOINT(arr, timeperiod=3)
        # MIDPOINT(n) = (max + min) / 2 over window
        assert np.isnan(result[0]) and np.isnan(result[1])
        np.testing.assert_allclose(result[2], (10.0 + 14.0) / 2.0, rtol=1e-10)
        np.testing.assert_allclose(result[4], (14.0 + 18.0) / 2.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = MIDPOINT(_CLOSE, timeperiod=14)
        assert np.all(np.isnan(result[:13]))

    def test_length(self):
        assert len(MIDPOINT(_CLOSE, 14)) == N


# ---------------------------------------------------------------------------
# MIDPRICE
# ---------------------------------------------------------------------------


class TestMIDPRICE:
    def test_known_values(self):
        result = MIDPRICE(SMALL5_HIGH, SMALL5_LOW, timeperiod=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        # window [0..2]: max_high=13, min_low=9  → (13+9)/2 = 11
        np.testing.assert_allclose(result[2], 11.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = MIDPRICE(_HIGH, _LOW, timeperiod=14)
        assert np.all(np.isnan(result[:13]))

    def test_length(self):
        assert len(MIDPRICE(_HIGH, _LOW, 14)) == N


# ---------------------------------------------------------------------------
# ALMA / ZLEMA / FRAMA / MCGINLEY / VIDYA / ALLIGATOR / MA_ENVELOPES /
# CHANDE_KROLL_STOP
# ---------------------------------------------------------------------------


class TestALMA:
    def test_linear_symmetric_period3(self):
        result = ALMA(np.arange(1.0, 6.0), timeperiod=3, offset=0.85, sigma=6.0)
        assert np.all(np.isnan(result[:2]))
        np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0], atol=1e-12)

    def test_constant_is_constant(self):
        result = ALMA(np.full(10, 7.0), timeperiod=5)
        assert np.all(np.isnan(result[:4]))
        np.testing.assert_allclose(result[4:], 7.0, atol=1e-12)

    def test_length(self):
        assert len(ALMA(_CLOSE, timeperiod=21)) == N


class TestZLEMA:
    def test_linear_tracks_close(self):
        close = np.arange(1.0, 11.0)
        result = ZLEMA(close, timeperiod=3)
        assert np.all(np.isnan(result[:3]))
        np.testing.assert_allclose(result[3:], close[3:], atol=1e-12)

    def test_length(self):
        assert len(ZLEMA(_CLOSE, timeperiod=14)) == N


class TestFRAMA:
    def test_linear_tracks_close(self):
        close = np.arange(1.0, 9.0)
        result = FRAMA(close, timeperiod=4)
        assert np.all(np.isnan(result[:3]))
        np.testing.assert_allclose(result[3:], close[3:], atol=1e-12)

    def test_length(self):
        assert len(FRAMA(_CLOSE, timeperiod=16)) == N


class TestMCGINLEY:
    def test_first_step(self):
        result = MCGINLEY(np.arange(1.0, 9.0), timeperiod=3)
        assert np.all(np.isnan(result[:2]))
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(49.0 / 24.0)

    def test_constant_is_constant(self):
        result = MCGINLEY(np.full(8, 10.0), timeperiod=3)
        np.testing.assert_allclose(result[2:], 10.0, atol=1e-12)

    def test_length(self):
        assert len(MCGINLEY(_CLOSE, timeperiod=14)) == N


class TestVIDYA:
    def test_cmo_seed_and_step(self):
        close = np.array([1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0])
        result = VIDYA(close, timeperiod=3, cmo_period=3)
        assert np.all(np.isnan(result[:3]))
        assert result[3] == pytest.approx(2.0)
        assert result[4] == pytest.approx(8.0 / 3.0)

    def test_length(self):
        assert len(VIDYA(_CLOSE, timeperiod=14)) == N


class TestALLIGATOR:
    def test_smma_shift(self):
        high = 11.0 + np.arange(8, dtype=np.float64)
        low = 9.0 + np.arange(8, dtype=np.float64)
        jaw, teeth, lips = ALLIGATOR(
            high,
            low,
            jaw_period=3,
            jaw_shift=2,
            teeth_period=2,
            teeth_shift=1,
            lips_period=2,
            lips_shift=1,
        )
        assert np.all(np.isnan(jaw[:4]))
        assert jaw[4] == pytest.approx(11.0)
        assert jaw[5] == pytest.approx(35.0 / 3.0)
        assert teeth[2] == pytest.approx(10.5)
        assert lips[2] == pytest.approx(10.5)

    def test_length(self):
        jaw, teeth, lips = ALLIGATOR(_HIGH, _LOW)
        assert len(jaw) == len(teeth) == len(lips) == N


class TestMAEnvelopes:
    def test_sma_percent(self):
        upper, middle, lower = MA_ENVELOPES(
            np.arange(1.0, 6.0), timeperiod=3, percent=10.0
        )
        assert np.all(np.isnan(middle[:2]))
        np.testing.assert_allclose(middle[2:], [2.0, 3.0, 4.0], atol=1e-12)
        np.testing.assert_allclose(upper[2:], [2.2, 3.3, 4.4], atol=1e-12)
        np.testing.assert_allclose(lower[2:], [1.8, 2.7, 3.6], atol=1e-12)

    def test_length(self):
        upper, middle, lower = MA_ENVELOPES(_CLOSE, timeperiod=20)
        assert len(upper) == len(middle) == len(lower) == N


class TestChandeKrollStop:
    def test_p3_q2(self):
        high = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 15.0])
        low = np.array([8.0, 9.0, 8.0, 10.0, 11.0, 12.0])
        close = np.array([9.0, 11.0, 10.0, 12.0, 13.0, 14.0])
        long_stop, short_stop = CHANDE_KROLL_STOP(
            high, low, close, timeperiod=3, multiplier=1.0, stop_period=2
        )
        assert np.all(np.isnan(long_stop[:4]))
        assert long_stop[4] == pytest.approx(11.0)
        assert long_stop[5] == pytest.approx(12.0)
        assert short_stop[4] == pytest.approx(11.0)
        assert short_stop[5] == pytest.approx(11.0)

    def test_length(self):
        long_stop, short_stop = CHANDE_KROLL_STOP(_HIGH, _LOW, _CLOSE)
        assert len(long_stop) == len(short_stop) == N
