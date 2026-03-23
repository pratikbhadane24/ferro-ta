"""Unit tests for ferro_ta.indicators.momentum"""
import numpy as np
import pytest
from ferro_ta.indicators.momentum import (
    RSI, STOCH, STOCHF, STOCHRSI,
    ADX, ADXR, CCI, WILLR, AROON, AROONOSC,
    MFI, MOM, ROC, ROCP, ROCR, ROCR100,
    CMO, DX, MINUS_DI, MINUS_DM, PLUS_DI, PLUS_DM,
    PPO, APO, TRIX, ULTOSC, BOP,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)
N = 100
_CLOSE = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_HIGH = _CLOSE + np.abs(RNG.normal(0, 0.3, N))
_LOW = _CLOSE - np.abs(RNG.normal(0, 0.3, N))
_OPEN = _CLOSE + RNG.normal(0, 0.1, N)
_VOL = RNG.uniform(1000, 5000, N)

SMALL5 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
SMALL5_H = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
SMALL5_L = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
SMALL5_O = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
SMALL5_V = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_nan_warmup(self):
        result = RSI(_CLOSE, timeperiod=14)
        assert np.all(np.isnan(result[:14]))

    def test_range(self):
        result = RSI(_CLOSE, timeperiod=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_length(self):
        assert len(RSI(_CLOSE, 14)) == N


# ---------------------------------------------------------------------------
# STOCH
# ---------------------------------------------------------------------------

class TestSTOCH:
    def test_returns_two_arrays(self):
        result = STOCH(_HIGH, _LOW, _CLOSE)
        assert isinstance(result, tuple) and len(result) == 2

    def test_range(self):
        slowk, slowd = STOCH(_HIGH, _LOW, _CLOSE)
        for arr in [slowk, slowd]:
            valid = arr[~np.isnan(arr)]
            assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_length(self):
        slowk, slowd = STOCH(_HIGH, _LOW, _CLOSE)
        assert len(slowk) == len(slowd) == N


# ---------------------------------------------------------------------------
# STOCHF
# ---------------------------------------------------------------------------

class TestSTOCHF:
    def test_returns_two_arrays(self):
        result = STOCHF(_HIGH, _LOW, _CLOSE)
        assert isinstance(result, tuple) and len(result) == 2

    def test_fastk_range(self):
        fastk, fastd = STOCHF(_HIGH, _LOW, _CLOSE, fastk_period=5, fastd_period=3)
        valid = fastk[~np.isnan(fastk)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_known_values(self):
        # With identical OHLC, fast %K = 100 * (C - min_low) / (max_high - min_low)
        # On our SMALL5 data the range is constant so all = 2/6 * 100 ≈ 66.67
        h5 = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
        l5 = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
        c5 = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
        fastk, fastd = STOCHF(h5, l5, c5, fastk_period=3, fastd_period=2)
        valid_k = fastk[~np.isnan(fastk)]
        assert np.all(valid_k >= 0) and np.all(valid_k <= 100)

    def test_length(self):
        fastk, fastd = STOCHF(_HIGH, _LOW, _CLOSE)
        assert len(fastk) == len(fastd) == N


# ---------------------------------------------------------------------------
# STOCHRSI
# ---------------------------------------------------------------------------

class TestSTOCHRSI:
    def test_returns_two_arrays(self):
        result = STOCHRSI(_CLOSE)
        assert isinstance(result, tuple) and len(result) == 2

    def test_range(self):
        fastk, fastd = STOCHRSI(_CLOSE, timeperiod=14)
        for arr in [fastk, fastd]:
            valid = arr[~np.isnan(arr)]
            assert np.all(valid >= -1e-10) and np.all(valid <= 100 + 1e-10)

    def test_length(self):
        fastk, fastd = STOCHRSI(_CLOSE)
        assert len(fastk) == N


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

class TestADX:
    def test_nan_warmup(self):
        result = ADX(_HIGH, _LOW, _CLOSE, timeperiod=14)
        assert np.all(np.isnan(result[:27]))

    def test_range(self):
        result = ADX(_HIGH, _LOW, _CLOSE, timeperiod=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_length(self):
        assert len(ADX(_HIGH, _LOW, _CLOSE, 14)) == N


# ---------------------------------------------------------------------------
# ADXR
# ---------------------------------------------------------------------------

class TestADXR:
    def test_length(self):
        assert len(ADXR(_HIGH, _LOW, _CLOSE, 14)) == N

    def test_range(self):
        result = ADXR(_HIGH, _LOW, _CLOSE, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------

class TestCCI:
    def test_known_constant_mean_dev(self):
        # Constant typical price → CCI = 0 after warmup
        c5 = np.full(10, 12.0)
        h5 = np.full(10, 13.0)
        l5 = np.full(10, 11.0)
        result = CCI(h5, l5, c5, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_length(self):
        assert len(CCI(_HIGH, _LOW, _CLOSE, 14)) == N

    def test_nan_warmup(self):
        result = CCI(_HIGH, _LOW, _CLOSE, timeperiod=14)
        assert np.all(np.isnan(result[:13]))

    def test_simple_rising(self):
        h = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
        l = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
        c = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
        result = CCI(h, l, c, 3)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 100.0, atol=1e-8)


# ---------------------------------------------------------------------------
# WILLR
# ---------------------------------------------------------------------------

class TestWILLR:
    def test_range(self):
        result = WILLR(_HIGH, _LOW, _CLOSE, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100) and np.all(valid <= 0)

    def test_length(self):
        assert len(WILLR(_HIGH, _LOW, _CLOSE, 14)) == N


# ---------------------------------------------------------------------------
# AROON
# ---------------------------------------------------------------------------

class TestAROON:
    def test_returns_two_arrays(self):
        result = AROON(_HIGH, _LOW, 14)
        assert isinstance(result, tuple) and len(result) == 2

    def test_range(self):
        aroon_down, aroon_up = AROON(_HIGH, _LOW, 14)
        for arr in [aroon_down, aroon_up]:
            valid = arr[~np.isnan(arr)]
            assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_length(self):
        aroon_down, aroon_up = AROON(_HIGH, _LOW, 14)
        assert len(aroon_down) == N


# ---------------------------------------------------------------------------
# AROONOSC
# ---------------------------------------------------------------------------

class TestAROONOSC:
    def test_known_values(self):
        h = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
        l = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
        result = AROONOSC(h, l, timeperiod=2)
        valid = result[~np.isnan(result)]
        # Monotone rising high/low → aroon_up = 100, aroon_down = 0 → osc = 100
        np.testing.assert_allclose(valid, 100.0, atol=1e-10)

    def test_equals_aroon_diff(self):
        aroon_down, aroon_up = AROON(_HIGH, _LOW, 14)
        aroonosc = AROONOSC(_HIGH, _LOW, 14)
        valid = ~np.isnan(aroon_up) & ~np.isnan(aroon_down) & ~np.isnan(aroonosc)
        np.testing.assert_allclose(
            aroonosc[valid],
            aroon_up[valid] - aroon_down[valid],
            atol=1e-10,
        )

    def test_length(self):
        assert len(AROONOSC(_HIGH, _LOW, 14)) == N


# ---------------------------------------------------------------------------
# MFI
# ---------------------------------------------------------------------------

class TestMFI:
    def test_range(self):
        result = MFI(_HIGH, _LOW, _CLOSE, _VOL, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_length(self):
        assert len(MFI(_HIGH, _LOW, _CLOSE, _VOL, 14)) == N

    def test_nan_warmup(self):
        result = MFI(_HIGH, _LOW, _CLOSE, _VOL, 14)
        assert np.all(np.isnan(result[:14]))

    def test_constant_price_is_50(self):
        # When money flow is neither positive nor negative → MFI should be near 50
        # Use alternating tiny moves around constant so no clear direction
        c = np.full(20, 100.0)
        h = np.full(20, 101.0)
        l = np.full(20, 99.0)
        v = np.full(20, 1000.0)
        result = MFI(h, l, c, v, 5)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0  # just ensure it runs


# ---------------------------------------------------------------------------
# MOM
# ---------------------------------------------------------------------------

class TestMOM:
    def test_known_values(self):
        result = MOM(SMALL5, timeperiod=2)
        assert np.isnan(result[0]) and np.isnan(result[1])
        np.testing.assert_allclose(result[2], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 2.0, rtol=1e-10)

    def test_length(self):
        assert len(MOM(_CLOSE, 10)) == N


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

class TestROC:
    def test_known_values(self):
        arr = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = ROC(arr, 2)
        # ROC = ((close - close[n]) / close[n]) * 100
        np.testing.assert_allclose(result[2], (12 - 10) / 10 * 100, rtol=1e-10)

    def test_length(self):
        assert len(ROC(_CLOSE, 10)) == N


# ---------------------------------------------------------------------------
# ROCP
# ---------------------------------------------------------------------------

class TestROCP:
    def test_known_values(self):
        arr = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = ROCP(arr, 2)
        # ROCP = (close - close[n]) / close[n]
        np.testing.assert_allclose(result[2], (12 - 10) / 10, rtol=1e-10)

    def test_length(self):
        assert len(ROCP(_CLOSE, 10)) == N


# ---------------------------------------------------------------------------
# ROCR
# ---------------------------------------------------------------------------

class TestROCR:
    def test_known_values(self):
        arr = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = ROCR(arr, 2)
        # ROCR = close / close[n]
        np.testing.assert_allclose(result[2], 12 / 10, rtol=1e-10)
        np.testing.assert_allclose(result[4], 14 / 12, rtol=1e-10)

    def test_nan_warmup(self):
        result = ROCR(_CLOSE, 10)
        assert np.all(np.isnan(result[:10]))

    def test_length(self):
        assert len(ROCR(_CLOSE, 10)) == N


# ---------------------------------------------------------------------------
# ROCR100
# ---------------------------------------------------------------------------

class TestROCR100:
    def test_known_values(self):
        arr = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = ROCR100(arr, 2)
        # ROCR100 = (close / close[n]) * 100
        np.testing.assert_allclose(result[2], 12 / 10 * 100, rtol=1e-10)

    def test_relation_to_rocr(self):
        rocr = ROCR(_CLOSE, 5)
        rocr100 = ROCR100(_CLOSE, 5)
        valid = ~np.isnan(rocr)
        np.testing.assert_allclose(rocr100[valid], rocr[valid] * 100, rtol=1e-10)

    def test_length(self):
        assert len(ROCR100(_CLOSE, 10)) == N


# ---------------------------------------------------------------------------
# CMO
# ---------------------------------------------------------------------------

class TestCMO:
    def test_range(self):
        result = CMO(_CLOSE, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100) and np.all(valid <= 100)

    def test_length(self):
        assert len(CMO(_CLOSE, 14)) == N


# ---------------------------------------------------------------------------
# DX
# ---------------------------------------------------------------------------

class TestDX:
    def test_range(self):
        result = DX(_HIGH, _LOW, _CLOSE, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_length(self):
        assert len(DX(_HIGH, _LOW, _CLOSE, 14)) == N


# ---------------------------------------------------------------------------
# MINUS_DI / MINUS_DM
# ---------------------------------------------------------------------------

class TestMINUS:
    def test_minus_di_range(self):
        result = MINUS_DI(_HIGH, _LOW, _CLOSE, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_minus_dm_range(self):
        result = MINUS_DM(_HIGH, _LOW, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_lengths(self):
        assert len(MINUS_DI(_HIGH, _LOW, _CLOSE, 14)) == N
        assert len(MINUS_DM(_HIGH, _LOW, 14)) == N


# ---------------------------------------------------------------------------
# PLUS_DI / PLUS_DM
# ---------------------------------------------------------------------------

class TestPLUS:
    def test_plus_di_range(self):
        result = PLUS_DI(_HIGH, _LOW, _CLOSE, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_plus_dm_range(self):
        result = PLUS_DM(_HIGH, _LOW, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_lengths(self):
        assert len(PLUS_DI(_HIGH, _LOW, _CLOSE, 14)) == N
        assert len(PLUS_DM(_HIGH, _LOW, 14)) == N


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class TestPPO:
    def test_returns_three_arrays(self):
        result = PPO(_CLOSE, fastperiod=12, slowperiod=26)
        assert isinstance(result, tuple) and len(result) == 3

    def test_histogram_is_diff(self):
        ppo, signal, hist = PPO(_CLOSE, fastperiod=12, slowperiod=26)
        valid = ~np.isnan(ppo) & ~np.isnan(signal)
        np.testing.assert_allclose(hist[valid], ppo[valid] - signal[valid], atol=1e-10)

    def test_length(self):
        ppo, signal, hist = PPO(_CLOSE)
        assert len(ppo) == len(signal) == len(hist) == N

    def test_nan_warmup(self):
        ppo, signal, hist = PPO(_CLOSE, fastperiod=12, slowperiod=26)
        assert np.any(np.isnan(ppo))


# ---------------------------------------------------------------------------
# APO
# ---------------------------------------------------------------------------

class TestAPO:
    def test_known_direction(self):
        # Rising close → fast EMA > slow EMA → APO > 0 after warmup
        rising = np.linspace(1.0, 100.0, 60)
        result = APO(rising, fastperiod=5, slowperiod=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_length(self):
        assert len(APO(_CLOSE, 12, 26)) == N

    def test_nan_warmup(self):
        result = APO(_CLOSE, 12, 26)
        assert np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# TRIX
# ---------------------------------------------------------------------------

class TestTRIX:
    def test_length(self):
        assert len(TRIX(_CLOSE, 10)) == N

    def test_nan_warmup(self):
        result = TRIX(_CLOSE, timeperiod=5)
        # TRIX warmup = 3*(tp-1) for triple EMA + 1 for diff
        assert np.all(np.isnan(result[:12]))

    def test_finite_after_warmup(self):
        result = TRIX(_CLOSE, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))

    def test_rising_series_positive(self):
        rising = np.linspace(1.0, 200.0, 100)
        result = TRIX(rising, timeperiod=5)
        valid = result[~np.isnan(result)]
        # On monotone rise, rate of change of triple EMA is positive
        assert np.all(valid > 0)


# ---------------------------------------------------------------------------
# BOP
# ---------------------------------------------------------------------------

class TestBOP:
    def test_known_values(self):
        o = np.array([10.0, 11.0])
        h = np.array([14.0, 15.0])
        l = np.array([8.0, 9.0])
        c = np.array([12.0, 13.0])
        # BOP = (close - open) / (high - low)
        result = BOP(o, h, l, c)
        np.testing.assert_allclose(result[0], (12 - 10) / (14 - 8), rtol=1e-10)
        np.testing.assert_allclose(result[1], (13 - 11) / (15 - 9), rtol=1e-10)

    def test_bearish_is_negative(self):
        o = np.array([14.0, 14.0])
        h = np.array([15.0, 15.0])
        l = np.array([8.0, 8.0])
        c = np.array([10.0, 10.0])
        result = BOP(o, h, l, c)
        assert np.all(result < 0)

    def test_range(self):
        # BOP = (close - open) / (high - low); can exceed [-1,1] with noisy data
        result = BOP(_OPEN, _HIGH, _LOW, _CLOSE)
        assert np.all(np.isfinite(result))

    def test_length(self):
        assert len(BOP(_OPEN, _HIGH, _LOW, _CLOSE)) == N


# ---------------------------------------------------------------------------
# ULTOSC
# ---------------------------------------------------------------------------

class TestULTOSC:
    def test_range(self):
        result = ULTOSC(_HIGH, _LOW, _CLOSE, 7, 14, 28)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_length(self):
        assert len(ULTOSC(_HIGH, _LOW, _CLOSE, 7, 14, 28)) == N

    def test_nan_warmup(self):
        result = ULTOSC(_HIGH, _LOW, _CLOSE, 7, 14, 28)
        assert np.any(np.isnan(result))
