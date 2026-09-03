"""Unit tests for ferro_ta.indicators.oscillators"""

import numpy as np

from ferro_ta.indicators.momentum import ROC
from ferro_ta.indicators.oscillators import (
    AC,
    AO,
    CHO,
    COPPOCK,
    DPO,
    GATOR,
    KST,
    PO,
    RVI,
    STC,
    TSI,
    VORTEX,
)
from ferro_ta.indicators.overlap import SMA, WMA
from ferro_ta.indicators.volume import ADOSC

RNG = np.random.default_rng(11)
N = 120
_CLOSE = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_HIGH = _CLOSE + np.abs(RNG.normal(0, 0.3, N))
_LOW = _CLOSE - np.abs(RNG.normal(0, 0.3, N))
_OPEN = _CLOSE + RNG.normal(0, 0.1, N)
_VOL = RNG.uniform(1000, 5000, N)


class TestAOAC:
    def test_ao_is_sma_median_diff(self):
        n = 40
        high = np.arange(1.0, n + 1.0) + 1.0
        low = np.arange(1.0, n + 1.0) - 1.0
        mid = 0.5 * (high + low)
        result = AO(high, low)
        expected = SMA(mid, 5) - SMA(mid, 34)
        np.testing.assert_allclose(result, expected, equal_nan=True, atol=1e-10)
        np.testing.assert_allclose(result[33], 14.5, atol=1e-10)

    def test_ac_is_ao_minus_sma(self):
        ao = AO(_HIGH, _LOW)
        result = AC(_HIGH, _LOW)
        # SMA of AO: leading NaNs must not poison the window
        valid_start = int(np.argmax(np.isfinite(ao)))
        ao_sma = np.full_like(ao, np.nan)
        tail = SMA(ao[valid_start:], 5)
        ao_sma[valid_start:] = tail
        expected = ao - ao_sma
        np.testing.assert_allclose(result, expected, equal_nan=True, atol=1e-10)


class TestPOAndDPO:
    def test_po_sma_diff(self):
        result = PO(_CLOSE, fastperiod=5, slowperiod=10)
        expected = SMA(_CLOSE, 5) - SMA(_CLOSE, 10)
        np.testing.assert_allclose(result, expected, equal_nan=True, atol=1e-10)

    def test_dpo_displaced(self):
        close = np.arange(1.0, 11.0)
        result = DPO(close, timeperiod=4)
        assert np.isnan(result[2])
        np.testing.assert_allclose(result[3], 1.0 - 2.5, atol=1e-10)


class TestCHO:
    def test_matches_adosc(self):
        np.testing.assert_allclose(
            CHO(_HIGH, _LOW, _CLOSE, _VOL, 3, 10),
            ADOSC(_HIGH, _LOW, _CLOSE, _VOL, 3, 10),
            equal_nan=True,
            atol=1e-12,
        )


class TestRVIVortexGator:
    def test_rvi_pair(self):
        rvi, signal = RVI(_OPEN, _HIGH, _LOW, _CLOSE)
        assert len(rvi) == len(signal) == N
        assert np.any(np.isfinite(rvi))
        assert np.any(np.isfinite(signal))

    def test_vortex_warmup(self):
        plus_vi, minus_vi = VORTEX(_HIGH, _LOW, _CLOSE, timeperiod=14)
        assert np.all(np.isnan(plus_vi[:14]))
        assert np.all(plus_vi[14:] >= 0)
        assert np.all(minus_vi[14:] >= 0)

    def test_gator_signs(self):
        upper, lower = GATOR(_HIGH, _LOW)
        assert len(upper) == N
        assert np.all(upper[np.isfinite(upper)] >= 0)
        assert np.all(lower[np.isfinite(lower)] <= 0)


class TestKSTTSISTCCoppock:
    def test_kst_pair(self):
        kst, signal = KST(_CLOSE)
        assert len(kst) == N
        assert np.any(np.isfinite(kst))
        assert np.any(np.isfinite(signal))

    def test_tsi_pair(self):
        tsi, signal = TSI(_CLOSE)
        assert len(tsi) == N
        assert np.any(np.isfinite(tsi))
        assert np.any(np.isfinite(signal))

    def test_stc_range(self):
        result = STC(_CLOSE, fastperiod=8, slowperiod=17, cycleperiod=5)
        valid = result[np.isfinite(result)]
        assert len(valid) > 0
        assert np.all((valid >= 0) & (valid <= 100))

    def test_coppock_composition(self):
        result = COPPOCK(_CLOSE, wma_period=10, roc1_period=14, roc2_period=11)
        summed = ROC(_CLOSE, 14) + ROC(_CLOSE, 11)
        start = int(np.argmax(np.isfinite(summed)))
        expected = np.full_like(summed, np.nan)
        expected[start:] = WMA(summed[start:], 10)
        np.testing.assert_allclose(result, expected, equal_nan=True, atol=1e-10)
