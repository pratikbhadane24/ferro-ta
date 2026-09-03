"""Python public API vs cargo-tested / hand-computed ferro_ta_core goldens.

``ferro_ta_core`` is a Rust crate and cannot be imported from Python. These
tests lock the indicators changed in the calculation-bug campaign against the
same goldens asserted in ``cargo test -p ferro_ta_core``. Streaming vs batch
parity is re-checked at 1e-10 for ATR, BBands, MACD, STOCH, and Supertrend.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferro_ta
from ferro_ta.data.streaming import (
    StreamingATR,
    StreamingBBands,
    StreamingMACD,
    StreamingStoch,
    StreamingSupertrend,
)

TOL = 1e-10

# Shared fixture used by several campaign indicators (linear close + simple OHLC).
LINEAR_10 = np.arange(1.0, 11.0)
LINEAR_12 = np.arange(1.0, 13.0)
LINEAR_16 = np.arange(1.0, 17.0)
CMO_CLOSE = np.array([1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0])

# Wide first bar so ATR seed that includes TR[0] cannot match TA-Lib.
ATR_HIGH = np.array([20.0, 12.0, 13.0, 14.0, 15.0])
ATR_LOW = np.array([5.0, 10.0, 11.0, 12.0, 13.0])
ATR_CLOSE = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

STOCHF_HIGH = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 15.0])
STOCHF_LOW = np.array([8.0, 9.0, 8.0, 10.0, 11.0, 12.0])
STOCHF_CLOSE = np.array([9.0, 11.0, 10.0, 12.0, 13.0, 14.0])

ICHIMOKU_HIGH = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
ICHIMOKU_LOW = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
ICHIMOKU_CLOSE = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])

RNG = np.random.default_rng(42)
N_STREAM = 200
STREAM_CLOSE = 44.0 + np.cumsum(RNG.standard_normal(N_STREAM) * 0.5)
STREAM_HIGH = STREAM_CLOSE + RNG.uniform(0.1, 1.0, N_STREAM)
STREAM_LOW = STREAM_CLOSE - RNG.uniform(0.1, 1.0, N_STREAM)


def _assert_nan_prefix(arr: np.ndarray, n: int) -> None:
    assert np.all(np.isnan(arr[:n])), f"expected {n} leading NaNs, got {arr[:n]}"


def _assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=TOL, equal_nan=True)


class TestOverlapGoldens:
    def test_dema_period3(self):
        # cargo: dema_golden_period3 — warmup 2*(3-1)=4, then 5..10
        result = ferro_ta.DEMA(LINEAR_10, timeperiod=3)
        _assert_nan_prefix(result, 4)
        _assert_close(result[4:], np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))

    def test_tema_period3(self):
        # cargo: tema_golden_period3 — warmup 3*(3-1)=6, then 7..10
        result = ferro_ta.TEMA(LINEAR_10, timeperiod=3)
        _assert_nan_prefix(result, 6)
        _assert_close(result[6:], np.array([7.0, 8.0, 9.0, 10.0]))

    def test_t3_sma_seeded_period3(self):
        # cargo: t3_golden_sma_seeded_period3 — T3[i] = i + 0.1 after warmup 12
        result = ferro_ta.T3(LINEAR_16, timeperiod=3, vfactor=0.7)
        warmup = 12
        _assert_nan_prefix(result, warmup)
        expected = np.arange(warmup, 16, dtype=np.float64) + 0.1
        _assert_close(result[warmup:], expected)

    def test_kama_first_output_at_timeperiod(self):
        # cargo: kama_first_output_at_timeperiod
        result = ferro_ta.KAMA(np.arange(1.0, 9.0), timeperiod=3)
        _assert_nan_prefix(result, 3)
        assert result[3] == pytest.approx(31.0 / 9.0, abs=TOL)
        assert result[4] == pytest.approx(335.0 / 81.0, abs=TOL)
        assert np.all(np.isfinite(result[3:]))

    def test_bbands_population_std(self):
        # cargo: bbands_varying_prices — population std, nbdev=2
        upper, middle, lower = ferro_ta.BBANDS(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]), timeperiod=3, nbdevup=2.0, nbdevdn=2.0
        )
        assert np.isnan(middle[0]) and np.isnan(middle[1])
        std = np.sqrt(2.0 / 3.0)
        _assert_close(middle[2:], np.array([2.0, 3.0, 4.0]))
        _assert_close(upper[2:], np.array([2.0, 3.0, 4.0]) + 2.0 * std)
        _assert_close(lower[2:], np.array([2.0, 3.0, 4.0]) - 2.0 * std)

    def test_macd_padded_until_signal(self):
        # cargo: macd_basic — MACD line NaN-padded until sig_start
        macd, signal, hist = ferro_ta.MACD(
            LINEAR_10, fastperiod=3, slowperiod=5, signalperiod=2
        )
        sig_start = 5 - 1 + 2 - 1
        _assert_nan_prefix(macd, sig_start)
        _assert_nan_prefix(signal, sig_start)
        assert np.all(np.isfinite(macd[sig_start:]))
        assert np.all(np.isfinite(signal[sig_start:]))
        _assert_close(hist[sig_start:], macd[sig_start:] - signal[sig_start:])
        # Linear series: EMA3[i]=i, EMA5[i]=i-1 after both valid → MACD=1
        _assert_close(macd[sig_start:], np.ones(len(LINEAR_10) - sig_start))
        _assert_close(signal[sig_start:], np.ones(len(LINEAR_10) - sig_start))


class TestMomentumGoldens:
    def test_rsi_wilder_matches_cmo_identity(self):
        # cargo: cmo_wilder_golden_period3 — CMO = 2*RSI - 100
        rsi = ferro_ta.RSI(CMO_CLOSE, timeperiod=3)
        cmo = ferro_ta.CMO(CMO_CLOSE, timeperiod=3)
        _assert_nan_prefix(rsi, 3)
        _assert_close(cmo[3:], 2.0 * rsi[3:] - 100.0)
        _assert_close(
            rsi[3:],
            np.array([200.0 / 3.0, 250.0 / 3.0, 2000.0 / 33.0, 235.0 / 3.0]),
        )

    def test_cmo_wilder_period3(self):
        result = ferro_ta.CMO(CMO_CLOSE, timeperiod=3)
        _assert_nan_prefix(result, 3)
        _assert_close(
            result[3:],
            np.array([100.0 / 3.0, 200.0 / 3.0, 700.0 / 33.0, 170.0 / 3.0]),
        )

    def test_ppo_signal_from_first_finite(self):
        # cargo: ppo_signal_from_first_finite — PPO(2,3,2) on 1..=10
        ppo, signal, hist = ferro_ta.PPO(
            LINEAR_10, fastperiod=2, slowperiod=3, signalperiod=2
        )
        _assert_nan_prefix(ppo, 2)
        _assert_nan_prefix(signal, 3)
        assert ppo[2] == pytest.approx(25.0, abs=TOL)
        assert ppo[3] == pytest.approx(50.0 / 3.0, abs=TOL)
        assert signal[3] == pytest.approx(125.0 / 6.0, abs=TOL)
        _assert_close(hist[3:], ppo[3:] - signal[3:])
        assert np.all(np.isfinite(signal[3:]))

    def test_stochf_sma_d(self):
        # cargo: stochf_golden_sma_d — %D is SMA, not EMA
        fastk, fastd = ferro_ta.STOCHF(
            STOCHF_HIGH, STOCHF_LOW, STOCHF_CLOSE, fastk_period=3, fastd_period=2
        )
        first_valid = 3
        _assert_nan_prefix(fastk, first_valid)
        _assert_nan_prefix(fastd, first_valid)
        _assert_close(fastk[first_valid:], np.array([80.0, 250.0 / 3.0, 80.0]))
        _assert_close(fastd[first_valid:], np.array([65.0, 245.0 / 3.0, 245.0 / 3.0]))


class TestVolatilityVolumeGoldens:
    def test_trange_bar0_is_nan(self):
        # cargo: trange_bar0_is_nan
        high = np.array([11.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([10.0, 12.0, 13.0])
        result = ferro_ta.TRANGE(high, low, close)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(3.0, abs=TOL)
        assert result[2] == pytest.approx(3.0, abs=TOL)

    def test_atr_skips_tr0(self):
        # cargo: atr_seeds_from_tr_1_through_period
        result = ferro_ta.ATR(ATR_HIGH, ATR_LOW, ATR_CLOSE, timeperiod=3)
        _assert_nan_prefix(result, 3)
        assert result[3] == pytest.approx(2.0, abs=TOL)
        assert result[4] == pytest.approx(2.0, abs=TOL)

    def test_obv_bar0_is_first_volume(self):
        # cargo: obv_bar0_accumulates_like_talib
        close = np.array([1.0, 2.0, 3.0, 2.0, 2.0])
        volume = np.array([100.0, 200.0, 300.0, 400.0, 50.0])
        result = ferro_ta.OBV(close, volume)
        _assert_close(result, np.array([100.0, 300.0, 600.0, 200.0, 200.0]))


class TestExtendedGoldens:
    def test_ichimoku_senkou_no_lookahead(self):
        # cargo: ichimoku_senkou_golden_displaced_past
        tenkan, kijun, senkou_a, senkou_b, chikou = ferro_ta.ICHIMOKU(
            ICHIMOKU_HIGH,
            ICHIMOKU_LOW,
            ICHIMOKU_CLOSE,
            tenkan_period=2,
            kijun_period=3,
            senkou_b_period=4,
            displacement=2,
        )
        _assert_nan_prefix(senkou_a, 4)
        _assert_close(senkou_a[4:], np.array([11.25, 12.25, 13.25, 14.25]))
        _assert_nan_prefix(senkou_b, 5)
        _assert_close(senkou_b[5:], np.array([11.5, 12.5, 13.5]))
        assert chikou[2] == pytest.approx(10.0, abs=TOL)
        assert chikou[7] == pytest.approx(15.0, abs=TOL)
        # Senkou at i uses tenkan/kijun at i-d, never the future bar.
        assert tenkan[2] == pytest.approx(11.5, abs=TOL)
        assert kijun[2] == pytest.approx(11.0, abs=TOL)
        assert senkou_a[4] == pytest.approx((tenkan[2] + kijun[2]) / 2.0, abs=TOL)

    def test_hull_ma_floor_sqrt(self):
        # cargo: hull_ma_uses_floor_sqrt_and_full_length_wma
        result = ferro_ta.HULL_MA(LINEAR_12, timeperiod=8)
        _assert_nan_prefix(result, 8)
        assert result[8] == pytest.approx(9.0, abs=1e-12)
        assert result[9] == pytest.approx(10.0, abs=1e-12)


class TestStreamingMatchesBatch:
    """Streaming vs batch at 1e-10 for the campaign streaming indicators."""

    def test_atr(self):
        batch = ferro_ta.ATR(STREAM_HIGH, STREAM_LOW, STREAM_CLOSE, timeperiod=14)
        streamer = StreamingATR(period=14)
        streamed = np.array(
            [
                streamer.update(h, l, c)
                for h, l, c in zip(STREAM_HIGH, STREAM_LOW, STREAM_CLOSE)
            ]
        )
        np.testing.assert_allclose(streamed, batch, equal_nan=True, atol=TOL)

    def test_bbands(self):
        b_u, b_m, b_l = ferro_ta.BBANDS(STREAM_CLOSE, timeperiod=20)
        streamer = StreamingBBands(period=20, nbdevup=2.0, nbdevdn=2.0)
        rows = [streamer.update(c) for c in STREAM_CLOSE]
        np.testing.assert_allclose(
            np.array([r[0] for r in rows]), b_u, equal_nan=True, atol=TOL
        )
        np.testing.assert_allclose(
            np.array([r[1] for r in rows]), b_m, equal_nan=True, atol=TOL
        )
        np.testing.assert_allclose(
            np.array([r[2] for r in rows]), b_l, equal_nan=True, atol=TOL
        )

    def test_macd(self):
        b_m, b_s, b_h = ferro_ta.MACD(
            STREAM_CLOSE, fastperiod=12, slowperiod=26, signalperiod=9
        )
        streamer = StreamingMACD(fastperiod=12, slowperiod=26, signalperiod=9)
        rows = [streamer.update(c) for c in STREAM_CLOSE]
        np.testing.assert_allclose(
            np.array([r[0] for r in rows]), b_m, equal_nan=True, atol=TOL
        )
        np.testing.assert_allclose(
            np.array([r[1] for r in rows]), b_s, equal_nan=True, atol=TOL
        )
        np.testing.assert_allclose(
            np.array([r[2] for r in rows]), b_h, equal_nan=True, atol=TOL
        )

    def test_stoch(self):
        b_k, b_d = ferro_ta.STOCH(
            STREAM_HIGH,
            STREAM_LOW,
            STREAM_CLOSE,
            fastk_period=5,
            slowk_period=3,
            slowd_period=3,
        )
        streamer = StreamingStoch(fastk_period=5, slowk_period=3, slowd_period=3)
        rows = [
            streamer.update(h, l, c)
            for h, l, c in zip(STREAM_HIGH, STREAM_LOW, STREAM_CLOSE)
        ]
        np.testing.assert_allclose(
            np.array([r[0] for r in rows]), b_k, equal_nan=True, atol=TOL
        )
        np.testing.assert_allclose(
            np.array([r[1] for r in rows]), b_d, equal_nan=True, atol=TOL
        )

    def test_supertrend(self):
        b_line, b_dir = ferro_ta.SUPERTREND(
            STREAM_HIGH, STREAM_LOW, STREAM_CLOSE, timeperiod=7, multiplier=3.0
        )
        streamer = StreamingSupertrend(period=7, multiplier=3.0)
        rows = [
            streamer.update(h, l, c)
            for h, l, c in zip(STREAM_HIGH, STREAM_LOW, STREAM_CLOSE)
        ]
        np.testing.assert_allclose(
            np.array([r[0] for r in rows]), b_line, equal_nan=True, atol=TOL
        )
        np.testing.assert_allclose(
            np.array([r[1] for r in rows]), b_dir, equal_nan=True, atol=TOL
        )
