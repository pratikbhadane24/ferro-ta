"""Tests for alerts, crypto helpers, chunked processing,
regime detection, performance attribution, and dashboard helpers.
"""

from __future__ import annotations

import importlib
import runpy

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(31415)


def _make_close(n: int = 200) -> np.ndarray:
    return np.cumprod(1 + RNG.normal(0, 0.01, n)) * 100.0


def _make_ohlcv(n: int = 200):
    close = _make_close(n)
    open_ = close * RNG.uniform(0.995, 1.005, n)
    high = np.maximum(close, open_) + RNG.uniform(0, 0.5, n)
    low = np.minimum(close, open_) - RNG.uniform(0, 0.5, n)
    volume = RNG.uniform(500, 5000, n)
    return open_, high, low, close, volume


# ===========================================================================
# Alerts
# ===========================================================================


class TestAlertsLowLevel:
    """Tests for low-level alert condition functions."""

    def test_check_threshold_cross_above(self):
        from ferro_ta.tools.alerts import check_threshold

        series = np.array([20.0, 25.0, 30.0, 35.0, 28.0])
        mask = check_threshold(series, level=29.0, direction=1)
        # Cross above fires when series was <= level and now > level.
        # Bar 0: no prior bar, always 0.
        # Bar 2: prev=25 <= 29, curr=30 > 29  → fires
        assert mask[0] == 0
        assert mask[2] == 1
        assert mask[1] == 0
        assert mask[3] == 0  # 30 > 29 previously, so no new crossing

    def test_check_threshold_cross_below(self):
        from ferro_ta.tools.alerts import check_threshold

        series = np.array([70.0, 65.0, 28.0, 25.0, 35.0])
        mask = check_threshold(series, level=30.0, direction=-1)
        # index 2: 65 >= 30 → 28 < 30: cross below
        assert mask[2] == 1
        assert mask[0] == 0
        assert mask[4] == 0  # 25 < 30 already, 35 > 30 is not a cross-below

    def test_check_threshold_invalid_direction(self):
        from ferro_ta.tools.alerts import check_threshold

        with pytest.raises(Exception):
            check_threshold(np.array([1.0, 2.0]), level=1.5, direction=0)

    def test_check_cross_bullish(self):
        from ferro_ta.tools.alerts import check_cross

        fast = np.array([10.0, 12.0, 15.0, 14.0, 16.0])
        slow = np.array([13.0, 13.0, 13.0, 13.0, 13.0])
        mask = check_cross(fast, slow)
        # fast crosses above slow at index 2 (12 <= 13 → 15 > 13)
        assert mask[2] == 1  # bullish
        assert mask[0] == 0

    def test_check_cross_bearish(self):
        from ferro_ta.tools.alerts import check_cross

        fast = np.array([15.0, 15.0, 12.0, 11.0])
        slow = np.array([13.0, 13.0, 13.0, 13.0])
        mask = check_cross(fast, slow)
        # fast crosses below slow at index 2 (15 >= 13 → 12 < 13)
        assert mask[2] == -1  # bearish

    def test_check_cross_length_mismatch_raises(self):
        from ferro_ta.tools.alerts import check_cross

        with pytest.raises(Exception):
            check_cross(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_collect_alert_bars(self):
        from ferro_ta.tools.alerts import collect_alert_bars

        mask = np.array([0, 1, 0, 0, 1, -1], dtype=np.int8)
        bars = collect_alert_bars(mask)
        assert list(bars) == [1, 4, 5]

    def test_collect_alert_bars_empty(self):
        from ferro_ta.tools.alerts import collect_alert_bars

        mask = np.zeros(10, dtype=np.int8)
        bars = collect_alert_bars(mask)
        assert len(bars) == 0


class TestAlertManager:
    """Tests for the AlertManager class."""

    def test_run_backtest_returns_list(self):
        from ferro_ta import RSI
        from ferro_ta.tools.alerts import AlertManager

        close = _make_close(200)
        rsi = np.asarray(RSI(close, timeperiod=14), dtype=np.float64)
        am = AlertManager(symbol="TEST")
        am.add_threshold_condition("rsi_os", rsi, level=30.0, direction=-1)
        events = am.run_backtest()
        assert isinstance(events, list)

    def test_backtest_no_external_calls_by_default(self):
        """Backtest mode must not invoke callback unless force_live=True."""
        from ferro_ta import SMA
        from ferro_ta.tools.alerts import AlertManager

        close = _make_close(100)
        sma10 = np.asarray(SMA(close, timeperiod=10), dtype=np.float64)
        sma30 = np.asarray(SMA(close, timeperiod=30), dtype=np.float64)

        called = []

        def cb(ev):
            called.append(ev)

        am = AlertManager()
        am.add_cross_condition("sma_x", sma10, sma30, callback=cb)
        events = am.run_backtest()  # default live=False
        assert len(called) == 0, "callback must not fire in backtest mode"
        assert isinstance(events, list)

    def test_backtest_force_live_invokes_callback(self):
        from ferro_ta import RSI
        from ferro_ta.tools.alerts import AlertManager

        close = _make_close(300)
        rsi = np.asarray(RSI(close, timeperiod=14), dtype=np.float64)

        fired = []

        def cb(ev):
            fired.append(ev)

        am = AlertManager()
        am.add_threshold_condition("rsi_os", rsi, level=30.0, direction=-1, callback=cb)
        events = am.run_backtest(force_live=True)
        assert len(fired) == len(events)

    def test_event_payload_contains_symbol(self):
        from ferro_ta import RSI
        from ferro_ta.tools.alerts import AlertManager

        close = _make_close(200)
        rsi = np.asarray(RSI(close, timeperiod=14), dtype=np.float64)
        am = AlertManager(symbol="BTCUSD")
        am.add_threshold_condition("rsi_os", rsi, level=30.0, direction=-1)
        events = am.run_backtest()
        for ev in events:
            assert ev.payload.get("symbol") == "BTCUSD"

    def test_event_bar_index_valid(self):
        from ferro_ta import SMA
        from ferro_ta.tools.alerts import AlertManager

        close = _make_close(100)
        sma5 = np.asarray(SMA(close, timeperiod=5), dtype=np.float64)
        sma20 = np.asarray(SMA(close, timeperiod=20), dtype=np.float64)
        am = AlertManager()
        am.add_cross_condition("x", sma5, sma20)
        events = am.run_backtest()
        for ev in events:
            assert 0 <= ev.bar_index < len(close)

    def test_alert_event_to_dict(self):
        from ferro_ta.tools.alerts import AlertEvent

        ev = AlertEvent("my_cond", 42, value=27.5, payload={"symbol": "X"})
        d = ev.to_dict()
        assert d["condition_id"] == "my_cond"
        assert d["bar_index"] == 42
        assert d["symbol"] == "X"


# ===========================================================================
# Crypto helpers
# ===========================================================================


class TestCryptoFunding:
    def test_funding_pnl_shape(self):
        from ferro_ta.analysis.crypto import funding_pnl

        pos = np.ones(100)
        rate = RNG.normal(0, 0.0001, 100)
        pnl = funding_pnl(pos, rate)
        assert pnl.shape == (100,)

    def test_funding_pnl_cumulative(self):
        from ferro_ta.analysis.crypto import funding_pnl

        pos = np.ones(5)
        rate = np.array([0.0001, 0.0002, -0.0001, 0.0001, 0.0001])
        pnl = funding_pnl(pos, rate)
        expected = np.cumsum(-pos * rate)
        np.testing.assert_allclose(pnl, expected)

    def test_funding_pnl_long_pays_positive_rate(self):
        """Long position should pay (negative PnL) when funding rate > 0."""
        from ferro_ta.analysis.crypto import funding_pnl

        pos = np.ones(1)
        rate = np.array([0.001])  # positive rate → long pays
        pnl = funding_pnl(pos, rate)
        assert pnl[0] < 0

    def test_funding_pnl_short_receives_positive_rate(self):
        """Short position should receive (positive PnL) when funding rate > 0."""
        from ferro_ta.analysis.crypto import funding_pnl

        pos = np.array([-1.0])
        rate = np.array([0.001])
        pnl = funding_pnl(pos, rate)
        assert pnl[0] > 0

    def test_funding_pnl_length_mismatch_raises(self):
        from ferro_ta.analysis.crypto import funding_pnl

        with pytest.raises(Exception):
            funding_pnl(np.ones(5), np.ones(4))


class TestCryptoBarLabels:
    def test_continuous_bar_labels_shape(self):
        from ferro_ta.analysis.crypto import continuous_bar_labels

        labels = continuous_bar_labels(10, 3)
        assert labels.shape == (10,)

    def test_continuous_bar_labels_values(self):
        from ferro_ta.analysis.crypto import continuous_bar_labels

        labels = continuous_bar_labels(10, 3)
        expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
        np.testing.assert_array_equal(labels, expected)

    def test_continuous_bar_labels_period_one(self):
        from ferro_ta.analysis.crypto import continuous_bar_labels

        labels = continuous_bar_labels(5, 1)
        np.testing.assert_array_equal(labels, [0, 1, 2, 3, 4])

    def test_session_boundaries_daily(self):
        from ferro_ta.analysis.crypto import session_boundaries

        NS_PER_HOUR = np.int64(3_600_000_000_000)
        # Use a UTC midnight timestamp as base: 1_699_920_000 seconds = Nov 14, 2023 00:00:00 UTC
        base = np.int64(1_699_920_000) * np.int64(1_000_000_000)  # midnight UTC
        # 48 hourly bars = 2 full days
        ts = base + np.arange(48, dtype=np.int64) * NS_PER_HOUR
        bounds = session_boundaries(ts)
        assert bounds[0] == 0  # first bar always included
        # Should have exactly 2 boundaries (day 0 and day 1)
        assert len(bounds) == 2
        assert bounds[1] == 24  # second day starts at bar 24


class TestResampleContinuous:
    def test_resample_continuous_shape(self):
        from ferro_ta.analysis.crypto import resample_continuous

        o, h, l, c, v = _make_ohlcv(100)
        ro, rh, rl, rc, rv = resample_continuous((o, h, l, c, v), period_bars=5)
        assert len(rc) == 20  # 100 / 5

    def test_resample_continuous_high_ge_low(self):
        from ferro_ta.analysis.crypto import resample_continuous

        o, h, l, c, v = _make_ohlcv(100)
        _, rh, rl, _, _ = resample_continuous((o, h, l, c, v), period_bars=5)
        assert np.all(rh >= rl)

    def test_resample_continuous_invalid_period_raises(self):
        from ferro_ta.analysis.crypto import resample_continuous

        o, h, l, c, v = _make_ohlcv(10)
        with pytest.raises(ValueError):
            resample_continuous((o, h, l, c, v), period_bars=0)


# ===========================================================================
# Chunked processing
# ===========================================================================


class TestChunked:
    def test_make_chunk_ranges_shape(self):
        from ferro_ta.data.chunked import make_chunk_ranges

        ranges = make_chunk_ranges(100, 30, 10)
        assert ranges.ndim == 2
        assert ranges.shape[1] == 2

    def test_make_chunk_ranges_coverage(self):
        """All input indices must be covered by some range."""
        from ferro_ta.data.chunked import make_chunk_ranges

        n = 97
        ranges = make_chunk_ranges(n, 20, 5)
        covered = set()
        for start, end in ranges:
            covered.update(range(int(start), int(end)))
        assert 0 in covered
        assert (n - 1) in covered

    def test_trim_overlap_basic(self):
        from ferro_ta.data.chunked import trim_overlap

        arr = np.arange(10, dtype=np.float64)
        trimmed = trim_overlap(arr, overlap=3)
        np.testing.assert_array_equal(trimmed, arr[3:])

    def test_trim_overlap_zero(self):
        from ferro_ta.data.chunked import trim_overlap

        arr = np.arange(5, dtype=np.float64)
        trimmed = trim_overlap(arr, overlap=0)
        np.testing.assert_array_equal(trimmed, arr)

    def test_stitch_chunks_basic(self):
        from ferro_ta.data.chunked import stitch_chunks

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        result = stitch_chunks([a, b])
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])

    def test_chunk_apply_sma_matches_full(self):
        """chunk_apply(SMA, …) should produce the same result as SMA on the full series."""
        from ferro_ta import SMA
        from ferro_ta.data.chunked import chunk_apply

        close = _make_close(500)
        full_out = np.asarray(SMA(close, timeperiod=20), dtype=np.float64)
        chunked_out = chunk_apply(SMA, close, chunk_size=100, overlap=30, timeperiod=20)

        # Compare non-NaN region
        valid = ~np.isnan(full_out)
        np.testing.assert_allclose(
            chunked_out[valid],
            full_out[valid],
            rtol=1e-10,
            err_msg="chunk_apply SMA must match full SMA for non-NaN bars",
        )

    def test_chunk_apply_output_length(self):
        from ferro_ta import EMA
        from ferro_ta.data.chunked import chunk_apply

        close = _make_close(300)
        out = chunk_apply(EMA, close, chunk_size=80, overlap=20, timeperiod=10)
        assert len(out) == len(close)


# ===========================================================================
# Regime detection
# ===========================================================================


class TestRegimeDetection:
    def test_regime_adx_shape(self):
        from ferro_ta import ADX
        from ferro_ta.analysis.regime import regime_adx

        o, h, l, c, v = _make_ohlcv(200)
        adx = np.asarray(ADX(h, l, c, timeperiod=14), dtype=np.float64)
        labels = regime_adx(adx, threshold=25.0)
        assert labels.shape == (200,)

    def test_regime_adx_values_valid(self):
        from ferro_ta import ADX
        from ferro_ta.analysis.regime import regime_adx

        o, h, l, c, v = _make_ohlcv(200)
        adx = np.asarray(ADX(h, l, c, timeperiod=14), dtype=np.float64)
        labels = regime_adx(adx, threshold=25.0)
        # Values must be -1, 0, or 1
        assert set(labels).issubset({-1, 0, 1})

    def test_regime_adx_nan_bars_are_minus_one(self):
        from ferro_ta.analysis.regime import regime_adx

        adx = np.full(20, np.nan)
        adx[15:] = 30.0  # last 5 are trend
        labels = regime_adx(adx, threshold=25.0)
        assert np.all(labels[:15] == -1)
        assert np.all(labels[15:] == 1)

    def test_regime_combined_shape(self):
        from ferro_ta import ADX, ATR
        from ferro_ta.analysis.regime import regime_combined

        o, h, l, c, v = _make_ohlcv(200)
        adx = np.asarray(ADX(h, l, c, timeperiod=14), dtype=np.float64)
        atr = np.asarray(ATR(h, l, c, timeperiod=14), dtype=np.float64)
        labels = regime_combined(
            adx, atr, c, adx_threshold=25.0, atr_pct_threshold=0.005
        )
        assert labels.shape == (200,)
        assert set(labels).issubset({-1, 0, 1})

    def test_regime_high_level_adx(self):
        from ferro_ta.analysis.regime import regime

        o, h, l, c, v = _make_ohlcv(200)
        labels = regime((o, h, l, c, v), method="adx", adx_threshold=25.0)
        assert labels.shape == (200,)
        assert set(labels).issubset({-1, 0, 1})

    def test_regime_high_level_combined(self):
        from ferro_ta.analysis.regime import regime

        o, h, l, c, v = _make_ohlcv(200)
        labels = regime((o, h, l, c, v), method="combined")
        assert labels.shape == (200,)

    def test_regime_unknown_method_raises(self):
        from ferro_ta.analysis.regime import regime

        o, h, l, c, v = _make_ohlcv(50)
        with pytest.raises(ValueError):
            regime((o, h, l, c, v), method="unknown")


class TestStructuralBreaks:
    def test_detect_breaks_cusum_shape(self):
        from ferro_ta.analysis.regime import detect_breaks_cusum

        series = _make_close(200)
        mask = detect_breaks_cusum(series, window=20, threshold=3.0, slack=0.5)
        assert mask.shape == (200,)

    def test_detect_breaks_cusum_fires_near_break(self):
        """CUSUM should detect a level shift."""
        from ferro_ta.analysis.regime import detect_breaks_cusum

        rng = np.random.default_rng(99)
        s1 = rng.normal(0, 1, 100)
        s2 = rng.normal(10, 1, 100)  # large level shift
        series = np.concatenate([s1, s2])
        mask = detect_breaks_cusum(series, window=20, threshold=2.0, slack=0.3)
        # Should fire somewhere near the shift
        assert mask[100:130].any()

    def test_rolling_variance_break_shape(self):
        from ferro_ta.analysis.regime import rolling_variance_break

        series = _make_close(200)
        mask = rolling_variance_break(
            series, short_window=10, long_window=50, threshold=2.0
        )
        assert mask.shape == (200,)

    def test_structural_breaks_cusum(self):
        from ferro_ta.analysis.regime import structural_breaks

        series = _make_close(200)
        mask = structural_breaks(series, method="cusum")
        assert mask.shape == (200,)

    def test_structural_breaks_variance(self):
        from ferro_ta.analysis.regime import structural_breaks

        series = _make_close(200)
        mask = structural_breaks(series, method="variance")
        assert mask.shape == (200,)

    def test_structural_breaks_unknown_method_raises(self):
        from ferro_ta.analysis.regime import structural_breaks

        with pytest.raises(ValueError):
            structural_breaks(_make_close(50), method="xyz")


# ===========================================================================
# Performance attribution
# ===========================================================================


class TestTradeStats:
    def test_basic_stats(self):
        from ferro_ta.analysis.attribution import trade_stats

        pnl = np.array([10.0, -5.0, 8.0, -3.0, 15.0, -2.0])
        hold = np.array([5.0, 3.0, 7.0, 2.0, 10.0, 1.0])
        ts = trade_stats(pnl, hold)
        assert ts.n_trades == 6
        assert abs(ts.win_rate - 0.5) < 1e-10  # 3 wins out of 6
        assert ts.avg_win > 0
        assert ts.avg_loss < 0
        assert ts.profit_factor > 0
        assert ts.avg_hold_bars == pytest.approx(4.67, abs=0.01)

    def test_all_wins(self):
        from ferro_ta.analysis.attribution import trade_stats

        pnl = np.array([5.0, 10.0, 3.0])
        ts = trade_stats(pnl)
        assert ts.win_rate == 1.0
        assert ts.avg_loss == 0.0
        assert ts.profit_factor == float("inf")

    def test_all_losses(self):
        from ferro_ta.analysis.attribution import trade_stats

        pnl = np.array([-5.0, -3.0])
        ts = trade_stats(pnl)
        assert ts.win_rate == 0.0
        assert ts.avg_win == 0.0
        assert ts.profit_factor == 0.0

    def test_empty_raises(self):
        from ferro_ta.analysis.attribution import trade_stats

        with pytest.raises(Exception):
            trade_stats(np.array([]))

    def test_to_dict(self):
        from ferro_ta.analysis.attribution import trade_stats

        pnl = np.array([1.0, -1.0])
        ts = trade_stats(pnl)
        d = ts.to_dict()
        assert "win_rate" in d
        assert "profit_factor" in d


class TestFromBacktest:
    def test_from_backtest_returns_arrays(self):
        from ferro_ta.analysis.attribution import from_backtest
        from ferro_ta.analysis.backtest import backtest

        close = _make_close(200)
        result = backtest(close, strategy="rsi_30_70")
        pnl, hold = from_backtest(result)
        assert isinstance(pnl, np.ndarray)
        assert isinstance(hold, np.ndarray)
        assert len(pnl) == len(hold)
        # n_trades counts position *changes* (entries + exits);
        # from_backtest counts round-trips (position runs), so len(pnl) <= n_trades
        assert len(pnl) <= result.n_trades
        # Each hold duration should be >= 1
        if len(hold) > 0:
            assert np.all(hold >= 1)

    def test_from_backtest_no_trades(self):
        from ferro_ta.analysis.attribution import from_backtest
        from ferro_ta.analysis.backtest import BacktestResult

        n = 50
        result = BacktestResult(
            signals=np.zeros(n),
            positions=np.zeros(n),
            bar_returns=np.zeros(n),
            strategy_returns=np.zeros(n),
            equity=np.ones(n),
        )
        pnl, hold = from_backtest(result)
        assert len(pnl) == 0


class TestAttribution:
    def test_attribution_by_signal_basic(self):
        from ferro_ta.analysis.attribution import attribution_by_signal

        ret = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        labels = np.array([0, 0, 1, 1, -1], dtype=np.int64)
        contrib = attribution_by_signal(ret, labels)
        assert isinstance(contrib, dict)
        assert "signal_0" in contrib
        assert "signal_1" in contrib
        assert abs(contrib["signal_0"] - 0.03) < 1e-10  # 0.01 + 0.02
        assert abs(contrib["signal_1"] - 0.02) < 1e-10  # -0.01 + 0.03

    def test_attribution_by_month_returns_dict(self):
        from ferro_ta.analysis.attribution import attribution_by_month

        ret = RNG.normal(0, 0.01, 252)
        contrib = attribution_by_month(ret)
        assert isinstance(contrib, dict)
        assert len(contrib) > 0

    def test_attribution_by_month_sum_close_to_total(self):
        """Sum of monthly contributions should approximate total strategy return."""
        from ferro_ta.analysis.attribution import attribution_by_month

        ret = RNG.normal(0, 0.01, 252)
        contrib = attribution_by_month(ret)
        total_monthly = sum(contrib.values())
        total_direct = float(np.sum(ret))
        assert abs(total_monthly - total_direct) < 1e-8


# ===========================================================================
# Dashboard (smoke tests, no display)
# ===========================================================================


class TestDashboard:
    def test_streamlit_app_import(self):
        """Module should import without errors even if streamlit not installed."""
        try:
            from ferro_ta.tools import dashboard  # noqa: F401
        except ImportError:
            pytest.skip("dashboard module not importable")

    def test_indicator_widget_raises_without_ipywidgets(self, monkeypatch):
        from ferro_ta import SMA
        from ferro_ta.tools.dashboard import indicator_widget

        close = _make_close(50)
        # If ipywidgets not installed, should raise ImportError
        import sys

        fake_modules = dict(sys.modules)
        fake_modules["ipywidgets"] = None  # type: ignore[assignment]
        fake_modules["matplotlib"] = None  # type: ignore[assignment]
        fake_modules["matplotlib.pyplot"] = None  # type: ignore[assignment]
        monkeypatch.setattr(sys, "modules", fake_modules)
        with pytest.raises((ImportError, TypeError)):
            indicator_widget(close, SMA, "timeperiod", range(5, 10))


# ===========================================================================
# Web API (unit test with TestClient if fastapi is available)
# ===========================================================================


class TestWebAPI:
    @pytest.fixture(scope="class")
    def client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")
        import os
        import sys

        # Insert project root so that `api.main` is importable
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            from api.main import app
        except ImportError:
            pytest.skip("api/main.py not importable")
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_sma_endpoint(self, client):
        close = list(np.linspace(100, 110, 30))
        resp = client.post("/indicators/sma", json={"close": close, "timeperiod": 5})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert len(result) == 30
        assert result[0] is None  # warm-up is null

    def test_ema_endpoint(self, client):
        close = list(np.linspace(100, 110, 30))
        resp = client.post("/indicators/ema", json={"close": close, "timeperiod": 5})
        assert resp.status_code == 200
        assert len(resp.json()["result"]) == 30

    def test_rsi_endpoint(self, client):
        close = list(np.linspace(100, 110, 30))
        resp = client.post("/indicators/rsi", json={"close": close, "timeperiod": 14})
        assert resp.status_code == 200

    def test_macd_endpoint(self, client):
        close = list(np.linspace(100, 120, 60))
        resp = client.post("/indicators/macd", json={"close": close})
        assert resp.status_code == 200
        keys = resp.json()["result"].keys()
        assert {"macd", "signal", "hist"} == set(keys)

    def test_bbands_endpoint(self, client):
        close = list(np.linspace(100, 110, 30))
        resp = client.post("/indicators/bbands", json={"close": close, "timeperiod": 5})
        assert resp.status_code == 200
        keys = resp.json()["result"].keys()
        assert {"upper", "middle", "lower"} == set(keys)

    def test_backtest_endpoint(self, client):
        close = list(
            np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 100)) * 100
        )
        resp = client.post(
            "/backtest",
            json={"close": close, "strategy": "rsi_30_70"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "final_equity" in body
        assert "n_trades" in body

    def test_unknown_strategy_returns_422(self, client):
        close = list(np.linspace(100, 110, 30))
        resp = client.post(
            "/backtest",
            json={"close": close, "strategy": "no_such_strategy"},
        )
        assert resp.status_code == 422

    def test_too_short_series_returns_422(self, client):
        resp = client.post("/indicators/sma", json={"close": [100.0], "timeperiod": 5})
        assert resp.status_code == 422


# ===========================================================================
# Benchmark suite sanity
# ===========================================================================


class TestBenchmarkSuite:
    def test_canonical_fixture_exists(self):
        import pathlib

        fixture = (
            pathlib.Path(__file__).parent.parent.parent
            / "benchmarks"
            / "fixtures"
            / "canonical_ohlcv.npz"
        )
        assert fixture.exists(), f"Canonical fixture not found: {fixture}"

    def test_canonical_fixture_loadable(self):
        import pathlib

        fixture = (
            pathlib.Path(__file__).parent.parent.parent
            / "benchmarks"
            / "fixtures"
            / "canonical_ohlcv.npz"
        )
        if not fixture.exists():
            pytest.skip("Canonical fixture not found")
        data = np.load(fixture)
        for key in ["open", "high", "low", "close", "volume"]:
            assert key in data.files, f"Missing key '{key}' in fixture"
        assert len(data["close"]) == 2000

    def test_benchmark_indicators_run(self):
        import pathlib

        fixture = (
            pathlib.Path(__file__).parent.parent.parent
            / "benchmarks"
            / "fixtures"
            / "canonical_ohlcv.npz"
        )
        if not fixture.exists():
            pytest.skip("Canonical fixture not found")
        import ferro_ta as ft

        data = np.load(fixture)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        out_sma = np.asarray(ft.SMA(close, timeperiod=20))
        out_rsi = np.asarray(ft.RSI(close, timeperiod=14))
        out_atr = np.asarray(ft.ATR(high, low, close, timeperiod=14))

        assert len(out_sma) == len(close)
        assert len(out_rsi) == len(close)
        assert len(out_atr) == len(close)
        # Last value should be finite
        assert np.isfinite(out_sma[-1])
        assert np.isfinite(out_rsi[-1])
        assert np.isfinite(out_atr[-1])


# ===========================================================================
# Options / IV helpers
# ===========================================================================


class TestIVRank:
    def test_basic_shape(self):
        from ferro_ta.analysis.options import iv_rank

        iv = _make_close(100)
        result = iv_rank(iv, window=20)
        assert result.shape == (100,)

    def test_warmup_nan(self):
        from ferro_ta.analysis.options import iv_rank

        iv = _make_close(50)
        result = iv_rank(iv, window=10)
        assert np.all(np.isnan(result[:9]))
        assert not np.isnan(result[9])

    def test_values_in_0_1(self):
        from ferro_ta.analysis.options import iv_rank

        iv = _make_close(100)
        result = iv_rank(iv, window=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_max_value_is_1(self):
        from ferro_ta.analysis.options import iv_rank

        # The maximum of a window should produce rank = 1
        iv = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = iv_rank(iv, window=5)
        assert result[4] == pytest.approx(1.0)

    def test_min_value_is_0(self):
        from ferro_ta.analysis.options import iv_rank

        iv = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        result = iv_rank(iv, window=5)
        assert result[4] == pytest.approx(0.0)

    def test_empty_raises(self):
        from ferro_ta.analysis.options import iv_rank

        with pytest.raises(Exception):
            iv_rank(np.array([]), window=5)

    def test_window_1(self):
        from ferro_ta.analysis.options import iv_rank

        iv = np.array([10.0, 20.0, 30.0])
        result = iv_rank(iv, window=1)
        # With window=1, all values are equal to min=max, so rank=0
        assert np.all(result == 0.0)

    def test_invalid_window_raises(self):
        from ferro_ta.analysis.options import iv_rank

        with pytest.raises(Exception):
            iv_rank(np.array([1.0, 2.0]), window=0)

    def test_flat_series(self):
        from ferro_ta.analysis.options import iv_rank

        iv = np.ones(30) * 25.0
        result = iv_rank(iv, window=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid == 0.0)


class TestIVPercentile:
    def test_basic_shape(self):
        from ferro_ta.analysis.options import iv_percentile

        iv = _make_close(100)
        result = iv_percentile(iv, window=20)
        assert result.shape == (100,)

    def test_warmup_nan(self):
        from ferro_ta.analysis.options import iv_percentile

        iv = _make_close(50)
        result = iv_percentile(iv, window=10)
        assert np.all(np.isnan(result[:9]))

    def test_values_in_0_1(self):
        from ferro_ta.analysis.options import iv_percentile

        iv = _make_close(100)
        result = iv_percentile(iv, window=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_empty_raises(self):
        from ferro_ta.analysis.options import iv_percentile

        with pytest.raises(Exception):
            iv_percentile(np.array([]), window=5)

    def test_known_value(self):
        from ferro_ta.analysis.options import iv_percentile

        iv = np.array([10.0, 20.0, 30.0, 15.0, 22.0])
        result = iv_percentile(iv, window=3)
        # At index 2: window=[10,20,30], current=30. All 3 <= 30 → 3/3 = 1.0
        assert result[2] == pytest.approx(1.0)
        # At index 3: window=[20,30,15], current=15. Only 15 <= 15 → 1/3
        assert result[3] == pytest.approx(1.0 / 3.0)


class TestIVZScore:
    def test_basic_shape(self):
        from ferro_ta.analysis.options import iv_zscore

        iv = _make_close(100)
        result = iv_zscore(iv, window=20)
        assert result.shape == (100,)

    def test_warmup_nan(self):
        from ferro_ta.analysis.options import iv_zscore

        iv = _make_close(50)
        result = iv_zscore(iv, window=10)
        assert np.all(np.isnan(result[:9]))

    def test_flat_is_nan(self):
        from ferro_ta.analysis.options import iv_zscore

        # Flat series has std=0, so z-score should be NaN
        iv = np.ones(30) * 20.0
        result = iv_zscore(iv, window=10)
        valid = result[~np.isnan(result)]
        assert len(valid) == 0 or np.all(np.isnan(valid))

    def test_empty_raises(self):
        from ferro_ta.analysis.options import iv_zscore

        with pytest.raises(Exception):
            iv_zscore(np.array([]), window=5)

    def test_known_value(self):
        from ferro_ta.analysis.options import iv_zscore

        iv = np.array([10.0, 20.0, 30.0])
        result = iv_zscore(iv, window=3)
        # mean=20, std=std([10,20,30],ddof=0)=8.165...
        expected = (30.0 - 20.0) / np.std([10.0, 20.0, 30.0], ddof=0)
        assert result[2] == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# Agentic tools and workflow
# ===========================================================================


class TestComputeIndicator:
    def test_sma_basic(self):
        from ferro_ta.tools import compute_indicator

        close = np.linspace(100, 110, 20)
        result = compute_indicator("SMA", close, timeperiod=5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20,)

    def test_rsi_basic(self):
        from ferro_ta.tools import compute_indicator

        close = _make_close(100)
        result = compute_indicator("RSI", close, timeperiod=14)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)

    def test_bbands_multi_output(self):
        from ferro_ta.tools import compute_indicator

        close = _make_close(50)
        result = compute_indicator("BBANDS", close, timeperiod=10)
        assert isinstance(result, dict)
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result

    def test_macd_multi_output(self):
        from ferro_ta.tools import compute_indicator

        close = _make_close(100)
        result = compute_indicator(
            "MACD", close, fastperiod=5, slowperiod=10, signalperiod=3
        )
        assert isinstance(result, dict)
        assert "macd" in result
        assert "signal" in result
        assert "hist" in result

    def test_unknown_indicator_raises(self):
        from ferro_ta.tools import compute_indicator

        with pytest.raises(Exception):
            compute_indicator("NO_SUCH_INDICATOR", np.ones(20))


class TestRunBacktest:
    def test_basic_result_shape(self):
        from ferro_ta.tools import run_backtest

        close = _make_close(200)
        summary = run_backtest("rsi_30_70", close)
        assert isinstance(summary, dict)
        assert "final_equity" in summary
        assert "n_trades" in summary
        assert "n_bars" in summary
        assert "equity" in summary
        assert "signals" in summary
        assert "max_drawdown" in summary
        assert summary["n_bars"] == 200
        assert isinstance(summary["final_equity"], float)

    def test_equity_list(self):
        from ferro_ta.tools import run_backtest

        close = _make_close(100)
        summary = run_backtest("rsi_30_70", close)
        assert isinstance(summary["equity"], list)
        assert len(summary["equity"]) == 100

    def test_sma_crossover_strategy(self):
        from ferro_ta.tools import run_backtest

        close = _make_close(200)
        summary = run_backtest("sma_crossover", close, fast=5, slow=20)
        assert "final_equity" in summary

    def test_macd_crossover_strategy(self):
        from ferro_ta.tools import run_backtest

        close = _make_close(200)
        summary = run_backtest("macd_crossover", close)
        assert "final_equity" in summary

    def test_unknown_strategy_raises(self):
        from ferro_ta.tools import run_backtest

        with pytest.raises(Exception):
            run_backtest("no_such_strategy", _make_close(100))

    def test_max_drawdown_non_negative(self):
        from ferro_ta.tools import run_backtest

        close = _make_close(200)
        summary = run_backtest("rsi_30_70", close)
        assert summary["max_drawdown"] >= 0.0


class TestListIndicators:
    def test_returns_list(self):
        from ferro_ta.tools import list_indicators

        names = list_indicators()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_contains_sma_rsi(self):
        from ferro_ta.tools import list_indicators

        names = list_indicators()
        assert "SMA" in names
        assert "RSI" in names

    def test_sorted(self):
        from ferro_ta.tools import list_indicators

        names = list_indicators()
        assert names == sorted(names)


class TestDescribeIndicator:
    def test_returns_string(self):
        from ferro_ta.tools import describe_indicator

        desc = describe_indicator("SMA")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_unknown_raises(self):
        from ferro_ta.tools import describe_indicator

        with pytest.raises(Exception):
            describe_indicator("NO_SUCH_INDICATOR")


class TestWorkflow:
    def test_basic_indicators(self):
        from ferro_ta.tools.workflow import Workflow

        close = _make_close(200)
        result = (
            Workflow()
            .add_indicator("sma_20", "SMA", timeperiod=20)
            .add_indicator("rsi_14", "RSI", timeperiod=14)
            .run(close)
        )
        assert "sma_20" in result
        assert "rsi_14" in result
        assert result["sma_20"].shape == (200,)
        assert result["rsi_14"].shape == (200,)

    def test_with_strategy(self):
        from ferro_ta.tools.workflow import Workflow

        close = _make_close(200)
        result = (
            Workflow()
            .add_indicator("rsi_14", "RSI", timeperiod=14)
            .add_strategy("rsi_30_70")
            .run(close)
        )
        assert "backtest" in result
        assert "final_equity" in result["backtest"]

    def test_with_alert(self):
        from ferro_ta.tools.workflow import Workflow

        close = _make_close(200)
        result = (
            Workflow()
            .add_indicator("rsi_14", "RSI", timeperiod=14)
            .add_alert("rsi_14", level=30.0, direction=-1)
            .run(close)
        )
        assert "rsi_14" in result
        # Alert key should be present
        alert_keys = [k for k in result if k.startswith("alert_")]
        assert len(alert_keys) > 0

    def test_empty_workflow(self):
        from ferro_ta.tools.workflow import Workflow

        close = _make_close(50)
        result = Workflow().run(close)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_multi_output_indicator(self):
        from ferro_ta.tools.workflow import Workflow

        close = _make_close(100)
        result = Workflow().add_indicator("bb", "BBANDS", timeperiod=10).run(close)
        assert "bb" in result
        # BBANDS returns dict from compute_indicator
        assert isinstance(result["bb"], dict)


class TestRunPipeline:
    def test_basic_pipeline(self):
        from ferro_ta.tools.workflow import run_pipeline

        close = _make_close(200)
        result = run_pipeline(
            close,
            indicators={
                "sma_20": {"name": "SMA", "timeperiod": 20},
                "rsi_14": {"name": "RSI", "timeperiod": 14},
            },
        )
        assert "sma_20" in result
        assert "rsi_14" in result

    def test_with_strategy(self):
        from ferro_ta.tools.workflow import run_pipeline

        close = _make_close(200)
        result = run_pipeline(
            close,
            indicators={"rsi_14": {"name": "RSI", "timeperiod": 14}},
            strategy="rsi_30_70",
        )
        assert "backtest" in result

    def test_no_indicators(self):
        from ferro_ta.tools.workflow import run_pipeline

        close = _make_close(100)
        result = run_pipeline(close)
        assert isinstance(result, dict)

    def test_with_alert(self):
        from ferro_ta.tools.workflow import run_pipeline

        close = _make_close(200)
        result = run_pipeline(
            close,
            indicators={"rsi_14": {"name": "RSI", "timeperiod": 14}},
            alert_indicator="rsi_14",
            alert_level=30.0,
            alert_direction=-1,
        )
        assert "rsi_14" in result
        alert_keys = [k for k in result if k.startswith("alert_")]
        assert len(alert_keys) > 0


# ===========================================================================
# MCP server
# ===========================================================================


class TestMCPListTools:
    def test_list_tools_returns_dict(self):
        from ferro_ta.mcp import handle_list_tools

        result = handle_list_tools()
        assert isinstance(result, dict)
        assert "tools" in result

    def test_list_tools_has_required_tools(self):
        from ferro_ta.mcp import handle_list_tools

        result = handle_list_tools()
        names = [t["name"] for t in result["tools"]]
        assert len(names) > 250
        for expected in (
            "sma",
            "ema",
            "rsi",
            "macd",
            "backtest",
            "SMA",
            "compute_indicator",
            "about",
            "check_cross",
            "TickAggregator",
            "call_instance_method",
            "call_stored_callable",
            "delete_instance",
        ):
            assert expected in names, f"Expected tool '{expected}' not found"

    def test_each_tool_has_schema(self):
        from ferro_ta.mcp import handle_list_tools

        result = handle_list_tools()
        for tool in result["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


class TestMCPCallTool:
    def test_sma_call(self):
        from ferro_ta.mcp import handle_call_tool

        close = list(np.linspace(100, 110, 30))
        result = handle_call_tool("sma", {"close": close, "timeperiod": 5})
        assert "content" in result
        import json

        payload = json.loads(result["content"][0]["text"])
        assert len(payload) == 30

    def test_ema_call(self):
        from ferro_ta.mcp import handle_call_tool

        close = list(np.linspace(100, 110, 30))
        result = handle_call_tool("ema", {"close": close, "timeperiod": 5})
        assert "content" in result

    def test_rsi_call(self):
        from ferro_ta.mcp import handle_call_tool

        close = list(_make_close(50))
        result = handle_call_tool("rsi", {"close": close, "timeperiod": 14})
        assert "content" in result

    def test_macd_call(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        close = list(_make_close(100))
        result = handle_call_tool("macd", {"close": close})
        assert "content" in result
        payload = json.loads(result["content"][0]["text"])
        assert "macd" in payload

    def test_backtest_call(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        close = list(_make_close(200))
        result = handle_call_tool("backtest", {"close": close, "strategy": "rsi_30_70"})
        assert "content" in result
        payload = json.loads(result["content"][0]["text"])
        assert "final_equity" in payload
        assert "n_trades" in payload

    def test_top_level_sma_call(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        close = list(np.linspace(100, 110, 30))
        result = handle_call_tool("SMA", {"close": close, "timeperiod": 5})
        payload = json.loads(result["content"][0]["text"])
        assert len(payload) == 30

    def test_compute_indicator_call(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        close = list(_make_close(100))
        result = handle_call_tool(
            "compute_indicator",
            {
                "name": "MACD",
                "args": [close],
            },
        )
        payload = json.loads(result["content"][0]["text"])
        assert "macd" in payload

    def test_about_call(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        result = handle_call_tool("about", {})
        payload = json.loads(result["content"][0]["text"])
        assert payload["indicator_count"] >= 200
        assert payload["method_count"] >= 400

    def test_check_cross_call(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        result = handle_call_tool(
            "check_cross",
            {
                "fast": [1.0, 2.0, 3.0, 2.0, 1.0],
                "slow": [2.0, 2.0, 2.0, 2.0, 2.0],
            },
        )
        payload = json.loads(result["content"][0]["text"])
        assert len(payload) == 5

    def test_list_indicators_call(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        result = handle_call_tool("list_indicators", {})
        assert "content" in result
        payload = json.loads(result["content"][0]["text"])
        assert isinstance(payload, list)
        assert "SMA" in payload

    def test_describe_indicator_call(self):
        from ferro_ta.mcp import handle_call_tool

        result = handle_call_tool("describe_indicator", {"name": "SMA"})
        assert "content" in result
        text = result["content"][0]["text"]
        assert isinstance(text, str)
        assert len(text) > 0

    def test_unknown_tool_returns_error(self):
        from ferro_ta.mcp import handle_call_tool

        result = handle_call_tool("no_such_tool", {})
        assert result.get("isError") is True

    def test_tool_error_handling(self):
        from ferro_ta.mcp import handle_call_tool

        # Pass an invalid series to trigger an error
        result = handle_call_tool("sma", {"close": [], "timeperiod": 5})
        # Should return error content, not raise
        assert "content" in result or "isError" in result

    def test_backtest_unknown_strategy(self):
        from ferro_ta.mcp import handle_call_tool

        close = list(_make_close(100))
        result = handle_call_tool(
            "backtest", {"close": close, "strategy": "no_strategy"}
        )
        assert result.get("isError") is True or "content" in result

    def test_tick_aggregator_instance_lifecycle(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        created = json.loads(
            handle_call_tool("TickAggregator", {"rule": "tick:2"})["content"][0]["text"]
        )
        instance_id = created["instance_id"]

        described = json.loads(
            handle_call_tool(
                "describe_instance", {"instance_id": instance_id}
            )["content"][0]["text"]
        )
        method_names = [item["name"] for item in described["methods"]]
        assert "aggregate" in method_names

        aggregated = json.loads(
            handle_call_tool(
                "call_instance_method",
                {
                    "instance_id": instance_id,
                    "method": "aggregate",
                    "args": [
                        {
                            "price": [1.0, 2.0, 3.0, 4.0],
                            "size": [1.0, 1.0, 1.0, 1.0],
                        }
                    ],
                },
            )["content"][0]["text"]
        )
        assert "open" in aggregated
        assert "close" in aggregated

        deleted = json.loads(
            handle_call_tool(
                "delete_instance", {"instance_id": instance_id}
            )["content"][0]["text"]
        )
        assert deleted["deleted"] is True

    def test_stored_callable_can_be_invoked(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        wrapped = json.loads(
            handle_call_tool(
                "traced", {"func": {"callable": "SMA"}}
            )["content"][0]["text"]
        )
        instance_id = wrapped["instance_id"]

        called = json.loads(
            handle_call_tool(
                "call_stored_callable",
                {
                    "instance_id": instance_id,
                    "args": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                    "kwargs": {"timeperiod": 3},
                },
            )["content"][0]["text"]
        )
        assert len(called) == 5

        handle_call_tool("delete_instance", {"instance_id": instance_id})

    def test_benchmark_accepts_callable_reference(self):
        import json

        from ferro_ta.mcp import handle_call_tool

        result = handle_call_tool(
            "benchmark",
            {
                "func": {"callable": "SMA"},
                "args": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "kwargs": {"timeperiod": 3},
                "n": 2,
                "warmup": 0,
            },
        )
        payload = json.loads(result["content"][0]["text"])
        assert payload["n"] == 2.0
        assert "mean_ms" in payload


class TestMCPServer:
    def test_create_server_requires_mcp_dependency(self, monkeypatch):
        import ferro_ta.mcp as mcp_mod

        real_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name.startswith("mcp"):
                raise ImportError("No module named 'mcp'")
            return real_import_module(name, package)

        mcp_mod.create_server.cache_clear()
        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        with pytest.raises(RuntimeError, match='pip install "ferro-ta\\[mcp\\]"'):
            mcp_mod.create_server()

    def test_main_entrypoint_invokes_run_server(self, monkeypatch):
        import ferro_ta.mcp as mcp_mod

        calls: list[str] = []

        monkeypatch.setattr(mcp_mod, "run_server", lambda: calls.append("called"))
        runpy.run_module("ferro_ta.mcp.__main__", run_name="__main__")

        assert calls == ["called"]

    def test_create_server_registers_generated_tools(self):
        import ferro_ta.mcp as mcp_mod

        server = mcp_mod.create_server()
        tool_names = [tool.name for tool in server._tool_manager.list_tools()]

        assert "SMA" in tool_names
        assert "TickAggregator" in tool_names
        assert "call_instance_method" in tool_names
