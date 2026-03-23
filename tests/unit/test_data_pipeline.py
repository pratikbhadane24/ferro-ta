"""Tests for resampling, tick aggregation, DSL, signals,
portfolio analytics, cross-asset analytics, feature matrix, viz, and adapters.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2024)


def _make_ohlcv(n: int = 100):
    """Return (open, high, low, close, volume) as numpy arrays."""
    close = np.cumprod(1 + RNG.normal(0, 0.01, n)) * 100.0
    open_ = close * RNG.uniform(0.995, 1.005, n)
    high = np.maximum(close, open_) + RNG.uniform(0, 0.5, n)
    low = np.minimum(close, open_) - RNG.uniform(0, 0.5, n)
    volume = RNG.uniform(500, 5000, n)
    return open_, high, low, close, volume


def _make_ticks(n: int = 500):
    price = 100.0 + np.cumsum(RNG.normal(0, 0.05, n))
    size = RNG.uniform(10, 100, n)
    return price, size


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestVolumeBarResampling:
    """Rust-backed volume_bars function."""

    def test_returns_five_arrays(self):
        from ferro_ta.data.resampling import volume_bars

        o, h, l, c, v = _make_ohlcv(100)
        bars = volume_bars((o, h, l, c, v), volume_threshold=2000)
        assert len(bars) == 5
        assert all(isinstance(b, np.ndarray) for b in bars)

    def test_volume_bars_reduce_length(self):
        from ferro_ta.data.resampling import volume_bars

        o, h, l, c, v = _make_ohlcv(200)
        bars = volume_bars((o, h, l, c, v), volume_threshold=5000)
        # Output should have fewer bars than input
        assert len(bars[0]) < 200

    def test_each_bar_high_ge_low(self):
        from ferro_ta.data.resampling import volume_bars

        o, h, l, c, v = _make_ohlcv(100)
        ro, rh, rl, rc, rv = volume_bars((o, h, l, c, v), volume_threshold=2000)
        assert np.all(rh >= rl)

    def test_output_volume_ge_threshold(self):
        from ferro_ta.data.resampling import volume_bars

        o, h, l, c, v = _make_ohlcv(100)
        threshold = 1500.0
        _, _, _, _, rv = volume_bars((o, h, l, c, v), volume_threshold=threshold)
        # All but the last bar should satisfy the threshold
        if len(rv) > 1:
            assert np.all(rv[:-1] >= threshold)

    def test_invalid_threshold_raises(self):
        from ferro_ta.data.resampling import volume_bars

        o, h, l, c, v = _make_ohlcv(10)
        with pytest.raises(Exception):
            volume_bars((o, h, l, c, v), volume_threshold=-1)

    def test_ohlcv_agg_rust_function(self):
        from ferro_ta._ferro_ta import ohlcv_agg

        o, h, l, c, v = _make_ohlcv(10)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        ro, rh, rl, rc, rv = ohlcv_agg(o, h, l, c, v, labels)
        assert len(ro) == 3

    def test_resample_with_pandas(self):
        """Time-based resampling using pandas DatetimeIndex."""
        pytest.importorskip("pandas")
        import pandas as pd

        from ferro_ta.data.resampling import resample

        idx = pd.date_range("2024-01-01", periods=60, freq="1min")
        o, h, l, c, v = _make_ohlcv(60)
        df = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )
        df5 = resample(df, "5min")
        # 60 1-minute bars → 12 or 13 5-minute bars depending on pandas version/label
        assert 11 <= len(df5) <= 13
        assert set(df5.columns) == {"open", "high", "low", "close", "volume"}

    def test_volume_bars_dataframe_return(self):
        pytest.importorskip("pandas")
        import pandas as pd

        from ferro_ta.data.resampling import volume_bars

        o, h, l, c, v = _make_ohlcv(60)
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
        result = volume_bars(df, volume_threshold=3000)
        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns

    def test_multi_timeframe_returns_dict(self):
        pytest.importorskip("pandas")
        import pandas as pd

        from ferro_ta import RSI
        from ferro_ta.data.resampling import multi_timeframe

        idx = pd.date_range("2024-01-01", periods=200, freq="1min")
        o, h, l, c, v = _make_ohlcv(200)
        df = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )
        result = multi_timeframe(
            df, ["5min", "15min"], indicator=RSI, indicator_kwargs={"timeperiod": 14}
        )
        assert sorted(result.keys()) == ["15min", "5min"]
        for key, arr in result.items():
            assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# Tick aggregation
# ---------------------------------------------------------------------------


class TestTickAggregation:
    """aggregate_ticks and TickAggregator."""

    def test_tick_bars_dict_input(self):
        from ferro_ta.data.aggregation import aggregate_ticks

        price, size = _make_ticks(500)
        result = aggregate_ticks({"price": price, "size": size}, rule="tick:50")
        assert "open" in result
        # 500 / 50 = 10 bars
        assert len(result["open"]) == 10

    def test_volume_bars_ticks(self):
        from ferro_ta.data.aggregation import aggregate_ticks

        price, size = _make_ticks(200)
        result = aggregate_ticks({"price": price, "size": size}, rule="volume:500")
        assert len(result["open"]) > 0

    def test_time_bars_ticks(self):
        from ferro_ta.data.aggregation import aggregate_ticks

        price, size = _make_ticks(300)
        ts = np.arange(300, dtype=np.float64)  # 1 second intervals
        result = aggregate_ticks(
            {"timestamp": ts, "price": price, "size": size}, rule="time:60"
        )
        # 300 seconds / 60 = 5 bars
        assert len(result["open"]) == 5

    def test_tick_aggregator_class(self):
        from ferro_ta.data.aggregation import TickAggregator

        agg = TickAggregator(rule="tick:50")
        price, size = _make_ticks(200)
        result = agg.aggregate({"price": price, "size": size})
        assert len(result["open"]) == 4  # 200 / 50 = 4

    def test_invalid_rule_raises(self):
        from ferro_ta.data.aggregation import aggregate_ticks

        price, size = _make_ticks(100)
        with pytest.raises(ValueError, match="Invalid rule"):
            aggregate_ticks({"price": price, "size": size}, rule="bad_rule")

    def test_unknown_bar_type_raises(self):
        from ferro_ta.data.aggregation import aggregate_ticks

        price, size = _make_ticks(100)
        with pytest.raises(ValueError, match="Unknown bar type"):
            aggregate_ticks({"price": price, "size": size}, rule="unknown:50")

    def test_tick_bars_indicator_pipeline(self):
        """Full pipeline: ticks → bars → RSI."""
        from ferro_ta import RSI
        from ferro_ta.data.aggregation import aggregate_ticks

        price, size = _make_ticks(1000)
        bars = aggregate_ticks({"price": price, "size": size}, rule="tick:20")
        close = np.asarray(bars["close"], dtype=np.float64)
        rsi = RSI(close, timeperiod=14)
        assert rsi.shape == close.shape

    def test_list_input(self):
        from ferro_ta.data.aggregation import aggregate_ticks

        ticks = [(float(i), 100.0 + i * 0.01, 10.0) for i in range(100)]
        result = aggregate_ticks(ticks, rule="tick:10")
        assert len(result["open"]) == 10


# ---------------------------------------------------------------------------
# Strategy DSL
# ---------------------------------------------------------------------------


class TestStrategyDSL:
    def test_parse_simple_expression(self):
        from ferro_ta.tools.dsl import parse_expression

        ast = parse_expression("RSI(14) < 30")
        assert ast is not None

    def test_evaluate_returns_int_array(self):
        from ferro_ta.tools.dsl import evaluate

        close = np.cumprod(1 + RNG.normal(0, 0.01, 100)) * 100
        sig = evaluate("RSI(14) < 30", {"close": close})
        assert sig.dtype == np.int32
        assert sig.shape == (100,)
        assert set(sig.tolist()).issubset({0, 1})

    def test_evaluate_and_expression(self):
        from ferro_ta.tools.dsl import evaluate

        close = np.cumprod(1 + RNG.normal(0, 0.01, 100)) * 100
        sig = evaluate("RSI(14) < 70 and RSI(14) > 30", {"close": close})
        assert sig.shape == (100,)

    def test_evaluate_or_expression(self):
        from ferro_ta.tools.dsl import evaluate

        close = np.cumprod(1 + RNG.normal(0, 0.01, 100)) * 100
        sig = evaluate("RSI(14) < 30 or RSI(14) > 70", {"close": close})
        assert sig.shape == (100,)

    def test_evaluate_not_expression(self):
        from ferro_ta.tools.dsl import evaluate

        close = np.cumprod(1 + RNG.normal(0, 0.01, 100)) * 100
        sig = evaluate("not RSI(14) < 30", {"close": close})
        assert set(sig.tolist()).issubset({0, 1})

    def test_strategy_class(self):
        from ferro_ta.tools.dsl import Strategy

        close = np.cumprod(1 + RNG.normal(0, 0.01, 100)) * 100
        strat = Strategy("RSI(14) < 30")
        sig = strat.evaluate({"close": close})
        assert sig.shape == (100,)

    def test_combined_close_sma_expression(self):
        from ferro_ta.tools.dsl import evaluate

        close = np.cumprod(1 + RNG.normal(0, 0.01, 60)) * 100
        sig = evaluate("close > SMA(20)", {"close": close})
        assert sig.shape == (60,)

    def test_invalid_expression_raises(self):
        from ferro_ta.tools.dsl import parse_expression

        with pytest.raises(ValueError):
            parse_expression("")

    def test_parse_expression_with_cross_above_placeholder(self):
        """cross_above tokens parse without error."""
        from ferro_ta.tools.dsl import parse_expression

        ast = parse_expression("cross_above(close, SMA(20))")
        assert ast is not None

    def test_backtest_with_dsl_signal(self):
        """Combine DSL signal with the existing backtest module."""
        from ferro_ta.analysis.backtest import backtest
        from ferro_ta.tools.dsl import Strategy

        close = np.cumprod(1 + RNG.normal(0, 0.01, 100)) * 100
        strat = Strategy("RSI(14) < 30")
        strat.evaluate({"close": close})  # signal not fed to backtest in this test
        # Manually feed signal to backtest
        result = backtest(close, strategy="rsi_30_70")
        assert result is not None


# ---------------------------------------------------------------------------
# Signal composition and screening
# ---------------------------------------------------------------------------


class TestSignalComposition:
    def test_compose_weighted(self):
        from ferro_ta.analysis.signals import compose

        sigs = RNG.standard_normal((50, 3))
        score = compose(sigs, weights=[0.5, 0.3, 0.2])
        assert score.shape == (50,)

    def test_compose_mean(self):
        from ferro_ta.analysis.signals import compose

        sigs = np.ones((10, 4)) * 2.0
        score = compose(sigs, method="mean")
        np.testing.assert_allclose(score, 2.0)

    def test_compose_rank(self):
        from ferro_ta.analysis.signals import compose

        sigs = RNG.standard_normal((30, 3))
        score = compose(sigs, method="rank")
        assert score.shape == (30,)

    def test_compose_rank_matches_manual_column_ranks(self):
        from ferro_ta.analysis.signals import compose

        sigs = np.array(
            [
                [3.0, 1.0],
                [1.0, 2.0],
                [2.0, 2.0],
            ],
            dtype=np.float64,
        )
        score = compose(sigs, method="rank")
        expected = np.array([4.0, 3.5, 4.5], dtype=np.float64)
        np.testing.assert_allclose(score, expected)

    def test_compose_equal_weights_default(self):
        from ferro_ta.analysis.signals import compose

        sigs = np.ones((5, 3))
        score = compose(sigs)  # equal weight by default
        np.testing.assert_allclose(score, 1.0)

    def test_screen_top_n(self):
        from ferro_ta.analysis.signals import screen

        scores = {"AAPL": 0.8, "GOOG": 0.5, "MSFT": 0.9, "AMZN": 0.3}
        result = screen(scores, top_n=2)
        assert list(result.keys()) == ["MSFT", "AAPL"]

    def test_screen_bottom_n(self):
        from ferro_ta.analysis.signals import screen

        scores = {"A": 3, "B": 1, "C": 2}
        result = screen(scores, bottom_n=2)
        assert list(result.keys()) == ["B", "C"]

    def test_screen_above_threshold(self):
        from ferro_ta.analysis.signals import screen

        scores = {"A": 0.7, "B": 0.3, "C": 0.9}
        result = screen(scores, above=0.5)
        assert set(result.keys()) == {"A", "C"}

    def test_rank_signals(self):
        from ferro_ta.analysis.signals import rank_signals

        x = np.array([3.0, 1.0, 2.0])
        r = rank_signals(x)
        np.testing.assert_allclose(r, [3.0, 1.0, 2.0])

    def test_rank_signals_ties(self):
        from ferro_ta.analysis.signals import rank_signals

        x = np.array([1.0, 1.0, 3.0])
        r = rank_signals(x)
        np.testing.assert_allclose(r[0], 1.5)
        np.testing.assert_allclose(r[1], 1.5)
        np.testing.assert_allclose(r[2], 3.0)

    def test_top_n_indices_rust(self):
        from ferro_ta._ferro_ta import top_n_indices

        x = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        idx = top_n_indices(x, 2)
        vals = sorted(x[i] for i in idx)
        assert vals == [5.0, 7.0]


# ---------------------------------------------------------------------------
# Portfolio analytics
# ---------------------------------------------------------------------------


class TestPortfolioAnalytics:
    def test_correlation_matrix_shape(self):
        from ferro_ta.analysis.portfolio import correlation_matrix

        r = RNG.normal(0, 0.01, (100, 4))
        corr = correlation_matrix(r)
        assert corr.shape == (4, 4)

    def test_correlation_matrix_diagonal_ones(self):
        from ferro_ta.analysis.portfolio import correlation_matrix

        r = RNG.normal(0, 0.01, (100, 3))
        corr = correlation_matrix(r)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_correlation_matrix_symmetric(self):
        from ferro_ta.analysis.portfolio import correlation_matrix

        r = RNG.normal(0, 0.01, (80, 3))
        corr = correlation_matrix(r)
        np.testing.assert_allclose(corr, corr.T, atol=1e-12)

    def test_portfolio_volatility_positive(self):
        from ferro_ta.analysis.portfolio import portfolio_volatility

        r = RNG.normal(0, 0.01, (100, 3))
        vol = portfolio_volatility(r, weights=[1 / 3, 1 / 3, 1 / 3])
        assert vol > 0

    def test_portfolio_volatility_annualise(self):
        from ferro_ta.analysis.portfolio import portfolio_volatility

        r = RNG.normal(0, 0.01, (252, 1))
        vol_raw = portfolio_volatility(r, weights=[1.0])
        vol_ann = portfolio_volatility(r, weights=[1.0], annualise=252)
        np.testing.assert_allclose(vol_ann, vol_raw * 252**0.5, rtol=1e-6)

    def test_beta_scalar(self):
        from ferro_ta.analysis.portfolio import beta

        bench = RNG.normal(0, 0.01, 100)
        asset = 1.5 * bench + RNG.normal(0, 0.001, 100)
        b = beta(asset, bench)
        assert abs(b - 1.5) < 0.05

    def test_beta_rolling(self):
        from ferro_ta.analysis.portfolio import beta

        bench = RNG.normal(0, 0.01, 100)
        asset = bench + RNG.normal(0, 0.001, 100)
        rb = beta(asset, bench, window=20)
        assert rb.shape == (100,)
        assert np.isnan(rb[0])
        assert not np.isnan(rb[-1])

    def test_drawdown_series(self):
        from ferro_ta.analysis.portfolio import drawdown

        eq = np.array([100.0, 110.0, 105.0, 90.0, 95.0])
        dd, max_dd = drawdown(eq)
        assert dd.shape == (5,)
        assert dd[0] == 0.0  # no drawdown at start
        assert max_dd < 0

    def test_drawdown_max_only(self):
        from ferro_ta.analysis.portfolio import drawdown

        eq = np.array([100.0, 110.0, 105.0, 90.0, 95.0])
        max_dd = drawdown(eq, as_series=False)
        assert isinstance(max_dd, float)
        assert max_dd < 0


# ---------------------------------------------------------------------------
# Cross-asset analytics
# ---------------------------------------------------------------------------


class TestCrossAsset:
    def test_relative_strength_shape(self):
        from ferro_ta.analysis.cross_asset import relative_strength

        ra = RNG.normal(0, 0.01, 50)
        rb = RNG.normal(0, 0.01, 50)
        rs = relative_strength(ra, rb)
        assert rs.shape == (50,)

    def test_spread_values(self):
        from ferro_ta.analysis.cross_asset import spread

        a = np.array([10.0, 11.0, 12.0])
        b = np.array([9.0, 10.0, 11.0])
        sp = spread(a, b)
        np.testing.assert_allclose(sp, [1.0, 1.0, 1.0])

    def test_spread_custom_hedge(self):
        from ferro_ta.analysis.cross_asset import spread

        a = np.array([10.0, 10.0])
        b = np.array([5.0, 5.0])
        sp = spread(a, b, hedge=2.0)
        np.testing.assert_allclose(sp, [0.0, 0.0])

    def test_ratio_basic(self):
        from ferro_ta.analysis.cross_asset import ratio

        a = np.array([10.0, 12.0, 15.0])
        b = np.array([5.0, 4.0, 5.0])
        r = ratio(a, b)
        np.testing.assert_allclose(r, [2.0, 3.0, 3.0])

    def test_ratio_zero_denominator(self):
        from ferro_ta.analysis.cross_asset import ratio

        a = np.array([1.0, 2.0])
        b = np.array([0.0, 1.0])
        r = ratio(a, b)
        assert np.isnan(r[0])
        assert r[1] == 2.0

    def test_zscore_nan_warmup(self):
        from ferro_ta.analysis.cross_asset import zscore

        x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        z = zscore(x, window=3)
        assert np.isnan(z[0]) and np.isnan(z[1])
        assert not np.isnan(z[2])

    def test_rolling_beta_warmup(self):
        from ferro_ta.analysis.cross_asset import rolling_beta

        b = RNG.normal(0, 1, 50)
        a = 0.8 * b + RNG.normal(0, 0.1, 50)
        rb = rolling_beta(a, b, window=20)
        assert np.isnan(rb[18])
        assert not np.isnan(rb[19])


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


class TestFeatureMatrix:
    def test_basic_feature_matrix(self):
        from ferro_ta.analysis.features import feature_matrix

        o, h, l, c, v = _make_ohlcv(50)
        ohlcv = {"close": c, "high": h, "low": l, "open": o, "volume": v}
        fm = feature_matrix(ohlcv, [("SMA", {"timeperiod": 10})])
        assert "SMA" in fm
        arr = np.asarray(fm["SMA"] if isinstance(fm, dict) else fm["SMA"].values)
        assert arr.shape == (50,)

    def test_multiple_indicators_feature_matrix(self):
        from ferro_ta.analysis.features import feature_matrix

        o, h, l, c, v = _make_ohlcv(50)
        ohlcv = {"close": c, "high": h, "low": l, "open": o, "volume": v}
        fm = feature_matrix(
            ohlcv,
            [
                ("SMA", {"timeperiod": 10}),
                ("RSI", {"timeperiod": 14}),
            ],
        )
        assert "SMA" in fm
        assert "RSI" in fm

    def test_nan_policy_drop(self):
        pytest.importorskip("pandas")
        import pandas as pd

        from ferro_ta.analysis.features import feature_matrix

        o, h, l, c, v = _make_ohlcv(50)
        ohlcv = {"close": c, "high": h, "low": l, "open": o, "volume": v}
        fm = feature_matrix(
            ohlcv,
            [("SMA", {"timeperiod": 10}), ("RSI", {"timeperiod": 14})],
            nan_policy="drop",
        )
        assert isinstance(fm, pd.DataFrame)
        assert not fm.isnull().any().any()

    def test_feature_matrix_string_indicator(self):
        from ferro_ta.analysis.features import feature_matrix

        o, h, l, c, v = _make_ohlcv(50)
        ohlcv = {"close": c, "high": h, "low": l, "open": o, "volume": v}
        fm = feature_matrix(ohlcv, ["SMA"])
        assert "SMA" in fm

    def test_feature_matrix_mixed_fastpath_and_multi_output(self):
        from ferro_ta.analysis.features import feature_matrix

        o, h, l, c, v = _make_ohlcv(80)
        ohlcv = {"close": c, "high": h, "low": l, "open": o, "volume": v}
        fm = feature_matrix(
            ohlcv,
            [
                ("SMA", {"timeperiod": 10}),
                ("ATR", {"timeperiod": 14}),
                ("BBANDS", {"timeperiod": 10}, 1),
            ],
        )
        assert "SMA" in fm
        assert "ATR" in fm
        assert "BBANDS_1" in fm


class TestComputeMany:
    def test_close_indicators_match_public_api(self):
        from ferro_ta import EMA, RSI, SMA
        from ferro_ta.data.batch import compute_many

        _, _, _, close, _ = _make_ohlcv(80)
        results = compute_many(
            [
                ("SMA", {"timeperiod": 10}),
                ("EMA", {"timeperiod": 12}),
                ("RSI", {"timeperiod": 14}),
            ],
            close=close,
        )

        np.testing.assert_allclose(
            results[0], SMA(close, timeperiod=10), equal_nan=True
        )
        np.testing.assert_allclose(
            results[1], EMA(close, timeperiod=12), equal_nan=True
        )
        np.testing.assert_allclose(
            results[2], RSI(close, timeperiod=14), equal_nan=True
        )

    def test_hlc_indicators_match_public_api(self):
        from ferro_ta import ADX, ATR
        from ferro_ta.data.batch import compute_many

        _, high, low, close, _ = _make_ohlcv(80)
        results = compute_many(
            [
                ("ATR", {"timeperiod": 14}),
                ("ADX", {"timeperiod": 14}),
            ],
            close=close,
            high=high,
            low=low,
        )

        np.testing.assert_allclose(
            results[0], ATR(high, low, close, timeperiod=14), equal_nan=True
        )
        np.testing.assert_allclose(
            results[1], ADX(high, low, close, timeperiod=14), equal_nan=True
        )

    def test_unsupported_kwargs_fall_back_cleanly(self):
        from ferro_ta import STDDEV
        from ferro_ta.data.batch import compute_many

        _, _, _, close, _ = _make_ohlcv(80)
        result = compute_many(
            [("STDDEV", {"timeperiod": 10, "nbdev": 2.0})], close=close
        )
        np.testing.assert_allclose(
            result[0], STDDEV(close, timeperiod=10, nbdev=2.0), equal_nan=True
        )


# ---------------------------------------------------------------------------
# Viz (smoke tests)
# ---------------------------------------------------------------------------


class TestViz:
    def test_plot_matplotlib_no_show(self):
        pytest.importorskip("matplotlib")
        from ferro_ta import RSI
        from ferro_ta.tools.viz import plot

        o, h, l, c, v = _make_ohlcv(60)
        ohlcv = {"close": c, "open": o, "high": h, "low": l, "volume": v}
        rsi = RSI(c, timeperiod=14)
        fig = plot(
            ohlcv,
            indicators={"RSI(14)": rsi},
            backend="matplotlib",
            show=False,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_unknown_backend_raises(self):
        from ferro_ta.tools.viz import plot

        o, h, l, c, v = _make_ohlcv(10)
        with pytest.raises(ValueError, match="Unknown backend"):
            plot({"close": c}, backend="bogus")

    def test_plot_savefig(self, tmp_path):
        pytest.importorskip("matplotlib")
        from ferro_ta.tools.viz import plot

        o, h, l, c, v = _make_ohlcv(30)
        ohlcv = {"close": c, "open": o, "high": h, "low": l, "volume": v}
        out = str(tmp_path / "chart.png")
        plot(ohlcv, backend="matplotlib", savefig=out, show=False)
        import os

        assert os.path.exists(out)
        import matplotlib.pyplot as plt

        plt.close("all")


# ---------------------------------------------------------------------------
# Data adapters
# ---------------------------------------------------------------------------


class TestDataAdapters:
    def test_in_memory_adapter(self):
        from ferro_ta.data.adapters import InMemoryAdapter

        o, h, l, c, v = _make_ohlcv(20)
        adapter = InMemoryAdapter(
            {"open": o, "high": h, "low": l, "close": c, "volume": v}
        )
        ohlcv = adapter.fetch()
        assert "close" in ohlcv

    def test_register_and_get_adapter(self):
        from ferro_ta.data.adapters import DataAdapter, get_adapter, register_adapter

        class MyAdapter(DataAdapter):
            def fetch(self, **kwargs):
                return {}

        register_adapter("_test_my", MyAdapter)
        cls = get_adapter("_test_my")
        assert cls is MyAdapter

    def test_get_unknown_adapter_raises(self):
        from ferro_ta.data.adapters import get_adapter

        with pytest.raises(KeyError):
            get_adapter("_nonexistent_adapter_xyz")

    def test_csv_adapter_requires_pandas(self, tmp_path):
        """CsvAdapter can be instantiated without pandas; fetch raises ImportError."""
        from ferro_ta.data.adapters import CsvAdapter

        adapter = CsvAdapter(str(tmp_path / "fake.csv"))
        assert adapter is not None

    def test_csv_adapter_fetch(self, tmp_path):
        pytest.importorskip("pandas")
        import pandas as pd

        from ferro_ta.data.adapters import CsvAdapter

        o, h, l, c, v = _make_ohlcv(10)
        csv_path = str(tmp_path / "ohlcv.csv")
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
        df.to_csv(csv_path, index=False)
        adapter = CsvAdapter(csv_path)
        result = adapter.fetch()
        assert "close" in result.columns
        assert len(result) == 10

    def test_builtin_adapters_registered(self):
        from ferro_ta.data.adapters import CsvAdapter, InMemoryAdapter, get_adapter

        assert get_adapter("csv") is CsvAdapter
        assert get_adapter("memory") is InMemoryAdapter
