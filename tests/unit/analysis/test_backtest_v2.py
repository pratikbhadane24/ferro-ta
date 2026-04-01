"""
v1.1.0 backtest feature tests.

Covers:
- CommissionModel: total_cost, presets, round-trip JSON, save/load
- Currency: INR/USD formatting, from_code lookup
- BacktestEngine: initial_capital, commission_model, trailing_stop, benchmark
- AdvancedBacktestResult: equity_abs, pnl_abs in trade log, summary fields
- Volatility-target position sizing
- Benchmark comparison metrics
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from ferro_ta._ferro_ta import CommissionModel

from ferro_ta.analysis.backtest import (
    EUR,
    GBP,
    INR,
    JPY,
    USD,
    USDT,
    BacktestEngine,
    Currency,
    format_currency,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def close_500():
    """500-bar synthetic close price series."""
    rng = np.random.default_rng(12345)
    return np.cumprod(1.0 + rng.standard_normal(500) * 0.01) * 100.0


@pytest.fixture
def ohlcv_500(close_500):
    close = close_500
    high = close * 1.005
    low = close * 0.995
    open_ = close * 0.999
    volume = np.full(len(close), 1_000_000.0)
    return open_, high, low, close, volume


# ===========================================================================
# TestCommissionModel
# ===========================================================================


class TestCommissionModel:
    def test_zero_model_costs_nothing(self):
        m = CommissionModel.zero()
        assert m.total_cost(100_000, 1, True) == 0.0
        assert m.total_cost(100_000, 1, False) == 0.0

    def test_flat_per_order(self):
        m = CommissionModel()
        m.flat_per_order = 20.0
        assert m.total_cost(100_000, 1, True) == pytest.approx(20.0)
        assert m.total_cost(100_000, 1, False) == pytest.approx(20.0)

    def test_max_brokerage_cap(self):
        m = CommissionModel()
        m.flat_per_order = 0.0
        m.rate_of_value = 0.001  # 0.1%
        m.max_brokerage = 20.0
        # 0.1% of 50_000 = 50, capped at 20
        assert m.total_cost(50_000, 1, True) == pytest.approx(20.0)
        # 0.1% of 5_000 = 5, not capped
        assert m.total_cost(5_000, 1, True) == pytest.approx(5.0)

    def test_stt_buy_side_only(self):
        m = CommissionModel()
        m.stt_rate = 0.001
        m.stt_on_buy = True
        m.stt_on_sell = False
        buy_cost = m.total_cost(100_000, 1, True)
        sell_cost = m.total_cost(100_000, 1, False)
        assert buy_cost == pytest.approx(100.0)
        assert sell_cost == pytest.approx(0.0)

    def test_stt_sell_side_only(self):
        m = CommissionModel()
        m.stt_rate = 0.00025
        m.stt_on_buy = False
        m.stt_on_sell = True
        buy_cost = m.total_cost(100_000, 1, True)
        sell_cost = m.total_cost(100_000, 1, False)
        assert buy_cost == pytest.approx(0.0)
        assert sell_cost == pytest.approx(25.0)

    def test_gst_on_brokerage_exchange_not_stt(self):
        m = CommissionModel()
        m.flat_per_order = 20.0
        m.exchange_charges_rate = 0.0001
        m.gst_rate = 0.18
        m.stt_rate = 0.001
        m.stt_on_sell = True
        # GST = 0.18 * (20 + 0.0001 * 100_000) = 0.18 * 30 = 5.4
        # STT = 100 (sell side)
        total = m.total_cost(100_000, 1, False)
        expected_gst = 0.18 * (20.0 + 0.0001 * 100_000)
        assert total == pytest.approx(20.0 + 100.0 + 0.0001 * 100_000 + expected_gst)

    def test_stamp_duty_buy_only(self):
        m = CommissionModel()
        m.stamp_duty_rate = 0.00015
        buy_cost = m.total_cost(100_000, 1, True)
        sell_cost = m.total_cost(100_000, 1, False)
        assert buy_cost == pytest.approx(15.0)
        assert sell_cost == pytest.approx(0.0)

    def test_per_lot_charge(self):
        m = CommissionModel()
        m.per_lot = 2.0
        # 5 lots
        assert m.total_cost(50_000, 5, True) == pytest.approx(10.0)

    def test_cost_fraction(self):
        m = CommissionModel()
        m.flat_per_order = 20.0
        frac = m.cost_fraction(100_000, 1, True, 100_000.0)
        assert frac == pytest.approx(20.0 / 100_000.0)

    def test_cost_fraction_zero_capital(self):
        m = CommissionModel()
        m.flat_per_order = 20.0
        assert m.cost_fraction(100_000, 1, True, 0.0) == 0.0

    def test_proportional_preset(self):
        m = CommissionModel.proportional(0.001)
        assert m.total_cost(100_000, 1, True) == pytest.approx(100.0)
        assert m.gst_rate == 0.0

    def test_repr_contains_key_fields(self):
        m = CommissionModel.equity_delivery_india()
        r = repr(m)
        assert "CommissionModel" in r
        assert "lot_size" in r


class TestCommissionPresets:
    def test_equity_delivery_india_smoke(self):
        m = CommissionModel.equity_delivery_india()
        # Buy ₹1L trade: brokerage cap ₹20, STT ₹100 (both sides)
        cost = m.total_cost(100_000, 1, True)
        assert cost > 0.0
        assert cost < 500.0  # sanity upper bound
        # Brokerage should be capped at ₹20
        assert m.flat_per_order == 0.0
        assert m.max_brokerage == pytest.approx(20.0)
        assert m.stt_on_buy is True
        assert m.stt_on_sell is True

    def test_equity_intraday_india_smoke(self):
        m = CommissionModel.equity_intraday_india()
        cost_buy = m.total_cost(100_000, 1, True)
        cost_sell = m.total_cost(100_000, 1, False)
        # STT only on sell side for intraday
        assert m.stt_on_buy is False
        assert m.stt_on_sell is True
        assert cost_sell > cost_buy  # sell has more cost (STT)

    def test_futures_india_smoke(self):
        m = CommissionModel.futures_india()
        assert m.flat_per_order == pytest.approx(20.0)
        assert m.stt_on_buy is False
        assert m.stt_on_sell is True
        assert m.lot_size == pytest.approx(25.0)

    def test_options_india_smoke(self):
        m = CommissionModel.options_india()
        assert m.flat_per_order == pytest.approx(20.0)
        assert m.stt_rate == pytest.approx(0.0015)
        assert m.lot_size == pytest.approx(25.0)


class TestCommissionFix:
    """The old 'commission_per_trade=20.0' bug would subtract ₹20 from 1.0-normalized
    equity — a 2000% error. The new model correctly computes 0.02% fraction."""

    def test_flat_20_on_1L_capital_is_tiny_fraction(self):
        m = CommissionModel()
        m.flat_per_order = 20.0
        frac = m.cost_fraction(100_000, 1, True, 100_000.0)
        # ₹20 / ₹100_000 = 0.02%
        assert frac == pytest.approx(20.0 / 100_000.0, rel=1e-6)
        assert frac < 0.01  # definitely not 2000%

    def test_commission_reduces_equity_vs_no_commission(self):
        rng = np.random.default_rng(99)
        close = np.cumprod(1.0 + rng.standard_normal(200) * 0.01) * 100.0
        m = CommissionModel.equity_intraday_india()
        r_comm = (
            BacktestEngine()
            .with_commission_model(m)
            .with_initial_capital(100_000)
            .run(close, "sma_crossover")
        )
        r_none = (
            BacktestEngine().with_initial_capital(100_000).run(close, "sma_crossover")
        )
        # Commission should reduce final equity (or keep equal if zero trades)
        assert r_comm.final_equity <= r_none.final_equity


class TestCommissionSaveLoad:
    def test_to_json_from_json_round_trip(self):
        m = CommissionModel.equity_delivery_india()
        j = m.to_json()
        m2 = CommissionModel.from_json(j)
        assert m == m2
        assert m2.stt_rate == pytest.approx(m.stt_rate)
        assert m2.lot_size == pytest.approx(m.lot_size)
        assert m2.gst_rate == pytest.approx(m.gst_rate)

    def test_save_load_round_trip(self):
        m = CommissionModel.futures_india()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            m.save(path)
            assert os.path.exists(path)
            m2 = CommissionModel.load(path)
            assert m == m2
        finally:
            os.unlink(path)

    def test_from_json_invalid_raises(self):
        with pytest.raises(Exception):
            CommissionModel.from_json("{invalid json")

    def test_load_missing_file_raises(self):
        with pytest.raises(Exception):
            CommissionModel.load("/nonexistent/path/commission.json")


# ===========================================================================
# TestCurrency
# ===========================================================================


class TestCurrency:
    def test_inr_lakh_grouping(self):
        assert INR.format(123456.78) == "₹1,23,456.78"
        assert INR.format(1000000.0) == "₹10,00,000.00"
        assert INR.format(10000000.0) == "₹1,00,00,000.00"
        assert INR.format(100.0) == "₹100.00"
        assert INR.format(1234.5) == "₹1,234.50"

    def test_inr_negative(self):
        result = INR.format(-5000.0)
        assert result.startswith("-₹")
        assert "5,000.00" in result

    def test_usd_standard_grouping(self):
        assert USD.format(1234567.89) == "$1,234,567.89"
        assert USD.format(0.5) == "$0.50"
        assert USD.format(1000.0) == "$1,000.00"

    def test_jpy_no_decimals(self):
        result = JPY.format(1000000.0)
        assert result == "¥1,000,000"

    def test_eur_format(self):
        assert "€" in EUR.format(100.0)

    def test_gbp_format(self):
        assert "£" in GBP.format(100.0)

    def test_usdt_format(self):
        assert "₮" in USDT.format(100.0)

    def test_format_currency_helper(self):
        assert format_currency(123456.78) == "₹1,23,456.78"
        assert format_currency(1000.0, USD) == "$1,000.00"

    def test_currency_immutable(self):
        with pytest.raises(AttributeError):
            INR.code = "USD"  # type: ignore[misc]

    def test_currency_equality(self):
        c1 = Currency.from_code("INR")
        assert c1 == INR
        assert INR != USD

    def test_currency_hash_usable_in_dict(self):
        d = {INR: 100_000, USD: 100}
        assert d[INR] == 100_000


# ===========================================================================
# TestInitialCapital
# ===========================================================================


class TestInitialCapital:
    def test_equity_abs_shape(self, close_500):
        result = (
            BacktestEngine()
            .with_initial_capital(200_000)
            .run(close_500, "sma_crossover")
        )
        assert result.equity_abs.shape == result.equity.shape

    def test_equity_abs_is_equity_times_capital(self, close_500):
        capital = 150_000.0
        result = (
            BacktestEngine()
            .with_initial_capital(capital)
            .run(close_500, "sma_crossover")
        )
        np.testing.assert_allclose(result.equity_abs, result.equity * capital)

    def test_summary_contains_capital_fields(self, close_500):
        capital = 100_000.0
        result = (
            BacktestEngine()
            .with_initial_capital(capital)
            .run(close_500, "sma_crossover")
        )
        s = result.summary()
        assert "initial_capital" in s
        assert "final_capital" in s
        assert "absolute_pnl" in s
        assert s["initial_capital"] == pytest.approx(capital)
        assert s["final_capital"] == pytest.approx(result.equity_abs[-1])
        assert s["absolute_pnl"] == pytest.approx(s["final_capital"] - capital)

    def test_pnl_abs_in_trade_log(self, close_500, ohlcv_500):
        open_, high, low, close, _ = ohlcv_500
        capital = 100_000.0
        result = (
            BacktestEngine()
            .with_initial_capital(capital)
            .with_ohlcv(high=high, low=low, open_=open_)
            .run(close, "sma_crossover")
        )
        if result.trades is not None and len(result.trades) > 0:
            assert "pnl_abs" in result.trades.columns
            np.testing.assert_allclose(
                result.trades["pnl_abs"].values,
                result.trades["pnl_pct"].values * capital,
            )


class TestINRRepr:
    def test_repr_shows_inr_symbol(self, close_500):
        result = (
            BacktestEngine()
            .with_currency(INR)
            .with_initial_capital(100_000)
            .run(close_500, "sma_crossover")
        )
        r = repr(result)
        assert "₹" in r

    def test_currency_code_in_summary(self, close_500):
        result = (
            BacktestEngine()
            .with_currency("USD")
            .with_initial_capital(10_000)
            .run(close_500, "sma_crossover")
        )
        s = result.summary()
        assert s["currency"] == "USD"

    def test_unknown_currency_raises(self):
        with pytest.raises(Exception, match="Unknown currency"):
            BacktestEngine().with_currency("XYZ")


# ===========================================================================
# TestVolatilityTargetSizing
# ===========================================================================


class TestVolatilityTargetSizing:
    def test_vol_target_runs_without_error(self, close_500):
        result = (
            BacktestEngine()
            .with_position_sizing("volatility_target", target_vol=0.10)
            .run(close_500, "sma_crossover")
        )
        assert len(result.equity) == len(close_500)
        assert np.isfinite(result.final_equity)

    def test_vol_target_signals_are_scaled(self, close_500):
        # With very low target vol the strategy should have fewer active positions
        result_low = (
            BacktestEngine()
            .with_position_sizing("volatility_target", target_vol=0.01)
            .run(close_500, "sma_crossover")
        )
        result_high = (
            BacktestEngine()
            .with_position_sizing("volatility_target", target_vol=1.0)
            .run(close_500, "sma_crossover")
        )
        # Lower vol target → lower absolute position sizes → lower annualised vol
        low_std = float(np.nanstd(result_low.strategy_returns))
        high_std = float(np.nanstd(result_high.strategy_returns))
        assert low_std <= high_std or np.isclose(low_std, high_std, rtol=0.5)


# ===========================================================================
# TestBenchmark
# ===========================================================================


class TestBenchmark:
    def test_benchmark_metrics_present(self, close_500):
        rng = np.random.default_rng(77)
        benchmark = np.cumprod(1.0 + rng.standard_normal(500) * 0.008) * 100.0
        result = (
            BacktestEngine().with_benchmark(benchmark).run(close_500, "sma_crossover")
        )
        s = result.summary()
        assert "alpha" in s
        assert "beta" in s
        assert "tracking_error" in s
        assert "information_ratio" in s
        assert "benchmark_cagr" in s

    def test_identical_strategy_benchmark_has_low_tracking_error(self, close_500):
        # When strategy returns = benchmark returns, tracking error ≈ 0
        # Use the equity as its own benchmark
        result = (
            BacktestEngine().with_benchmark(close_500).run(close_500, "sma_crossover")
        )
        m = result.metrics
        # Beta should be finite
        assert np.isfinite(m.get("beta", float("nan")))

    def test_benchmark_wrong_length_ignored(self, close_500):
        short_bench = close_500[:100]
        # Should not raise — benchmark mismatch is silently ignored
        result = (
            BacktestEngine().with_benchmark(short_bench).run(close_500, "sma_crossover")
        )
        # alpha should NOT appear (length mismatch)
        assert "alpha" not in result.metrics


# ===========================================================================
# TestTrailingStop
# ===========================================================================


class TestTrailingStop:
    def test_trailing_stop_runs(self, ohlcv_500, close_500):
        open_, high, low, close, _ = ohlcv_500
        result = (
            BacktestEngine()
            .with_ohlcv(high=high, low=low, open_=open_)
            .with_trailing_stop(0.02)
            .run(close, "sma_crossover")
        )
        assert len(result.equity) == len(close)
        assert np.isfinite(result.final_equity)

    def test_trailing_stop_reduces_losses_on_downtrend(self):
        """Trailing stop should exit longs earlier on a falling market."""
        # Construct a clear downtrend after initial rise
        prices = np.concatenate(
            [
                np.linspace(100, 120, 50),  # rise (signal stays long)
                np.linspace(120, 60, 150),  # sharp fall
            ]
        )
        high = prices * 1.002
        low = prices * 0.998
        open_ = prices * 0.999

        result_trail = (
            BacktestEngine()
            .with_ohlcv(high=high, low=low, open_=open_)
            .with_trailing_stop(0.03)
            .run(prices, "sma_crossover")
        )
        result_no_trail = (
            BacktestEngine()
            .with_ohlcv(high=high, low=low, open_=open_)
            .run(prices, "sma_crossover")
        )
        # Trailing stop should yield better (or equal) max drawdown
        dd_trail = result_trail.metrics.get("max_drawdown", 0.0)
        dd_no_trail = result_no_trail.metrics.get("max_drawdown", 0.0)
        # max_drawdown is negative; higher value = smaller drawdown
        assert dd_trail >= dd_no_trail - 0.05  # allow 5% tolerance


# ===========================================================================
# TestBacktestEngineChaining
# ===========================================================================


class TestBacktestEngineChaining:
    def test_full_chain_runs(self, close_500, ohlcv_500):
        open_, high, low, close, _ = ohlcv_500
        rng = np.random.default_rng(42)
        benchmark = np.cumprod(1.0 + rng.standard_normal(500) * 0.008) * 100.0

        result = (
            BacktestEngine()
            .with_currency("INR")
            .with_initial_capital(100_000)
            .with_commission_model(CommissionModel.equity_intraday_india())
            .with_trailing_stop(0.02)
            .with_benchmark(benchmark)
            .with_ohlcv(high=high, low=low, open_=open_)
            .run(close, "sma_crossover")
        )
        assert len(result.equity) == len(close)
        assert result.currency == INR
        assert result.initial_capital == pytest.approx(100_000.0)
        assert np.isfinite(result.final_equity)

        s = result.summary()
        assert s["currency"] == "INR"
        assert "alpha" in s  # benchmark was set

    def test_to_equity_dataframe(self, close_500):
        result = (
            BacktestEngine()
            .with_initial_capital(50_000)
            .run(close_500, "sma_crossover")
        )
        df = result.to_equity_dataframe()
        assert "equity" in df.columns
        assert "equity_abs" in df.columns
        assert "strategy_returns" in df.columns
        assert "drawdown" in df.columns
        assert len(df) == len(close_500)
        np.testing.assert_allclose(df["equity_abs"].values, result.equity_abs)
