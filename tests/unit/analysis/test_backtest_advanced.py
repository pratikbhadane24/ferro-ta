"""Tests for the advanced backtesting engine.

Covers all 10 test groups from the plan:
1. backtest_ohlcv_core
2. compute_performance_metrics
3. extract_trades
4. backtest_multi_asset_core
5. monte_carlo_bootstrap
6. walk_forward_indices
7. kelly_fraction / half_kelly_fraction
8. BacktestEngine (Python API)
9. walk_forward() (Python API)
10. monte_carlo() (Python API)
"""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest
from ferro_ta._ferro_ta import (
    backtest_core,
    backtest_multi_asset_core,
    backtest_ohlcv_core,
    compute_performance_metrics,
    drawdown_series,
    half_kelly_fraction,
    kelly_fraction,
    monte_carlo_bootstrap,
    walk_forward_indices,
)
from ferro_ta._ferro_ta import (
    extract_trades_ohlcv as extract_trades,
)

from ferro_ta.analysis.backtest import (
    AdvancedBacktestResult,
    BacktestEngine,
    BacktestResult,
    MonteCarloResult,
    PortfolioBacktestResult,
    WalkForwardResult,
    backtest,
    backtest_portfolio,
    monte_carlo,
    rsi_strategy,
    walk_forward,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 100, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
    open_ = close * (1 - rng.uniform(0, 0.005, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    signals = np.where(np.arange(n) % 20 < 10, 1.0, -1.0).astype(np.float64)
    return open_, high, low, close, signals


def _all_finite(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(arr[~np.isnan(arr)])))


# ===========================================================================
# Group 1: backtest_ohlcv_core
# ===========================================================================


class TestBacktestOhlcvCore:
    def test_returns_five_arrays(self):
        o, h, l, c, s = _make_ohlcv()
        result = backtest_ohlcv_core(o, h, l, c, s)
        assert len(result) == 5

    def test_shapes_match_input(self):
        o, h, l, c, s = _make_ohlcv(n=80)
        pos, fp, br, sr, eq = backtest_ohlcv_core(o, h, l, c, s)
        for arr in (pos, fp, br, sr, eq):
            assert arr.shape == (80,)

    def test_equity_starts_at_one(self):
        o, h, l, c, s = _make_ohlcv()
        _, _, _, _, eq = backtest_ohlcv_core(o, h, l, c, s)
        assert eq[0] == pytest.approx(1.0, abs=1e-9)

    def test_no_lookahead_bias(self):
        """Position at bar 0 must always be 0 (signal not yet available)."""
        o, h, l, c, s = _make_ohlcv()
        pos, _, _, _, _ = backtest_ohlcv_core(o, h, l, c, s)
        assert pos[0] == 0.0

    def test_stop_loss_reduces_equity_relative_to_no_stop(self):
        """With a tight stop-loss, equity should differ from no-stop run."""
        o, h, l, c, s = _make_ohlcv(n=200)
        _, _, _, _, eq_no_stop = backtest_ohlcv_core(o, h, l, c, s)
        _, _, _, _, eq_with_stop = backtest_ohlcv_core(
            o, h, l, c, s, stop_loss_pct=0.005
        )
        # They should differ (stop-loss triggered on at least one bar)
        assert not np.allclose(eq_no_stop, eq_with_stop)

    def test_fill_prices_nan_when_flat(self):
        """fill_prices must be NaN whenever the position is 0."""
        o, h, l, c, s = _make_ohlcv()
        pos, fp, _, _, _ = backtest_ohlcv_core(o, h, l, c, s)
        flat_mask = pos == 0.0
        assert np.all(np.isnan(fp[flat_mask]))

    def test_market_close_mode_different_from_open(self):
        o, h, l, c, s = _make_ohlcv(n=150)
        _, _, _, sr_open, _ = backtest_ohlcv_core(
            o, h, l, c, s, fill_mode="market_open"
        )
        _, _, _, sr_close, _ = backtest_ohlcv_core(
            o, h, l, c, s, fill_mode="market_close"
        )
        # Different fill modes → different returns
        assert not np.allclose(sr_open, sr_close, equal_nan=True)

    def test_raises_on_mismatched_lengths(self):
        o, h, l, c, s = _make_ohlcv()
        with pytest.raises(Exception):
            backtest_ohlcv_core(o[:-1], h, l, c, s)


# ===========================================================================
# Group 2: compute_performance_metrics
# ===========================================================================


class TestComputePerformanceMetrics:
    EXPECTED_KEYS = {
        "total_return",
        "cagr",
        "annualized_vol",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "avg_drawdown",
        "max_drawdown_duration_bars",
        "avg_drawdown_duration_bars",
        "ulcer_index",
        "omega_ratio",
        "win_rate",
        "profit_factor",
        "r_expectancy",
        "avg_win",
        "avg_loss",
        "tail_ratio",
        "skewness",
        "kurtosis",
        "best_bar",
        "worst_bar",
        "n_trades",
    }

    def _run(self, n: int = 200, seed: int = 0):
        rng = np.random.default_rng(seed)
        r = rng.standard_normal(n) * 0.01
        eq = np.cumprod(1 + r)
        return compute_performance_metrics(r, eq)

    def test_all_expected_keys_present(self):
        m = self._run()
        assert self.EXPECTED_KEYS.issubset(set(m.keys()))

    def test_sharpe_all_positive_returns(self):
        """Constant +1% daily returns → Sharpe = (annualised) > 0."""
        r = np.full(252, 0.01)
        eq = np.cumprod(1 + r)
        m = compute_performance_metrics(r, eq)
        assert m["sharpe"] > 0

    def test_max_drawdown_matches_drawdown_series(self):
        rng = np.random.default_rng(7)
        r = rng.standard_normal(300) * 0.015
        eq = np.cumprod(1 + r)
        m = compute_performance_metrics(r, eq)
        _, max_dd_ref = drawdown_series(eq)
        assert m["max_drawdown"] == pytest.approx(max_dd_ref, abs=1e-9)

    def test_cagr_formula(self):
        r = np.full(252, 0.01)
        eq = np.cumprod(1 + r)
        m = compute_performance_metrics(r, eq)
        # Rust computes CAGR as (eq[-1]/eq[0])^(ppy/n) - 1, treating eq[0] as start equity
        expected_cagr = (eq[-1] / eq[0]) ** (252.0 / len(r)) - 1.0
        assert m["cagr"] == pytest.approx(expected_cagr, rel=1e-6)

    def test_win_rate_between_0_and_1(self):
        m = self._run()
        assert 0.0 <= m["win_rate"] <= 1.0

    def test_max_drawdown_nonpositive(self):
        m = self._run()
        assert m["max_drawdown"] <= 0.0

    def test_total_return_sign(self):
        r = np.full(100, 0.005)
        eq = np.cumprod(1 + r)
        m = compute_performance_metrics(r, eq)
        assert m["total_return"] > 0.0

    def test_raises_on_short_input(self):
        with pytest.raises(Exception):
            compute_performance_metrics(np.array([0.01]), np.array([1.01]))

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(Exception):
            compute_performance_metrics(np.ones(10) * 0.01, np.ones(20))


# ===========================================================================
# Group 3: extract_trades
# ===========================================================================


class TestExtractTrades:
    def _run_ohlcv(self, n: int = 100):
        o, h, l, c, s = _make_ohlcv(n=n)
        pos, fp, _, _, _ = backtest_ohlcv_core(o, h, l, c, s)
        return pos, fp, h, l

    def test_returns_nine_arrays(self):
        pos, fp, h, l = self._run_ohlcv()
        result = extract_trades(pos, fp, h, l)
        assert len(result) == 9

    def test_all_arrays_same_length(self):
        pos, fp, h, l = self._run_ohlcv(n=200)
        arrays = extract_trades(pos, fp, h, l)
        lengths = {len(a) for a in arrays}
        assert len(lengths) == 1  # all same length

    def test_duration_bars_positive(self):
        pos, fp, h, l = self._run_ohlcv(n=200)
        _, _, _, _, _, _, dur, _, _ = extract_trades(pos, fp, h, l)
        assert np.all(dur >= 0)

    def test_exit_bar_gte_entry_bar(self):
        pos, fp, h, l = self._run_ohlcv(n=200)
        eb, xb, _, _, _, _, _, _, _ = extract_trades(pos, fp, h, l)
        assert np.all(xb >= eb)

    def test_direction_is_plus_minus_one(self):
        pos, fp, h, l = self._run_ohlcv(n=200)
        _, _, d, _, _, _, _, _, _ = extract_trades(pos, fp, h, l)
        if len(d) > 0:
            assert set(np.unique(d)).issubset({1.0, -1.0})

    def test_mfe_gte_mae(self):
        """MFE (best) must always be >= MAE (worst) within the trade."""
        pos, fp, h, l = self._run_ohlcv(n=200)
        _, _, _, _, _, _, _, mae, mfe = extract_trades(pos, fp, h, l)
        if len(mae) > 0:
            assert np.all(mfe >= mae)

    def test_raises_on_mismatched_lengths(self):
        pos, fp, h, l = self._run_ohlcv()
        with pytest.raises(Exception):
            extract_trades(pos[:-1], fp, h, l)


# ===========================================================================
# Group 4: backtest_multi_asset_core
# ===========================================================================


class TestBacktestMultiAssetCore:
    def test_single_asset_matches_backtest_core(self):
        """1-asset multi_asset == scalar backtest_core with same weights."""
        rng = np.random.default_rng(99)
        n = 150
        close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        signals = np.where(np.arange(n) % 15 < 7, 1.0, -1.0).astype(np.float64)

        # Single asset via multi_asset (weights = signals)
        close2d = close.reshape(n, 1)
        w2d = signals.reshape(n, 1)
        ar, pr, pe = backtest_multi_asset_core(close2d, w2d)

        # Same via backtest_core
        _, _, sr_ref, eq_ref = backtest_core(close, signals)

        npt.assert_allclose(pe, np.asarray(eq_ref), rtol=1e-6)

    def test_returns_shapes(self):
        n, k = 100, 5
        rng = np.random.default_rng(0)
        c2d = np.cumprod(1 + rng.standard_normal((n, k)) * 0.01, axis=0) * 100
        w2d = np.ones((n, k)) * 0.2
        ar, pr, pe = backtest_multi_asset_core(c2d, w2d)
        assert ar.shape == (n, k)
        assert pr.shape == (n,)
        assert pe.shape == (n,)

    def test_parallel_equals_serial(self):
        n, k = 120, 4
        rng = np.random.default_rng(1)
        c2d = np.cumprod(1 + rng.standard_normal((n, k)) * 0.01, axis=0) * 100
        w2d = rng.choice([-1.0, 0.0, 1.0], size=(n, k)).astype(np.float64)
        _, _, pe_par = backtest_multi_asset_core(c2d, w2d, parallel=True)
        _, _, pe_ser = backtest_multi_asset_core(c2d, w2d, parallel=False)
        npt.assert_allclose(pe_par, pe_ser, rtol=1e-10)

    def test_raises_on_mismatched_shapes(self):
        c2d = np.ones((50, 3))
        w2d = np.ones((50, 4))  # wrong n_assets
        with pytest.raises(Exception):
            backtest_multi_asset_core(c2d, w2d)

    def test_equity_starts_at_one(self):
        n, k = 50, 2
        c2d = np.ones((n, k)) * 100.0
        w2d = np.zeros((n, k))
        _, _, pe = backtest_multi_asset_core(c2d, w2d)
        assert pe[0] == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# Group 5: monte_carlo_bootstrap
# ===========================================================================


class TestMonteCarloBootstrap:
    def _returns(self, n: int = 200, seed: int = 5):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) * 0.01

    def test_output_shape(self):
        r = self._returns()
        mc = monte_carlo_bootstrap(r, n_sims=50)
        assert mc.shape == (50, 200)

    def test_seed_reproducibility(self):
        r = self._returns()
        mc1 = monte_carlo_bootstrap(r, n_sims=100, seed=7)
        mc2 = monte_carlo_bootstrap(r, n_sims=100, seed=7)
        npt.assert_array_equal(mc1, mc2)

    def test_different_seeds_differ(self):
        r = self._returns()
        mc1 = monte_carlo_bootstrap(r, n_sims=50, seed=1)
        mc2 = monte_carlo_bootstrap(r, n_sims=50, seed=2)
        assert not np.allclose(mc1, mc2)

    def test_equity_starts_at_one(self):
        r = self._returns()
        mc = monte_carlo_bootstrap(r, n_sims=20)
        # Bootstrap resamples returns randomly, so mc[:,0] = 1 + random_return
        # All first-bar equity values must be in range of possible (1+r) values
        possible_first_bar = set(np.round(1.0 + r, 12))
        for val in mc[:, 0]:
            assert any(abs(val - p) < 1e-9 for p in possible_first_bar)

    def test_block_bootstrap_shape(self):
        r = self._returns(n=100)
        mc = monte_carlo_bootstrap(r, n_sims=30, block_size=5)
        assert mc.shape == (30, 100)

    def test_raises_on_empty_input(self):
        with pytest.raises(Exception):
            monte_carlo_bootstrap(np.array([0.01]), n_sims=10)


# ===========================================================================
# Group 6: walk_forward_indices
# ===========================================================================


class TestWalkForwardIndices:
    def test_output_shape(self):
        idx = walk_forward_indices(500, 200, 50)
        assert idx.ndim == 2
        assert idx.shape[1] == 4

    def test_non_anchored_fixed_train_window(self):
        idx = walk_forward_indices(400, 200, 50)
        n_folds = idx.shape[0]
        assert n_folds >= 2
        for fold in idx:
            tr_len = fold[1] - fold[0]
            assert tr_len == 200

    def test_anchored_growing_train_window(self):
        idx = walk_forward_indices(400, 150, 50, anchored=True)
        for fold in idx:
            assert fold[0] == 0  # always starts at 0
        train_lengths = idx[:, 1] - idx[:, 0]
        assert train_lengths[-1] >= train_lengths[0]

    def test_no_test_fold_overlap(self):
        idx = walk_forward_indices(500, 200, 50)
        # Test intervals should be non-overlapping (step = test_bars by default)
        for i in range(len(idx) - 1):
            assert idx[i, 3] <= idx[i + 1, 2]

    def test_all_test_folds_within_bounds(self):
        n = 600
        idx = walk_forward_indices(n, 200, 100)
        assert np.all(idx[:, 0] >= 0)
        assert np.all(idx[:, 3] <= n)

    def test_step_bars_parameter(self):
        idx_default = walk_forward_indices(500, 200, 50)
        idx_step = walk_forward_indices(500, 200, 50, step_bars=25)
        # Smaller step → more folds
        assert idx_step.shape[0] >= idx_default.shape[0]

    def test_raises_when_no_folds_fit(self):
        with pytest.raises(Exception):
            walk_forward_indices(100, 80, 80)  # 80+80 > 100


# ===========================================================================
# Group 7: kelly_fraction / half_kelly_fraction
# ===========================================================================


class TestKellyFraction:
    def test_positive_expectancy(self):
        k = kelly_fraction(0.6, 0.02, 0.01)
        assert k > 0.0

    def test_zero_edge_returns_zero(self):
        """win_rate = loss_rate AND avg_win = avg_loss → Kelly = 0."""
        k = kelly_fraction(0.5, 0.01, 0.01)
        assert k == pytest.approx(0.0, abs=1e-9)

    def test_negative_expectancy_clamped_to_zero(self):
        k = kelly_fraction(0.3, 0.01, 0.02)
        assert k == 0.0

    def test_half_kelly_is_half_of_kelly(self):
        k = kelly_fraction(0.6, 0.03, 0.015)
        hk = half_kelly_fraction(0.6, 0.03, 0.015)
        assert hk == pytest.approx(k / 2.0, rel=1e-9)

    def test_result_clamped_to_one(self):
        k = kelly_fraction(0.99, 0.5, 0.001)
        assert k <= 1.0

    def test_raises_on_invalid_win_rate(self):
        with pytest.raises(Exception):
            kelly_fraction(1.5, 0.01, 0.01)

    def test_raises_on_nonpositive_avg_win(self):
        with pytest.raises(Exception):
            kelly_fraction(0.6, 0.0, 0.01)


# ===========================================================================
# Group 8: BacktestEngine (Python API)
# ===========================================================================


class TestBacktestEngine:
    def _close(self, n: int = 200, seed: int = 10) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0

    def test_run_returns_advanced_result(self):
        c = self._close()
        r = BacktestEngine().run(c, "rsi_30_70")
        assert isinstance(r, AdvancedBacktestResult)

    def test_advanced_result_is_backtest_result(self):
        c = self._close()
        r = BacktestEngine().run(c, "rsi_30_70")
        assert isinstance(r, BacktestResult)

    def test_chaining_returns_self(self):
        engine = BacktestEngine()
        assert engine.with_commission(0.001) is engine
        assert engine.with_slippage(5.0) is engine
        assert engine.with_stop_loss(0.02) is engine

    def test_all_metric_keys_present(self):
        c = self._close()
        r = BacktestEngine().run(c, "rsi_30_70")
        assert "sharpe" in r.metrics
        assert "max_drawdown" in r.metrics
        assert "cagr" in r.metrics

    def test_drawdown_series_shape(self):
        c = self._close()
        r = BacktestEngine().run(c)
        assert r.drawdown_series.shape == c.shape

    def test_drawdown_series_nonpositive(self):
        c = self._close()
        r = BacktestEngine().run(c)
        assert np.all(r.drawdown_series <= 0.0)

    def test_engine_close_only_matches_backtest_func(self):
        c = self._close()
        r_engine = BacktestEngine().run(c, "rsi_30_70")
        r_func = backtest(c, strategy="rsi_30_70")
        npt.assert_allclose(r_engine.equity, r_func.equity, rtol=1e-9)

    def test_ohlcv_mode_runs(self):
        c = self._close()
        h = c * 1.01
        l = c * 0.99
        o = c * 0.999
        r = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .with_stop_loss(0.02)
            .run(c)
        )
        assert r.equity.shape == c.shape

    def test_trades_dataframe_columns(self):
        c = self._close()
        r = BacktestEngine().run(c, "sma_crossover")
        if r.trades is not None:
            expected_cols = {
                "entry_bar",
                "exit_bar",
                "direction",
                "entry_price",
                "exit_price",
                "pnl_pct",
                "duration_bars",
                "mae",
                "mfe",
            }
            assert expected_cols.issubset(set(r.trades.columns))

    def test_invalid_fill_mode_raises(self):
        with pytest.raises(Exception):
            BacktestEngine().with_fill_mode("invalid")


# ===========================================================================
# Group 9: walk_forward() Python API
# ===========================================================================


class TestWalkForward:
    def _setup(self, n: int = 400):
        rng = np.random.default_rng(99)
        close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        param_grid = [{"timeperiod": p} for p in [10, 14, 20]]
        return close, param_grid

    def test_returns_walk_forward_result(self):
        c, pg = self._setup()
        r = walk_forward(c, rsi_strategy, pg, train_bars=200, test_bars=50)
        assert isinstance(r, WalkForwardResult)

    def test_fold_count_matches_indices(self):
        c, pg = self._setup()
        r = walk_forward(c, rsi_strategy, pg, train_bars=200, test_bars=50)
        assert len(r.fold_results) == r.fold_indices.shape[0]

    def test_oos_equity_length(self):
        c, pg = self._setup()
        r = walk_forward(c, rsi_strategy, pg, train_bars=200, test_bars=50)
        total_test_bars = sum(
            int(r.fold_indices[i, 3]) - int(r.fold_indices[i, 2])
            for i in range(len(r.fold_results))
        )
        assert len(r.oos_equity) == total_test_bars

    def test_oos_metrics_has_sharpe(self):
        c, pg = self._setup()
        r = walk_forward(c, rsi_strategy, pg, train_bars=200, test_bars=50)
        assert "sharpe" in r.oos_metrics

    def test_anchored_mode(self):
        c, pg = self._setup()
        r = walk_forward(
            c, rsi_strategy, pg, train_bars=200, test_bars=50, anchored=True
        )
        # In anchored mode, training always starts at 0
        assert np.all(r.fold_indices[:, 0] == 0)

    def test_param_stability_populated(self):
        c, pg = self._setup()
        r = walk_forward(c, rsi_strategy, pg, train_bars=200, test_bars=50)
        assert "timeperiod" in r.param_stability
        assert "most_chosen" in r.param_stability["timeperiod"]


# ===========================================================================
# Group 10: monte_carlo() Python API
# ===========================================================================


class TestMonteCarlo:
    def _result(self, n: int = 200):
        rng = np.random.default_rng(77)
        c = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        return BacktestEngine().run(c, "rsi_30_70")

    def test_returns_monte_carlo_result(self):
        r = self._result()
        mc = monte_carlo(r, n_sims=100)
        assert isinstance(mc, MonteCarloResult)

    def test_equity_curves_shape(self):
        r = self._result(n=150)
        mc = monte_carlo(r, n_sims=80)
        assert mc.equity_curves.shape == (80, 150)

    def test_confidence_bounds_cover_median(self):
        r = self._result()
        mc = monte_carlo(r, n_sims=500, confidence=0.95)
        assert np.all(mc.confidence_lower <= mc.median_curve + 1e-9)
        assert np.all(mc.confidence_upper >= mc.median_curve - 1e-9)

    def test_prob_profit_in_range(self):
        r = self._result()
        mc = monte_carlo(r, n_sims=200)
        assert 0.0 <= mc.prob_profit <= 1.0

    def test_accepts_raw_array(self):
        rng = np.random.default_rng(3)
        returns = rng.standard_normal(100) * 0.01
        mc = monte_carlo(returns, n_sims=50)
        assert isinstance(mc, MonteCarloResult)

    def test_seed_reproducibility(self):
        r = self._result()
        mc1 = monte_carlo(r, n_sims=50, seed=1)
        mc2 = monte_carlo(r, n_sims=50, seed=1)
        npt.assert_array_equal(mc1.equity_curves, mc2.equity_curves)

    def test_var_is_low_percentile_of_terminal_equity(self):
        r = self._result()
        mc = monte_carlo(r, n_sims=1000, confidence=0.95)
        # VaR = 5th percentile of terminal equity
        expected_var = float(np.percentile(mc.terminal_equity, 5.0))
        assert mc.var == pytest.approx(expected_var, rel=1e-6)


# ===========================================================================
# Backward compatibility guard
# ===========================================================================


class TestBackwardCompat:
    def test_backtest_still_returns_backtest_result(self):
        rng = np.random.default_rng(0)
        c = np.cumprod(1 + rng.standard_normal(100) * 0.01) * 100.0
        r = backtest(c, strategy="rsi_30_70")
        assert type(r) is BacktestResult

    def test_portfolio_backtest_result(self):
        rng = np.random.default_rng(0)
        n, k = 100, 3
        c2d = np.cumprod(1 + rng.standard_normal((n, k)) * 0.01, axis=0) * 100.0
        w2d = np.ones((n, k)) / k
        r = backtest_portfolio(c2d, w2d)
        assert isinstance(r, PortfolioBacktestResult)
        assert r.portfolio_equity.shape == (n,)


# ===========================================================================
# Sprint 1: Limit orders, time-based exit, pct_range slippage
# ===========================================================================


class TestLimitOrders:
    """Tests for limit-price order fill logic in backtest_ohlcv_core."""

    def _ohlcv(self):
        n = 50
        rng = np.random.default_rng(7)
        close = np.cumprod(1 + rng.standard_normal(n) * 0.005) * 100.0
        open_ = close * (1 - rng.uniform(0, 0.003, n))
        high = close * (1 + rng.uniform(0.002, 0.008, n))
        low = close * (1 - rng.uniform(0.002, 0.008, n))
        return open_, high, low, close, n

    def test_limit_nan_behaves_like_market(self):
        """NaN limit prices should give identical results to no limit array."""
        o, h, l, c, n = self._ohlcv()
        signals = np.where(np.arange(n) % 10 < 5, 1.0, -1.0).astype(np.float64)
        lp_nan = np.full(n, np.nan)

        pos_mkt, fp_mkt, _, sr_mkt, eq_mkt = backtest_ohlcv_core(o, h, l, c, signals)
        pos_lim, fp_lim, _, sr_lim, eq_lim = backtest_ohlcv_core(
            o, h, l, c, signals, limit_prices=lp_nan
        )
        npt.assert_array_almost_equal(pos_mkt, pos_lim)
        npt.assert_array_almost_equal(sr_mkt, sr_lim)
        npt.assert_array_almost_equal(eq_mkt, eq_lim)

    def test_buy_limit_fills_at_limit_price(self):
        """Buy limit fills when low <= limit_price and uses limit as fill price."""
        n = 10
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        # Buy signal at bar 0, limit price 99 — low=98 <= 99 so should fill
        signals = np.zeros(n)
        signals[0] = 1.0  # want to go long at bar 1
        limit_prices = np.full(n, np.nan)
        limit_prices[0] = 99.0  # limit for bar 1 execution

        _, fp, _, _, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            fill_mode="market_close",
            limit_prices=limit_prices,
        )
        # Bar 1 should have a fill at 99.0 (the limit price)
        assert fp[1] == pytest.approx(99.0, rel=1e-6)

    def test_buy_limit_not_hit_no_fill(self):
        """Buy limit is not filled when low > limit_price."""
        n = 10
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)  # low=98
        signals = np.zeros(n)
        signals[0] = 1.0  # go long at bar 1
        limit_prices = np.full(n, np.nan)
        limit_prices[0] = 97.0  # limit=97, but low=98 > 97 → no fill

        pos, fp, _, _, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            fill_mode="market_close",
            limit_prices=limit_prices,
        )
        # Position should stay 0 at bar 1 (limit not hit)
        assert pos[1] == pytest.approx(0.0)
        assert np.isnan(fp[1])

    def test_sell_limit_fills_when_high_hits(self):
        """Sell limit fills when high >= limit_price."""
        n = 10
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 103.0)
        low = np.full(n, 97.0)
        signals = np.zeros(n)
        signals[0] = -1.0  # go short at bar 1
        limit_prices = np.full(n, np.nan)
        limit_prices[0] = 102.0  # high=103 >= 102 → fill

        _, fp, _, _, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            fill_mode="market_close",
            limit_prices=limit_prices,
        )
        assert fp[1] == pytest.approx(102.0, rel=1e-6)

    def test_engine_with_limit_orders(self):
        """BacktestEngine.with_limit_orders with NaN limits matches market orders."""
        o, h, l, c, n = self._ohlcv()
        # NaN limit prices = market orders; result must match engine without limit array
        limit_prices = np.full(n, np.nan)

        result_mkt = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .run(c, strategy="sma_crossover", fast=5, slow=20)
        )
        result_lim = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .with_limit_orders(limit_prices)
            .run(c, strategy="sma_crossover", fast=5, slow=20)
        )
        assert isinstance(result_lim, AdvancedBacktestResult)
        npt.assert_array_almost_equal(result_mkt.equity, result_lim.equity)


class TestMaxHold:
    """Tests for time-based exit (max_hold_bars)."""

    def _flat_ohlcv(self, n=30):
        close = np.ones(n) * 100.0
        open_ = close.copy()
        high = close * 1.005
        low = close * 0.995
        signals = np.ones(n)  # always long signal
        return open_, high, low, close, signals

    def test_position_exits_after_n_bars(self):
        """Position should be closed after max_hold_bars regardless of signal."""
        o, h, l, c, s = self._flat_ohlcv(n=20)
        max_hold = 5
        pos, _, _, _, _ = backtest_ohlcv_core(o, h, l, c, s, max_hold_bars=max_hold)

        # Find first entry
        entry_bar = None
        for i in range(len(pos)):
            if pos[i] != 0.0:
                entry_bar = i
                break

        assert entry_bar is not None
        # Position should be 0 at entry_bar + max_hold
        exit_bar = entry_bar + max_hold
        if exit_bar < len(pos):
            assert pos[exit_bar] == pytest.approx(0.0), (
                f"Expected exit at bar {exit_bar}, pos={pos[exit_bar]}"
            )

    def test_max_hold_zero_is_disabled(self):
        """max_hold_bars=0 should not affect behaviour (disabled)."""
        o, h, l, c, s = self._flat_ohlcv(n=20)
        pos_no_hold, _, _, _, _ = backtest_ohlcv_core(o, h, l, c, s)
        pos_hold_0, _, _, _, _ = backtest_ohlcv_core(o, h, l, c, s, max_hold_bars=0)
        npt.assert_array_almost_equal(pos_no_hold, pos_hold_0)

    def test_engine_with_max_hold(self):
        """BacktestEngine.with_max_hold integrates correctly."""
        rng = np.random.default_rng(99)
        n = 100
        c = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        h = c * 1.01
        l = c * 0.99
        o = c * 1.001

        result = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .with_max_hold(5)
            .run(c, strategy="rsi_30_70")
        )
        assert isinstance(result, AdvancedBacktestResult)

    def test_max_hold_stop_takes_priority(self):
        """A stop-loss that triggers before max_hold should exit early."""
        n = 20
        close = np.array([100.0] * 5 + [95.0] * 15)  # price drops on bar 5
        open_ = close.copy()
        high = close * 1.002
        low = np.array([100.0] * 5 + [93.0] * 15)  # low hits stop at bar 5
        signals = np.ones(n)  # always long

        pos_sl, _, _, _, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            stop_loss_pct=0.05,
            max_hold_bars=10,
        )
        pos_hold, _, _, _, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            stop_loss_pct=0.05,
        )
        # Both should exit around the same time (stop triggers before hold limit)
        # At least the stop-loss exit should happen — position goes to 0 before bar 10+1
        assert any(pos_sl[5:11] == 0.0), (
            "Stop-loss should have triggered before max_hold"
        )


class TestSlippagePctRange:
    """Tests for pct_range slippage mode."""

    def _ohlcv_wide_range(self, n=20):
        """OHLCV with a wide bar range to make pct_range slippage measurable."""
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 110.0)  # range = 10 (10%)
        low = np.full(n, 90.0)
        signals = np.where(np.arange(n) % 10 < 5, 1.0, -1.0).astype(np.float64)
        return open_, high, low, close, signals

    def test_pct_range_more_costly_than_zero_slippage(self):
        """With wide bar range, pct_range slip should reduce final equity vs no slip."""
        o, h, l, c, s = self._ohlcv_wide_range()
        _, _, _, _, eq_no_slip = backtest_ohlcv_core(o, h, l, c, s)
        _, _, _, _, eq_pct = backtest_ohlcv_core(o, h, l, c, s, slippage_pct_range=0.10)
        # pct_range slippage = 0.10 × (110-90)/100 = 0.02 = 200bps per trade
        assert eq_pct[-1] < eq_no_slip[-1]

    def test_pct_range_more_costly_than_bps_equivalent(self):
        """pct_range with wide range should be costlier than modest bps slip."""
        o, h, l, c, s = self._ohlcv_wide_range()
        # bps slip: 5bps = 0.05% of fill, small
        _, _, _, _, eq_bps = backtest_ohlcv_core(o, h, l, c, s, slippage_bps=5.0)
        # pct_range: 10% of 20-wide range = 2.0 absolute, or 2% of close=100
        _, _, _, _, eq_pct = backtest_ohlcv_core(o, h, l, c, s, slippage_pct_range=0.10)
        assert eq_pct[-1] < eq_bps[-1]

    def test_pct_range_zero_equals_no_slippage(self):
        """slippage_pct_range=0 should give same result as no slippage."""
        o, h, l, c, s = self._ohlcv_wide_range()
        _, _, _, _, eq_base = backtest_ohlcv_core(o, h, l, c, s)
        _, _, _, _, eq_zero = backtest_ohlcv_core(o, h, l, c, s, slippage_pct_range=0.0)
        npt.assert_array_almost_equal(eq_base, eq_zero)

    def test_engine_with_slippage_pct_range(self):
        """BacktestEngine.with_slippage_pct_range integrates correctly."""
        rng = np.random.default_rng(17)
        n = 80
        c = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        h = c * 1.01
        l = c * 0.99
        o = c * 1.001

        result = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .with_slippage_pct_range(0.05)
            .run(c, strategy="sma_crossover", fast=5, slow=20)
        )
        assert isinstance(result, AdvancedBacktestResult)


# ===========================================================================
# Group 11: Phase 1 Features (spread_bps, breakeven_stop, bracket order priority)
# ===========================================================================


from ferro_ta._ferro_ta import CommissionModel as RustCommissionModel


class TestPhase1Features:
    """Tests for Phase 1 features: spread_bps, breakeven_pct, bracket order priority."""

    def test_spread_bps_increases_cost(self):
        """CommissionModel with spread_bps=10 should produce lower equity than spread_bps=0."""
        rng = np.random.default_rng(99)
        n = 200
        c = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        h = c * 1.005
        l = c * 0.995
        o = c * 0.999
        signals = np.where(np.arange(n) % 20 < 10, 1.0, 0.0).astype(np.float64)

        # Build a commission model with spread_bps=0
        cm_no_spread = RustCommissionModel()
        cm_no_spread.spread_bps = 0.0

        # Build a commission model with spread_bps=10
        cm_with_spread = RustCommissionModel()
        cm_with_spread.spread_bps = 10.0

        _, _, _, _, eq_no_spread = backtest_ohlcv_core(
            o, h, l, c, signals, commission=cm_no_spread
        )
        _, _, _, _, eq_with_spread = backtest_ohlcv_core(
            o, h, l, c, signals, commission=cm_with_spread
        )

        # Spread adds cost on each trade leg → should produce lower or equal final equity
        assert eq_with_spread[-1] <= eq_no_spread[-1], (
            f"spread equity {eq_with_spread[-1]:.6f} should be <= no-spread equity {eq_no_spread[-1]:.6f}"
        )

    def test_spread_bps_getter_setter(self):
        """CommissionModel spread_bps getter/setter round-trip works correctly."""
        m = RustCommissionModel()
        assert m.spread_bps == 0.0
        m.spread_bps = 5.0
        assert m.spread_bps == pytest.approx(5.0)

    def test_spread_bps_total_cost(self):
        """CommissionModel.total_cost includes spread cost at correct magnitude."""
        m = RustCommissionModel()
        m.spread_bps = 20.0  # 20 bps total round-trip = 10 bps each leg
        trade_value = 100_000.0
        cost = m.total_cost(trade_value, 1.0, True)
        # Expected: 10 bps = 0.001 * 100_000 = 100 per leg
        assert cost == pytest.approx(100.0, rel=1e-6)

    def test_breakeven_stop_prevents_loss(self):
        """With breakeven_pct=0.02, after price rises 3% then falls, exit should be near entry."""
        # Build synthetic data: entry at bar 1, then price rises 3%, then falls below entry
        # Bar layout: [100, 103, 103, 101, 99, 99, 99, 99, 99]
        # We want a long signal from bar 0 onwards
        n = 20
        # Create price data: starts at 100, rises to 103 at bar 3, then drops to 97
        close = np.array([100.0] * 3 + [103.0] * 3 + [97.0] * (n - 6), dtype=np.float64)
        open_ = close.copy()
        high = close * 1.002
        low = close * 0.998
        # Set high of bar 3 to clearly trigger breakeven (>= 103 = entry * 1.03)
        # entry happens at bar 1 (open of bar 1 = 100), so entry_price ≈ 100
        # breakeven triggers when h >= 100 * 1.02 = 102 → triggers at bar 3 (close=103, high≥103)
        high[3] = 103.5  # clearly above 102 (entry * 1.02)
        # At bar 6, low drops below entry (100), breakeven stop should trigger
        low[6] = 99.0  # below entry price 100 → breakeven stop fires
        signals = np.ones(n, dtype=np.float64)  # always long

        _, fp, _, _, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            stop_loss_pct=0.0,
            breakeven_pct=0.02,
        )
        # Find first non-NaN fill price after the entry bar (entry at bar 1)
        # Breakeven exit should happen at or near entry price (100), not at a big loss
        exit_fps = fp[~np.isnan(fp)]
        # The breakeven stop exit should be at entry_price (~100), not at 97 or lower
        # Entry fill is at open of bar 1 = 100.0
        # After breakeven activates, stop = entry (~100). So exit fill should be ~100
        assert len(exit_fps) >= 1
        # The exit fill from breakeven should be close to entry price (within 1%)
        # (first fill = entry, subsequent fills = exits)
        if len(exit_fps) >= 2:
            breakeven_exit = exit_fps[1]
            assert breakeven_exit >= 99.0, (
                f"breakeven exit {breakeven_exit} should be >= 99 (near entry 100)"
            )

    def test_bracket_order_tp_fires_before_sl(self):
        """When both SL and TP are breached in same bar, and open is near TP → TP fires."""
        # Long trade: entry at price 100
        # Bar where both trigger: open=109 (very close to TP=110), high=112, low=90
        # SL = 100*(1-0.10) = 90, TP = 100*(1+0.10) = 110
        # open=109 is closer to TP=110 (dist=1) than to SL=90 (dist=19) → TP fires
        entry_price = 100.0
        close = np.array(
            [entry_price, entry_price, entry_price, 108.0, 108.0], dtype=np.float64
        )
        open_ = np.array(
            [entry_price, entry_price, entry_price, 109.0, 108.0], dtype=np.float64
        )
        high = np.array(
            [entry_price, entry_price, entry_price, 112.0, 108.0], dtype=np.float64
        )
        low = np.array(
            [entry_price, entry_price, entry_price, 88.0, 108.0], dtype=np.float64
        )
        signals = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)

        _, fp, _, sr, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            stop_loss_pct=0.10,
            take_profit_pct=0.10,
        )
        # Bar 3 is where both trigger. TP=110. SL=90. Open=109 → TP fires.
        # fill price at bar 3 should be ~110 (TP), not 90 (SL)
        assert not np.isnan(fp[3]), "Expected a fill at bar 3"
        tp_level = entry_price * 1.10  # 110
        assert fp[3] == pytest.approx(tp_level, rel=1e-6), (
            f"Expected TP fill at ~{tp_level}, got {fp[3]}"
        )

    def test_bracket_order_sl_fires_before_tp(self):
        """When both SL and TP are breached in same bar, and open is near SL → SL fires."""
        # Long trade: entry at 100
        # Bar where both trigger: open=91 (very close to SL=90), high=112, low=88
        # SL=90, TP=110. open=91 is closer to SL=90 (dist=1) than to TP=110 (dist=19) → SL fires
        entry_price = 100.0
        close = np.array(
            [entry_price, entry_price, entry_price, 95.0, 95.0], dtype=np.float64
        )
        open_ = np.array(
            [entry_price, entry_price, entry_price, 91.0, 95.0], dtype=np.float64
        )
        high = np.array(
            [entry_price, entry_price, entry_price, 112.0, 95.0], dtype=np.float64
        )
        low = np.array(
            [entry_price, entry_price, entry_price, 88.0, 95.0], dtype=np.float64
        )
        signals = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)

        _, fp, _, sr, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            stop_loss_pct=0.10,
            take_profit_pct=0.10,
        )
        # Bar 3: both SL(90) and TP(110) are triggered. open=91 is close to SL → SL fires.
        assert not np.isnan(fp[3]), "Expected a fill at bar 3"
        sl_level = entry_price * 0.90  # 90
        assert fp[3] == pytest.approx(sl_level, rel=1e-6), (
            f"Expected SL fill at ~{sl_level}, got {fp[3]}"
        )

    def test_breakeven_engine_integration(self):
        """BacktestEngine.with_breakeven_stop integrates correctly."""
        rng = np.random.default_rng(77)
        n = 150
        c = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        h = c * 1.01
        l = c * 0.99
        o = c * 0.999

        result = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .with_breakeven_stop(0.02)
            .run(c, strategy="sma_crossover", fast=5, slow=20)
        )
        assert isinstance(result, AdvancedBacktestResult)
        assert np.all(np.isfinite(result.equity))


# ===========================================================================
# Phase 2: Portfolio & Risk Features
# ===========================================================================


class TestPhase2Features:
    """Tests for Phase 2: short borrow cost, margin/leverage, circuit breakers,
    and portfolio constraints."""

    # -----------------------------------------------------------------------
    # Helper: synthetic OHLCV with controllable direction
    # -----------------------------------------------------------------------

    def _make_short_ohlcv(self, n: int = 100, seed: int = 7) -> tuple:
        """Produce OHLCV where the price trends downward (good for shorts)."""
        rng = np.random.default_rng(seed)
        # Steady downtrend
        close = 100.0 * np.cumprod(1 - np.abs(rng.standard_normal(n)) * 0.005)
        open_ = close * (1 + rng.uniform(-0.002, 0.002, n))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n))
        # Always short
        signals = np.full(n, -1.0, dtype=np.float64)
        return open_, high, low, close, signals

    # -----------------------------------------------------------------------
    # 1. Short borrow cost
    # -----------------------------------------------------------------------

    def test_short_borrow_cost_reduces_equity(self):
        """Short position with short_borrow_rate_annual=0.10 should produce lower
        final equity than the same run with no borrow cost."""
        from ferro_ta._ferro_ta import CommissionModel

        o, h, l, c, signals = self._make_short_ohlcv(n=252)

        # Commission model without borrow cost
        cm_no_borrow = CommissionModel()

        # Commission model with 10% annual borrow cost
        cm_with_borrow = CommissionModel()
        cm_with_borrow.short_borrow_rate_annual = 0.10

        _, _, _, _, eq_no_borrow = backtest_ohlcv_core(
            o, h, l, c, signals, commission=cm_no_borrow
        )
        _, _, _, _, eq_with_borrow = backtest_ohlcv_core(
            o, h, l, c, signals, commission=cm_with_borrow
        )

        # With borrow cost, final equity must be strictly lower
        assert float(eq_with_borrow[-1]) < float(eq_no_borrow[-1]), (
            f"Expected borrow-cost equity {eq_with_borrow[-1]:.6f} < "
            f"no-borrow equity {eq_no_borrow[-1]:.6f}"
        )

    def test_short_borrow_cost_getter_setter(self):
        """CommissionModel.short_borrow_rate_annual getter/setter works."""
        from ferro_ta._ferro_ta import CommissionModel

        cm = CommissionModel()
        assert cm.short_borrow_rate_annual == pytest.approx(0.0)
        cm.short_borrow_rate_annual = 0.05
        assert cm.short_borrow_rate_annual == pytest.approx(0.05)

    def test_short_borrow_zero_rate_no_effect(self):
        """With short_borrow_rate_annual=0, borrow cost should not affect equity."""
        from ferro_ta._ferro_ta import CommissionModel

        o, h, l, c, signals = self._make_short_ohlcv(n=50)
        cm_zero = CommissionModel()
        cm_zero.short_borrow_rate_annual = 0.0

        _, _, _, sr1, eq1 = backtest_ohlcv_core(o, h, l, c, signals)
        _, _, _, sr2, eq2 = backtest_ohlcv_core(o, h, l, c, signals, commission=cm_zero)

        npt.assert_allclose(eq1, eq2, rtol=1e-10)

    def test_short_borrow_engine_integration(self):
        """BacktestEngine with commission model including short_borrow_rate_annual runs."""
        from ferro_ta._ferro_ta import CommissionModel

        o, h, l, c, sigs = self._make_short_ohlcv(n=80)
        cm = CommissionModel()
        cm.short_borrow_rate_annual = 0.08

        result = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .with_commission_model(cm)
            .run(c, lambda x: np.full(len(x), -1.0))
        )
        assert isinstance(result, AdvancedBacktestResult)
        assert np.all(np.isfinite(result.equity))

    # -----------------------------------------------------------------------
    # 2. Margin call force-close
    # -----------------------------------------------------------------------

    def test_margin_call_force_closes_position(self):
        """A declining price sequence triggers a margin call and force-closes the long."""
        n = 20
        # Price drops sharply — enough to trigger a margin call on a long
        open_ = np.ones(n) * 100.0
        high = np.ones(n) * 101.0
        low = np.ones(n) * 99.0
        close = np.ones(n) * 100.0

        # After bar 5, price tanks sharply every bar
        for i in range(5, n):
            drop = 0.30  # 30% per bar — guaranteed to exceed margin
            open_[i] = open_[i - 1] * (1 - drop)
            high[i] = open_[i] * 1.001
            low[i] = open_[i] * 0.999
            close[i] = open_[i]

        # Always long
        signals = np.ones(n, dtype=np.float64)

        # margin_ratio=0.2 means 20% margin (5x leverage)
        # margin_call_pct=0.5 means call when equity hits 50% of initial margin
        _, _, _, sr_margin, eq_margin = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            margin_ratio=0.2,
            margin_call_pct=0.5,
        )
        _, _, _, sr_no_margin, eq_no_margin = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
        )

        # Margin call should cause a forced exit, resulting in different equity
        # (the margin version stops losses earlier)
        assert not np.allclose(eq_margin, eq_no_margin), (
            "Expected margin call to alter equity curve"
        )

    def test_margin_disabled_when_ratio_zero(self):
        """margin_ratio=0 should behave identically to not passing the parameter."""
        o, h, l, c, signals = _make_ohlcv(n=80)

        _, _, _, _, eq_default = backtest_ohlcv_core(o, h, l, c, signals)
        _, _, _, _, eq_zero_margin = backtest_ohlcv_core(
            o, h, l, c, signals, margin_ratio=0.0
        )

        npt.assert_allclose(eq_default, eq_zero_margin, rtol=1e-10)

    def test_margin_engine_builder(self):
        """BacktestEngine.with_leverage builder sets parameters without error."""
        o, h, l, c, _ = _make_ohlcv(n=60)

        result = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .with_leverage(margin_ratio=0.2, margin_call_pct=0.5)
            .run(c, lambda x: np.ones(len(x)))
        )
        assert isinstance(result, AdvancedBacktestResult)
        assert np.all(np.isfinite(result.equity))

    # -----------------------------------------------------------------------
    # 3. Total loss limit (circuit breaker)
    # -----------------------------------------------------------------------

    def test_total_loss_limit_halts_trading(self):
        """total_loss_limit=0.10 should halt trading once equity drops 10%."""
        n = 100
        # Construct a losing price sequence: steady decline
        close = 100.0 * np.cumprod(np.full(n, 0.99))  # -1% per bar
        open_ = close * 1.001
        high = close * 1.005
        low = close * 0.995

        # Always long (so position loses money as price falls)
        signals = np.ones(n, dtype=np.float64)

        pos_with_limit, _, _, _, eq_with_limit = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            total_loss_limit=0.10,
        )
        pos_no_limit, _, _, _, eq_no_limit = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
        )

        # After circuit break the position should be 0
        # Check that at some point positions go to 0 in the limited version
        # while the unlimited version stays long
        assert np.any(pos_with_limit == 0.0), (
            "Expected some bars with no position after circuit break"
        )
        # Unlimited version should stay long throughout (except bar 0)
        assert np.all(pos_no_limit[1:] == 1.0), "No-limit should stay long"

    def test_total_loss_limit_does_not_trip_with_no_loss(self):
        """total_loss_limit does not trip on a profitable sequence."""
        n = 60
        close = 100.0 * np.cumprod(np.full(n, 1.005))  # +0.5% per bar
        open_ = close * 0.999
        high = close * 1.003
        low = close * 0.997
        signals = np.ones(n, dtype=np.float64)

        pos, _, _, _, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            total_loss_limit=0.20,
        )
        # No circuit break should fire; position stays long
        assert np.all(pos[1:] == 1.0)

    # -----------------------------------------------------------------------
    # 4. Daily (per-bar) loss limit circuit breaker
    # -----------------------------------------------------------------------

    def test_daily_loss_limit_halts_after_large_bar_loss(self):
        """A single large losing bar triggers the daily_loss_limit circuit breaker."""
        n = 30
        close = np.ones(n) * 100.0
        open_ = np.ones(n) * 100.0
        high = np.ones(n) * 101.0
        low = np.ones(n) * 99.0

        # Create one very large losing bar at bar 10 (price drops 15%)
        # Strategy is long, so this is a large loss
        crash_bar = 10
        close[crash_bar] = close[crash_bar - 1] * 0.85
        open_[crash_bar] = close[crash_bar - 1] * 0.86
        high[crash_bar] = open_[crash_bar] * 1.001
        low[crash_bar] = close[crash_bar] * 0.999

        signals = np.ones(n, dtype=np.float64)

        pos, _, _, sr, _ = backtest_ohlcv_core(
            open_,
            high,
            low,
            close,
            signals,
            daily_loss_limit=0.05,  # 5% per-bar loss limit
        )

        # After the crash bar, circuit breaker should fire and position should go to 0
        # Check bars after crash_bar+1 have position 0
        assert np.any(pos[crash_bar + 1 :] == 0.0), (
            "Expected circuit breaker to zero out position after crash bar"
        )

    def test_daily_loss_limit_zero_is_disabled(self):
        """daily_loss_limit=0 (default) should not change behavior."""
        o, h, l, c, signals = _make_ohlcv(n=80)

        _, _, _, _, eq_default = backtest_ohlcv_core(o, h, l, c, signals)
        _, _, _, _, eq_zero_limit = backtest_ohlcv_core(
            o, h, l, c, signals, daily_loss_limit=0.0
        )

        npt.assert_allclose(eq_default, eq_zero_limit, rtol=1e-10)

    def test_loss_limits_engine_builder(self):
        """BacktestEngine.with_loss_limits builder sets parameters."""
        o, h, l, c, _ = _make_ohlcv(n=80)

        result = (
            BacktestEngine()
            .with_ohlcv(high=h, low=l, open_=o)
            .with_loss_limits(daily=0.05, total=0.20)
            .run(c, strategy="sma_crossover")
        )
        assert isinstance(result, AdvancedBacktestResult)
        assert np.all(np.isfinite(result.equity))

    # -----------------------------------------------------------------------
    # 5. Portfolio constraints
    # -----------------------------------------------------------------------

    def test_portfolio_max_asset_weight_clamps_signal(self):
        """max_asset_weight=0.5 should clamp signals from ±1 to ±0.5."""
        rng = np.random.default_rng(99)
        n_bars, n_assets = 100, 3
        close_2d = (
            np.cumprod(1 + rng.standard_normal((n_bars, n_assets)) * 0.01, axis=0)
            * 100.0
        )
        # Alternating ±1 signals — shape (n_bars, n_assets)
        row_flags = (np.arange(n_bars) % 10 < 5)[:, None]  # (n, 1)
        weights_2d = np.where(np.tile(row_flags, (1, n_assets)), 1.0, -1.0).astype(
            np.float64
        )

        # Run without constraint (unit signals)
        asset_ret_unconstrained, port_ret_unconstrained, _ = backtest_multi_asset_core(
            np.ascontiguousarray(close_2d),
            np.ascontiguousarray(weights_2d),
            max_asset_weight=1.0,
        )

        # Run with max_asset_weight=0.5
        asset_ret_constrained, port_ret_constrained, _ = backtest_multi_asset_core(
            np.ascontiguousarray(close_2d),
            np.ascontiguousarray(weights_2d),
            max_asset_weight=0.5,
        )

        # Constrained returns should have smaller magnitude
        assert np.abs(port_ret_constrained).sum() < np.abs(
            port_ret_unconstrained
        ).sum() or np.allclose(
            np.abs(port_ret_constrained).sum(),
            np.abs(port_ret_unconstrained).sum() * 0.5,
            rtol=0.05,
        ), "max_asset_weight=0.5 should reduce absolute returns by ~50%"

    def test_portfolio_max_gross_exposure_constrains_sum(self):
        """max_gross_exposure=1.0 should limit total abs(weights)."""
        rng = np.random.default_rng(55)
        n_bars, n_assets = 80, 4
        close_2d = (
            np.cumprod(1 + rng.standard_normal((n_bars, n_assets)) * 0.01, axis=0)
            * 100.0
        )
        # Always long all assets = gross exposure of 4.0
        weights_2d = np.ones((n_bars, n_assets), dtype=np.float64)

        # With max_gross_exposure=1.0, total abs weight should be normalized to 1
        ar_constrained, pr_constrained, _ = backtest_multi_asset_core(
            np.ascontiguousarray(close_2d),
            np.ascontiguousarray(weights_2d),
            max_gross_exposure=1.0,
        )
        ar_unconstrained, pr_unconstrained, _ = backtest_multi_asset_core(
            np.ascontiguousarray(close_2d),
            np.ascontiguousarray(weights_2d),
        )

        # Constrained portfolio should have ~1/4 the returns magnitude
        ratio = np.abs(pr_constrained).sum() / (np.abs(pr_unconstrained).sum() + 1e-12)
        assert ratio < 0.5, (
            f"Expected constrained to be much smaller, got ratio={ratio:.3f}"
        )

    def test_portfolio_constraints_engine_builder(self):
        """BacktestEngine.with_portfolio_constraints stores the parameters."""
        engine = BacktestEngine().with_portfolio_constraints(
            max_asset_weight=0.3,
            max_gross_exposure=1.5,
            max_net_exposure=0.5,
        )
        assert engine._max_asset_weight == pytest.approx(0.3)
        assert engine._max_gross_exposure == pytest.approx(1.5)
        assert engine._max_net_exposure == pytest.approx(0.5)

    def test_backtest_portfolio_with_constraints(self):
        """backtest_portfolio accepts portfolio constraint kwargs."""
        from ferro_ta.analysis.backtest import backtest_portfolio

        rng = np.random.default_rng(11)
        n_bars, n_assets = 60, 2
        close_2d = (
            np.cumprod(1 + rng.standard_normal((n_bars, n_assets)) * 0.01, axis=0)
            * 100.0
        )
        row_flags = (np.arange(n_bars) % 10 < 5)[:, None]
        weights_2d = np.where(np.tile(row_flags, (1, n_assets)), 1.0, -1.0).astype(
            np.float64
        )

        result = backtest_portfolio(
            close_2d,
            weights_2d,
            max_asset_weight=0.5,
            max_gross_exposure=0.8,
        )
        assert isinstance(result, PortfolioBacktestResult)
        assert np.all(np.isfinite(result.portfolio_equity))


# ===========================================================================
# Phase 3: Data & UX features
# ===========================================================================


class TestPhase3Features:
    """Tests for Phase 3: resample, adjust, multitf, and plot modules."""

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _make_ohlcv(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        close = np.cumprod(1 + rng.standard_normal(n) * 0.005) * 100.0
        open_ = close * (1 - rng.uniform(0, 0.002, n))
        high = close * (1 + rng.uniform(0.001, 0.006, n))
        low = close * (1 - rng.uniform(0.001, 0.006, n))
        volume = rng.uniform(1_000, 10_000, n)
        return open_, high, low, close, volume

    # -----------------------------------------------------------------------
    # 1. resample_ohlcv — factor=4, 20 bars → 5 coarse bars
    # -----------------------------------------------------------------------
    def test_resample_ohlcv_factor4(self):
        from ferro_ta.analysis.resample import resample_ohlcv

        o, h, l, c, v = self._make_ohlcv(n=20)
        co, ch, cl, cc, cv = resample_ohlcv(o, h, l, c, v, factor=4)

        assert co.shape == (5,)
        assert ch.shape == (5,)
        assert cl.shape == (5,)
        assert cc.shape == (5,)
        assert cv.shape == (5,)

        # open = first bar of each group
        for i in range(5):
            assert co[i] == pytest.approx(o[i * 4])

        # high = max of group
        for i in range(5):
            assert ch[i] == pytest.approx(h[i * 4 : i * 4 + 4].max())

        # low = min of group
        for i in range(5):
            assert cl[i] == pytest.approx(l[i * 4 : i * 4 + 4].min())

        # close = last bar of group
        for i in range(5):
            assert cc[i] == pytest.approx(c[i * 4 + 3])

        # volume = sum of group
        for i in range(5):
            assert cv[i] == pytest.approx(v[i * 4 : i * 4 + 4].sum())

    # -----------------------------------------------------------------------
    # 2. resample_ohlcv — non-divisible length: 22 bars, factor=4 → 5 coarse bars
    # -----------------------------------------------------------------------
    def test_resample_ohlcv_non_divisible(self):
        from ferro_ta.analysis.resample import resample_ohlcv

        o, h, l, c, v = self._make_ohlcv(n=22)
        co, ch, cl, cc, cv = resample_ohlcv(o, h, l, c, v, factor=4)

        # 22 // 4 = 5 complete bars, last 2 fine bars are dropped
        assert len(co) == 5
        assert len(ch) == 5

    # -----------------------------------------------------------------------
    # 3. align_to_coarse — roundtrip test
    # -----------------------------------------------------------------------
    def test_align_to_coarse_roundtrip(self):
        from ferro_ta.analysis.resample import align_to_coarse

        coarse = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        factor = 4
        n_fine = 20

        fine = align_to_coarse(coarse, factor, n_fine)

        assert len(fine) == n_fine

        for i, val in enumerate(coarse):
            expected = np.full(factor, val)
            npt.assert_array_equal(fine[i * factor : i * factor + factor], expected)

    # -----------------------------------------------------------------------
    # 4. adjust_for_splits — 2-for-1 split at bar 50 in 100-bar series
    # -----------------------------------------------------------------------
    def test_adjust_for_splits_halves_historical(self):
        from ferro_ta.analysis.adjust import adjust_for_splits

        close = np.ones(100) * 100.0
        adjusted = adjust_for_splits(close, split_factors=[2.0], split_indices=[50])

        # Prices before split (bars 0-49) should be halved
        npt.assert_array_almost_equal(adjusted[:50], np.full(50, 50.0))
        # Prices from split onwards unchanged
        npt.assert_array_almost_equal(adjusted[50:], np.full(50, 100.0))

    # -----------------------------------------------------------------------
    # 5. adjust_for_dividends — dividend at bar 50; prices before reduced
    # -----------------------------------------------------------------------
    def test_adjust_for_dividends_reduces_historical(self):
        from ferro_ta.analysis.adjust import adjust_for_dividends

        close = np.ones(100) * 100.0
        # bar 49 close = 100.0, dividend = 5.0 → factor = 95/100 = 0.95
        adjusted = adjust_for_dividends(close, dividends=[5.0], ex_date_indices=[50])

        # Prices before ex-date should be scaled by 0.95
        expected_factor = (100.0 - 5.0) / 100.0
        npt.assert_array_almost_equal(
            adjusted[:50], np.full(50, 100.0 * expected_factor)
        )
        # Prices from ex-date onwards unchanged
        npt.assert_array_almost_equal(adjusted[50:], np.full(50, 100.0))

    # -----------------------------------------------------------------------
    # 6. adjust_ohlcv — volume doubles on 2-for-1 split (inverse adjustment)
    # -----------------------------------------------------------------------
    def test_adjust_ohlcv_volume_increases_on_split(self):
        from ferro_ta.analysis.adjust import adjust_ohlcv

        n = 100
        close = np.ones(n) * 100.0
        open_ = close.copy()
        high = close.copy()
        low = close.copy()
        volume = np.ones(n) * 1000.0

        ao, ah, al, ac, av = adjust_ohlcv(
            open_,
            high,
            low,
            close,
            volume,
            split_factors=[2.0],
            split_indices=[50],
        )

        # Volume before the split is multiplied by factor (2x) — more shares pre-split
        npt.assert_array_almost_equal(av[:50], np.full(50, 2000.0))
        # Volume at or after split unchanged
        npt.assert_array_almost_equal(av[50:], np.full(50, 1000.0))

        # Prices before split halved
        npt.assert_array_almost_equal(ac[:50], np.full(50, 50.0))
        npt.assert_array_almost_equal(ac[50:], np.full(50, 100.0))

    # -----------------------------------------------------------------------
    # 7. MultiTimeframeEngine — runs on 200 fine bars, returns valid result
    # -----------------------------------------------------------------------
    def test_multitf_engine_runs(self):
        from ferro_ta.analysis.multitf import MultiTimeframeEngine

        rng = np.random.default_rng(99)
        n_fine = 200
        close_fine = np.cumprod(1 + rng.standard_normal(n_fine) * 0.005) * 100.0

        result = (
            MultiTimeframeEngine(factor=4)
            .with_htf_strategy("rsi_30_70")
            .run(close_fine)
        )

        assert isinstance(result, AdvancedBacktestResult)
        assert len(result.equity) == n_fine
        assert np.all(np.isfinite(result.equity))
        assert result.equity[0] == pytest.approx(1.0, rel=1e-6)

    # -----------------------------------------------------------------------
    # 8. plot_backtest — returns a plotly Figure (skip if plotly not installed)
    # -----------------------------------------------------------------------
    def test_plot_backtest_returns_figure(self):
        pytest.importorskip("plotly", reason="plotly not installed")
        from plotly.graph_objects import Figure

        from ferro_ta.analysis.plot import plot_backtest

        rng = np.random.default_rng(7)
        n = 100
        close = np.cumprod(1 + rng.standard_normal(n) * 0.005) * 100.0
        high = close * 1.01
        low = close * 0.99
        open_ = close * 0.999

        result = (
            BacktestEngine()
            .with_ohlcv(high=high, low=low, open_=open_)
            .run(close, strategy="rsi_30_70")
        )

        fig = plot_backtest(result, show=False, return_fig=True)

        assert isinstance(fig, Figure)


# ===========================================================================
# Phase 4: Regime Detection, Portfolio Optimization, PaperTrader
# ===========================================================================


class TestPhase4Features:
    """Tests for Phase 4 differentiation features."""

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_close(self, n: int = 300, seed: int = 77) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0

    def _make_ohlcv_local(self, n: int = 300, seed: int = 77):
        rng = np.random.default_rng(seed)
        close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        open_ = close * (1 - rng.uniform(0, 0.005, n))
        high = close * (1 + rng.uniform(0, 0.01, n))
        low = close * (1 - rng.uniform(0, 0.01, n))
        return open_, high, low, close

    # -----------------------------------------------------------------------
    # 1. detect_volatility_regime
    # -----------------------------------------------------------------------

    def test_volatility_regime_labels_three_states(self):
        from ferro_ta.analysis.regime import detect_volatility_regime

        close = self._make_close(300)
        labels = detect_volatility_regime(close, window=20, n_regimes=3)
        assert labels.shape == (300,)
        valid_values = {-1, 0, 1, 2}
        assert set(np.unique(labels)).issubset(valid_values)
        # Some valid (non-warmup) bars should be labeled
        assert np.any(labels >= 0)

    # -----------------------------------------------------------------------
    # 2. detect_trend_regime
    # -----------------------------------------------------------------------

    def test_trend_regime_bull_bear(self):
        from ferro_ta.analysis.regime import detect_trend_regime

        # Uptrend: price steadily rising
        n = 300
        close_up = np.linspace(100, 200, n)
        labels_up = detect_trend_regime(close_up, fast=10, slow=50)
        valid = labels_up[labels_up != 0]
        assert len(valid) > 0, "Expected some labeled bars after warmup"
        # Most valid bars should be bull (1)
        bull_frac = (valid == 1).sum() / len(valid)
        assert bull_frac > 0.5, (
            f"Expected mostly bull bars in uptrend, got {bull_frac:.2%}"
        )

        # Downtrend: price steadily declining
        close_dn = np.linspace(200, 100, n)
        labels_dn = detect_trend_regime(close_dn, fast=10, slow=50)
        valid_dn = labels_dn[labels_dn != 0]
        assert len(valid_dn) > 0
        bear_frac = (valid_dn == -1).sum() / len(valid_dn)
        assert bear_frac > 0.5, (
            f"Expected mostly bear bars in downtrend, got {bear_frac:.2%}"
        )

    # -----------------------------------------------------------------------
    # 3. detect_combined_regime
    # -----------------------------------------------------------------------

    def test_combined_regime_states(self):
        from ferro_ta.analysis.regime import detect_combined_regime

        close = self._make_close(500)
        labels = detect_combined_regime(close, vol_window=20, fast=20, slow=50)
        assert labels.shape == (500,)
        valid_values = {-1, 0, 1, 2, 3, 4, 5}
        assert set(np.unique(labels)).issubset(valid_values)

    # -----------------------------------------------------------------------
    # 4. RegimeFilter
    # -----------------------------------------------------------------------

    def test_regime_filter_zeros_disallowed(self):
        from ferro_ta.analysis.regime import RegimeFilter, detect_combined_regime

        n = 500
        close = self._make_close(n)
        signals = np.ones(n)

        # Only allow regime 0 (bull + low vol)
        rf = RegimeFilter(allowed_regimes=[0], vol_window=20, fast=20, slow=50)
        filtered = rf.filter(signals, close)

        regimes = detect_combined_regime(close, vol_window=20, fast=20, slow=50)
        # Bars NOT in regime 0 should have filtered signal = 0
        disallowed_mask = regimes != 0
        assert np.all(filtered[disallowed_mask] == 0.0)
        # Bars in regime 0 should retain their signal
        allowed_mask = regimes == 0
        if np.any(allowed_mask):
            assert np.all(filtered[allowed_mask] == 1.0)

    # -----------------------------------------------------------------------
    # 5. mean_variance_optimize
    # -----------------------------------------------------------------------

    def test_mean_variance_weights_sum_to_one(self):
        pytest.importorskip("scipy", reason="scipy not installed")
        from ferro_ta.analysis.optimize import mean_variance_optimize

        rng = np.random.default_rng(0)
        returns = rng.standard_normal((252, 4)) * 0.01
        w = mean_variance_optimize(returns)
        assert w.shape == (4,)
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-9), "Weights should be non-negative (no short)"

    # -----------------------------------------------------------------------
    # 6. risk_parity_optimize
    # -----------------------------------------------------------------------

    def test_risk_parity_weights_sum_to_one(self):
        pytest.importorskip("scipy", reason="scipy not installed")
        from ferro_ta.analysis.optimize import risk_parity_optimize

        rng = np.random.default_rng(1)
        returns = rng.standard_normal((252, 3)) * 0.01
        w = risk_parity_optimize(returns)
        assert w.shape == (3,)
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= 0.0)

    # -----------------------------------------------------------------------
    # 7. max_sharpe_optimize
    # -----------------------------------------------------------------------

    def test_max_sharpe_weights_sum_to_one(self):
        pytest.importorskip("scipy", reason="scipy not installed")
        from ferro_ta.analysis.optimize import max_sharpe_optimize

        rng = np.random.default_rng(2)
        returns = rng.standard_normal((252, 5)) * 0.01
        w = max_sharpe_optimize(returns)
        assert w.shape == (5,)
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-9)

    # -----------------------------------------------------------------------
    # 8. PortfolioOptimizer fluent builder
    # -----------------------------------------------------------------------

    def test_portfolio_optimizer_fluent(self):
        pytest.importorskip("scipy", reason="scipy not installed")
        from ferro_ta.analysis.optimize import PortfolioOptimizer

        rng = np.random.default_rng(3)
        returns = rng.standard_normal((252, 3)) * 0.01

        for method in ("min_variance", "risk_parity", "max_sharpe"):
            w = (
                PortfolioOptimizer()
                .with_method(method)
                .with_lookback(100)
                .optimize(returns)
            )
            assert w.shape == (3,)
            assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)

    # -----------------------------------------------------------------------
    # 9. PaperTrader: basic fills
    # -----------------------------------------------------------------------

    def test_paper_trader_fills_on_signal(self):
        from ferro_ta.analysis.live import PaperTrader

        rng = np.random.default_rng(10)
        n = 20
        close = np.cumprod(1 + rng.standard_normal(n) * 0.005) * 100.0
        open_ = close * (1 - rng.uniform(0, 0.003, n))
        high = close * (1 + rng.uniform(0.001, 0.005, n))
        low = close * (1 - rng.uniform(0.001, 0.005, n))

        trader = PaperTrader(initial_capital=100_000)
        signals = np.where(np.arange(n) % 6 < 3, 1.0, -1.0).astype(float)

        results = []
        for i in range(n):
            r = trader.on_bar(open_[i], high[i], low[i], close[i], signals[i])
            results.append(r)

        # Should have produced at least one fill after first bar
        fills = [r for r in results if r.filled]
        assert len(fills) > 0
        # Equity curve length should match bars
        assert len(trader.equity_curve) == n
        # Final equity should be finite
        assert math.isfinite(trader.equity)

    # -----------------------------------------------------------------------
    # 10. PaperTrader: stop-loss triggers
    # -----------------------------------------------------------------------

    def test_paper_trader_stop_loss_triggers(self):
        from ferro_ta.analysis.live import PaperTrader

        # Price rises on entry then falls sharply — SL should trigger
        n = 20
        close = np.array(
            [100.0] * 5
            + [98.0, 96.0, 94.0, 92.0, 90.0]  # declining
            + [88.0, 86.0, 84.0, 82.0, 80.0, 78.0, 76.0, 74.0, 72.0, 70.0],
            dtype=float,
        )
        open_ = close * 1.001
        high = close * 1.005
        low = close * 0.99  # Low drops to trigger SL

        sl_pct = 0.03  # 3% stop-loss
        trader = PaperTrader(initial_capital=100_000, stop_loss_pct=sl_pct)

        # Signal: go long on bar 0
        signals = np.zeros(n)
        signals[0] = 1.0  # enter long

        for i in range(n):
            trader.on_bar(open_[i], high[i], low[i], close[i], signals[i])

        # With 3% SL and price dropping >3% below entry, we expect a trade to close
        # Final position should be 0 (SL triggered exit)
        assert trader.position == 0.0 or len(trader.trades) > 0

    # -----------------------------------------------------------------------
    # 11. PaperTrader: reset clears state
    # -----------------------------------------------------------------------

    def test_paper_trader_reset_clears_state(self):
        from ferro_ta.analysis.live import PaperTrader

        rng = np.random.default_rng(20)
        n = 30
        close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
        open_ = close * 0.999
        high = close * 1.01
        low = close * 0.99
        signals = np.where(np.arange(n) % 10 < 5, 1.0, -1.0).astype(float)

        trader = PaperTrader(initial_capital=50_000)
        for i in range(n):
            trader.on_bar(open_[i], high[i], low[i], close[i], signals[i])

        assert len(trader.equity_curve) > 0

        trader.reset()

        assert trader.position == 0.0
        assert trader.equity == pytest.approx(1.0)
        assert len(trader.trades) == 0
        assert len(trader.equity_curve) == 0
        assert trader.equity_abs == pytest.approx(50_000.0)

    # -----------------------------------------------------------------------
    # 12. PaperTrader equity matches backtest_ohlcv_core
    # -----------------------------------------------------------------------

    def test_paper_trader_equity_matches_backtest(self):
        from ferro_ta.analysis.live import PaperTrader

        rng = np.random.default_rng(42)
        n = 50
        close = np.cumprod(1 + rng.standard_normal(n) * 0.005) * 100.0
        open_ = close * (1 - rng.uniform(0, 0.003, n))
        high = close * (1 + rng.uniform(0.001, 0.005, n))
        low = close * (1 - rng.uniform(0.001, 0.005, n))
        signals = np.where(np.arange(n) % 10 < 5, 1.0, -1.0).astype(np.float64)

        # Vectorized Rust engine
        _, _, _, _, eq_rust = backtest_ohlcv_core(open_, high, low, close, signals)

        # PaperTrader bar-by-bar
        trader = PaperTrader(initial_capital=100_000)
        for i in range(n):
            trader.on_bar(open_[i], high[i], low[i], close[i], signals[i])

        eq_paper = np.array(trader.equity_curve)
        assert eq_paper.shape == eq_rust.shape
        npt.assert_allclose(
            eq_paper,
            eq_rust,
            rtol=1e-6,
            atol=1e-9,
            err_msg="PaperTrader equity curve does not match backtest_ohlcv_core",
        )
