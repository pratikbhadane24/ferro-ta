"""Tests for ferro_ta.analysis.sweep (parallel parameter sweeps)."""

from __future__ import annotations

import numpy as np
import pytest
from ferro_ta._ferro_ta import compute_performance_metrics

from ferro_ta.analysis.backtest import backtest, sma_crossover_strategy
from ferro_ta.analysis.overfit import PBOResult
from ferro_ta.analysis.sweep import SweepResult, expand_grid, sweep
from ferro_ta.core.exceptions import FerroTAInputError, FerroTAValueError

GRID = {"fast": [5, 10, 20], "slow": [40, 80]}


@pytest.fixture(scope="module")
def close() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.cumprod(1 + rng.normal(0.0002, 0.01, 1500)) * 100


@pytest.fixture(scope="module")
def result(close) -> SweepResult:
    return sweep(close, "sma_crossover", GRID, commission_per_trade=1.0, slippage_bps=2)


class TestExpandGrid:
    def test_cartesian_product_in_key_order(self):
        assert expand_grid({"a": [1, 2], "b": ["x"]}) == (
            {"a": 1, "b": "x"},
            {"a": 2, "b": "x"},
        )

    def test_list_of_dicts_passthrough(self):
        assert expand_grid([{"a": 1}, {"a": 3}]) == ({"a": 1}, {"a": 3})

    @pytest.mark.parametrize("grid", [{}, [], {"a": []}])
    def test_rejects_empty(self, grid):
        with pytest.raises(FerroTAValueError):
            expand_grid(grid)


class TestSweep:
    def test_shapes(self, result, close):
        assert result.n_trials == 6
        assert result.returns.shape == (len(close), 6)
        assert all(v.shape == (6,) for v in result.metrics.values())
        assert {"sharpe", "total_return", "max_drawdown", "n_trades"} <= set(
            result.metrics
        )

    def test_each_column_matches_single_backtest(self, result, close):
        for j, params in enumerate(result.params):
            single = backtest(
                close,
                strategy="sma_crossover",
                commission_per_trade=1.0,
                slippage_bps=2,
                **params,
            )
            np.testing.assert_array_equal(result.returns[:, j], single.strategy_returns)
            expected = dict(
                compute_performance_metrics(single.strategy_returns, single.equity)
            )
            assert result.metrics["total_return"][j] == pytest.approx(
                expected["total_return"]
            )
            assert result.metrics["sharpe"][j] == pytest.approx(expected["sharpe"])

    def test_parallel_equals_serial(self, close):
        a = sweep(close, sma_crossover_strategy, GRID, parallel=True)
        b = sweep(close, sma_crossover_strategy, GRID, parallel=False)
        np.testing.assert_array_equal(a.returns, b.returns)
        for k in a.metrics:
            np.testing.assert_array_equal(a.metrics[k], b.metrics[k])

    def test_custom_callable_and_nan_warmup(self, close):
        def momentum(c, lookback):
            sig = np.full(len(c), np.nan)
            sig[lookback:] = np.sign(c[lookback:] - c[:-lookback])
            return sig

        res = sweep(close, momentum, {"lookback": [5, 20, 60]})
        assert np.all(np.isfinite(res.returns))
        assert res.returns[:5, 0].tolist() == [0.0] * 5

    def test_best_params_and_minimize(self, result):
        sharpe = result.metrics["sharpe"]
        assert result.best_params() == result.params[int(np.nanargmax(sharpe))]
        assert result.best_index("n_trades", minimize=True) == int(
            np.argmin(result.metrics["n_trades"])
        )
        with pytest.raises(FerroTAValueError):
            result.best_index("nope")

    def test_overfitting_hooks(self, result):
        dsr = result.deflated_sharpe()
        assert 0.0 <= dsr <= 1.0
        assert isinstance(result.pbo(n_splits=6), PBOResult)

    def test_pivot(self, result):
        rows, cols, mat = result.pivot("sharpe", "fast", "slow")
        assert rows == [5, 10, 20] and cols == [40, 80]
        assert mat[1, 1] == result.metrics["sharpe"][3]

    def test_pivot_takes_max_over_other_params(self, close):
        res = sweep(close, "sma_crossover", {"fast": [5], "slow": [40, 80]})
        _, _, mat = res.pivot("sharpe", "fast", "fast")
        assert mat[0, 0] == np.nanmax(res.metrics["sharpe"])

    def test_result_is_immutable(self, result):
        with pytest.raises(ValueError):
            result.returns[0, 0] = 1.0
        with pytest.raises(ValueError):
            result.metrics["sharpe"][0] = 99.0
        with pytest.raises(TypeError):
            result.metrics["sharpe"] = None  # type: ignore[index]
        with pytest.raises(TypeError):
            result.params[0]["fast"] = 1  # type: ignore[index]
        best = result.best_params()
        best["fast"] = -1
        assert result.best_params() != best

    def test_to_frame(self, result):
        pytest.importorskip("pandas")
        frame = result.to_frame()
        assert list(frame.columns[:2]) == ["fast", "slow"]
        assert len(frame) == 6

    def test_strategy_errors_are_wrapped_with_params(self, close):
        def broken(c, x):
            raise RuntimeError("boom")

        with pytest.raises(FerroTAValueError, match=r"\{'x': 1\}.*boom"):
            sweep(close, broken, {"x": [1]})

    def test_rejects_wrong_signal_length(self, close):
        with pytest.raises(FerroTAInputError):
            sweep(close, lambda c: np.ones(3), [{}])

    def test_rejects_bad_close(self):
        with pytest.raises(FerroTAInputError):
            sweep([1.0], "sma_crossover", GRID)

    def test_unknown_strategy_name(self, close):
        with pytest.raises(FerroTAValueError):
            sweep(close, "nope", GRID)


def _synthetic_result(values: dict[tuple[int, int], float]) -> SweepResult:
    combos = expand_grid({"a": list(range(5)), "b": list(range(5))})
    sharpe = np.array([values.get((p["a"], p["b"]), 0.0) for p in combos])
    return SweepResult(
        params=combos,
        returns=np.zeros((10, len(combos))),
        metrics={"sharpe": sharpe},
    )


class TestPlateauSelection:
    # A lone spike at (0, 4) in a hostile neighbourhood vs a broad plateau
    # centred on (2, 2).
    SURFACE = {
        (0, 4): 10.0,
        (0, 3): -5.0,
        (1, 3): -5.0,
        (1, 4): -5.0,
        **{(a, b): 3.0 for a in (1, 2, 3) for b in (1, 2, 3)},
    }

    def test_radius_zero_is_raw_argmax(self):
        res = _synthetic_result(self.SURFACE)
        assert res.best_params() == {"a": 0, "b": 4}
        assert res.best_params(radius=0) == {"a": 0, "b": 4}

    def test_radius_one_prefers_plateau_over_spike(self):
        res = _synthetic_result(self.SURFACE)
        assert res.best_params(radius=1) == {"a": 2, "b": 2}

    def test_minimize_with_radius(self):
        flipped = {k: -v for k, v in self.SURFACE.items()}
        res = _synthetic_result(flipped)
        assert res.best_params(radius=1, minimize=True) == {"a": 2, "b": 2}

    def test_nan_cells_are_ignored_in_neighbourhood(self):
        surface = dict(self.SURFACE)
        surface[(2, 1)] = np.nan
        res = _synthetic_result(surface)
        assert res.best_params(radius=1) == {"a": 2, "b": 2}

    def test_works_on_real_sweep_and_non_numeric_params(self, close):
        res = sweep(
            close,
            lambda c, fast, mode: (
                sma_crossover_strategy(c, fast=fast, slow=60)
                if mode == "long_short"
                else np.maximum(sma_crossover_strategy(c, fast=fast, slow=60), 0)
            ),
            {"fast": [5, 10, 20], "mode": ["long_short", "long_only"]},
        )
        assert 0 <= res.best_index(radius=1) < res.n_trials

    def test_rejects_negative_radius_and_ragged_params(self):
        res = _synthetic_result(self.SURFACE)
        with pytest.raises(FerroTAValueError):
            res.best_index(radius=-1)
        ragged = SweepResult(
            params=({"a": 1}, {"b": 2}),
            returns=np.zeros((5, 2)),
            metrics={"sharpe": np.array([1.0, 2.0])},
        )
        with pytest.raises(FerroTAValueError):
            ragged.best_index(radius=1)
