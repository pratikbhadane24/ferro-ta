"""
Parallel parameter sweeps
=========================

Grid-search a strategy over every parameter combination in one call. Signal
generation runs once per combination in Python (it calls the Rust
indicators); every backtest and metric then runs in Rust, in parallel, with
the GIL released.

The result keeps the full returns matrix, so the same sweep feeds straight
into the overfitting diagnostics in :mod:`ferro_ta.analysis.overfit`:

>>> import numpy as np
>>> from ferro_ta.analysis.sweep import sweep
>>> close = np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 2000)) * 100
>>> res = sweep(close, "sma_crossover", {"fast": [5, 10, 20], "slow": [50, 100, 200]})
>>> res.best_params("sharpe")              # doctest: +SKIP
{'fast': 10, 'slow': 100}
>>> res.deflated_sharpe()                  # selection-bias corrected  # doctest: +SKIP
>>> res.pbo(n_splits=10).pbo               # probability of overfitting  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import (
    backtest_sweep_builtin_core as _rust_backtest_sweep_builtin_core,
)
from ferro_ta._ferro_ta import backtest_sweep_core as _rust_backtest_sweep_core
from ferro_ta._ferro_ta import sweep_builtin_strategies as _rust_builtin_names
from ferro_ta._ferro_ta import sweep_plateau_scores as _rust_plateau_scores
from ferro_ta.analysis.backtest import _resolve_strategy
from ferro_ta.analysis.overfit import (
    PBOResult,
    _call_rust,
    _frozen,
    deflated_sharpe_ratio,
    pbo,
    per_period_sharpe,
)
from ferro_ta.core.exceptions import FerroTAInputError, FerroTAValueError

__all__ = ["ParamGrid", "SweepResult", "expand_grid", "sweep"]

#: Strategy names whose whole sweep (signals included) runs in Rust.
RUST_BUILTIN_STRATEGIES: frozenset[str] = frozenset(_rust_builtin_names())

ParamGrid = Union[Mapping[str, Sequence[Any]], Sequence[Mapping[str, Any]]]


def expand_grid(param_grid: ParamGrid) -> tuple[dict[str, Any], ...]:
    """Normalise a grid to a tuple of parameter dicts.

    A mapping of ``name -> values`` expands to its Cartesian product (keys in
    insertion order); a sequence of dicts is used as-is.
    """
    if isinstance(param_grid, Mapping):
        if not param_grid:
            raise FerroTAValueError("param_grid must not be empty")
        names = list(param_grid)
        values = [list(param_grid[k]) for k in names]
        empty = [k for k, v in zip(names, values) if not v]
        if empty:
            raise FerroTAValueError(f"param_grid has no values for: {empty}")
        return tuple(dict(zip(names, combo)) for combo in itertools.product(*values))
    combos = tuple(dict(p) for p in param_grid)
    if not combos:
        raise FerroTAValueError("param_grid must not be empty")
    return combos


@dataclasses.dataclass(frozen=True)
class SweepResult:
    """Output of :func:`sweep`.

    Parameters
    ----------
    params : tuple[Mapping]
        Parameters of each combination (column order of ``returns``).
    returns : NDArray[np.float64], shape (n_bars, n_combos)
        Per-bar strategy returns of every combination.
    metrics : dict[str, NDArray[np.float64]]
        Each performance metric as an array of length ``n_combos``.

    Immutable: arrays are read-only and mappings are read-only proxies.
    ``best_params()`` returns a fresh ``dict``.
    """

    params: tuple[Mapping[str, Any], ...]
    returns: NDArray[np.float64]
    metrics: Mapping[str, NDArray[np.float64]]

    @property
    def n_trials(self) -> int:
        return len(self.params)

    def _metric(self, metric: str) -> NDArray[np.float64]:
        if metric not in self.metrics:
            raise FerroTAValueError(
                f"Unknown metric '{metric}'. Available: {sorted(self.metrics)}"
            )
        return self.metrics[metric]

    def _grid_coordinates(self) -> NDArray[np.int64]:
        """Each combo's position on the grid: index of its value per param.

        Values are ordered as they first appear in ``params`` (the order the
        grid was written in), so any hashable parameter type works.
        """
        names = list(self.params[0])
        if any(list(p) != names for p in self.params):
            raise FerroTAValueError(
                "plateau selection needs every combination to have the same params"
            )
        axes = [list(dict.fromkeys(p[n] for p in self.params)) for n in names]
        return np.array(
            [[axis.index(p[n]) for n, axis in zip(names, axes)] for p in self.params],
            dtype=np.int64,
        ).reshape(len(self.params), len(names))

    def _plateau_scores(self, values: NDArray[np.float64], radius: int) -> NDArray:
        """Mean of ``values`` over each combo's grid neighbourhood (NaN-aware).

        Neighbours are combos within ``radius`` steps on *every* parameter
        axis (Chebyshev distance), including the combo itself.
        """
        coords = np.ascontiguousarray(self._grid_coordinates())
        return np.asarray(
            _call_rust(
                _rust_plateau_scores,
                np.ascontiguousarray(values, dtype=np.float64),
                coords,
                int(radius),
            )
        )

    def best_index(
        self, metric: str = "sharpe", *, minimize: bool = False, radius: int = 0
    ) -> int:
        """Column index of the best combination by ``metric`` (NaN-aware).

        ``radius=0`` takes the single best cell. ``radius >= 1`` scores each
        cell by the mean of its grid neighbourhood first, so a broad stable
        region beats an isolated spike, which is usually noise.
        """
        if radius < 0:
            raise FerroTAValueError(f"radius must be >= 0, got {radius}")
        values = self._metric(metric)
        if radius > 0:
            values = self._plateau_scores(values, radius)
        if np.all(np.isnan(values)):
            raise FerroTAValueError(f"metric '{metric}' is NaN for every combination")
        return int(np.nanargmin(values) if minimize else np.nanargmax(values))

    def best_params(
        self, metric: str = "sharpe", *, minimize: bool = False, radius: int = 0
    ) -> dict[str, Any]:
        """Parameter dict of the best combination (see :meth:`best_index`)."""
        return dict(
            self.params[self.best_index(metric, minimize=minimize, radius=radius)]
        )

    def deflated_sharpe(
        self, metric: str = "sharpe", *, minimize: bool = False, radius: int = 0
    ) -> float:
        """Deflated Sharpe Ratio of the combination selected by ``metric``.

        Deflates against all ``n_trials`` combinations, so the more you
        searched the higher the bar. ``metric`` only picks the winner; the
        deflation itself always uses raw per-period Sharpe (no risk-free
        adjustment), as the DSR formula requires.
        """
        best = self.best_index(metric, minimize=minimize, radius=radius)
        return deflated_sharpe_ratio(
            self.returns[:, best], trial_sharpes=per_period_sharpe(self.returns)
        )

    def pbo(self, n_splits: int = 16) -> PBOResult:
        """Probability of Backtest Overfitting across all combinations."""
        return pbo(self.returns, n_splits=n_splits)

    def pivot(
        self, metric: str, row: str, col: str
    ) -> tuple[list[Any], list[Any], NDArray[np.float64]]:
        """Heatmap-ready ``(row_values, col_values, matrix)`` for two params.

        When other parameters also vary, each cell holds the best (max)
        value over them; cells with no combination are NaN.
        """
        values = self._metric(metric)
        for name in (row, col):
            if any(name not in p for p in self.params):
                raise FerroTAValueError(
                    f"parameter '{name}' is not in every combination"
                )
        rows = list(dict.fromkeys(p[row] for p in self.params))
        cols = list(dict.fromkeys(p[col] for p in self.params))
        matrix = np.full((len(rows), len(cols)), np.nan)
        for p, v in zip(self.params, values):
            i, j = rows.index(p[row]), cols.index(p[col])
            matrix[i, j] = v if np.isnan(matrix[i, j]) else max(matrix[i, j], v)
        return rows, cols, matrix

    def to_frame(self) -> Any:
        """``pandas.DataFrame`` with one row per combination (params + metrics)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("SweepResult.to_frame() requires pandas") from exc
        frame = pd.DataFrame([dict(p) for p in self.params])
        for name, column in self.metrics.items():
            frame[name] = column
        return frame


def _build_signals(
    close: NDArray[np.float64],
    strategy_fn: Callable[..., ArrayLike],
    combos: tuple[dict[str, Any], ...],
) -> NDArray[np.float64]:
    signals = np.empty((len(close), len(combos)), dtype=np.float64)
    for j, params in enumerate(combos):
        try:
            column = np.asarray(strategy_fn(close, **params), dtype=np.float64)
        except Exception as exc:
            raise FerroTAValueError(
                f"strategy failed for params {params}: {exc}"
            ) from exc
        if column.shape != close.shape:
            raise FerroTAInputError(
                f"strategy returned shape {column.shape} for params {params}; "
                f"expected {close.shape}"
            )
        signals[:, j] = column
    return signals


def sweep(
    close: ArrayLike,
    strategy: Union[str, Callable[..., ArrayLike]],
    param_grid: ParamGrid,
    *,
    commission_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    initial_capital: float = 100_000.0,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
    parallel: bool = True,
) -> SweepResult:
    """Backtest ``strategy`` for every combination in ``param_grid``.

    Parameters
    ----------
    close : array-like
        Close prices.
    strategy : str or callable
        Strategy name from :func:`ferro_ta.analysis.backtest.backtest`'s
        registry, or ``fn(close, **params) -> signals``. Names in
        :data:`RUST_BUILTIN_STRATEGIES` (``sma_crossover``, ``rsi_30_70``,
        ``macd_crossover``) run entirely in Rust, signal generation included, and
        silently **skip** combinations that violate their constraints
        (e.g. ``fast >= slow``), so ``SweepResult.params`` lists only the
        combinations that ran. Callables run once per combination in Python
        and any exception they raise aborts the sweep.
    param_grid : mapping or sequence of mappings
        ``{"fast": [5, 10], "slow": [50, 100]}`` (Cartesian product) or an
        explicit list of parameter dicts.
    commission_per_trade, slippage_bps, initial_capital : float
        Costs, with the same semantics as :func:`backtest`.
    periods_per_year, risk_free_rate : float
        Annualisation settings for the metrics.
    parallel : bool
        Run combinations across threads (default True).
    """
    c = np.ascontiguousarray(close, dtype=np.float64)
    if c.ndim != 1 or len(c) < 2:
        raise FerroTAInputError("close must be 1-D with at least 2 bars")
    combos = expand_grid(param_grid)
    costs = (
        float(commission_per_trade),
        float(slippage_bps),
        float(initial_capital),
        float(periods_per_year),
        float(risk_free_rate),
        bool(parallel),
    )
    if isinstance(strategy, str) and strategy in RUST_BUILTIN_STRATEGIES:
        kept, returns_2d, raw_metrics = _call_rust(
            _rust_backtest_sweep_builtin_core,
            c,
            strategy,
            *_numeric_combos(combos),
            *costs,
        )
        combos = tuple(combos[int(k)] for k in np.asarray(kept))
    else:
        signals = _build_signals(c, _resolve_strategy(strategy), combos)
        returns_2d, raw_metrics = _call_rust(
            _rust_backtest_sweep_core, c, signals, *costs
        )
    return SweepResult(
        params=tuple(MappingProxyType(p) for p in combos),
        returns=_frozen(returns_2d),
        metrics=MappingProxyType({k: _frozen(v) for k, v in raw_metrics.items()}),
    )


def _numeric_combos(
    combos: tuple[dict[str, Any], ...],
) -> tuple[list[str], NDArray[np.float64]]:
    """``(names, matrix)`` for a built-in sweep; every combo needs the same keys."""
    names = list(combos[0])
    if any(list(p) != names for p in combos):
        raise FerroTAValueError(
            "built-in sweeps need every combination to set the same parameters"
        )
    try:
        matrix = np.array([[p[n] for n in names] for p in combos], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FerroTAValueError(
            f"built-in strategy parameters must be numeric: {exc}"
        ) from exc
    return names, matrix.reshape(len(combos), len(names))
