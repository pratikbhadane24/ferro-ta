"""
Minimal Backtesting Harness
============================

A lightweight vectorized backtester that uses ferro_ta indicators as the engine.

**Scope** (minimal harness):
- Vectorized approach: compute indicators once, then apply a signal function over bars.
- Single-asset, long-only or long/short, no leverage.
- Optional **commission** (per trade) and **slippage** (basis points) for more realistic equity.
- Returns a :class:`BacktestResult` with signals, positions, and equity curve.

For production backtesting consider `backtrader`, `zipline`, or `vectorbt`.

Quick start
-----------
>>> import numpy as np
>>> from ferro_ta.analysis.backtest import backtest, rsi_strategy
>>>
>>> # Generate synthetic OHLCV data
>>> np.random.seed(42)
>>> n = 100
>>> close = np.cumprod(1 + np.random.randn(n) * 0.01) * 100
>>> volume = np.random.randint(1_000, 10_000, n).astype(float)
>>>
>>> result = backtest(close, volume=volume, strategy="rsi_30_70")
>>> print(result)  # BacktestResult(bars=100, trades=…, final_equity=…)

API
---
backtest(close, *, high=None, low=None, open=None, volume=None,
         strategy="rsi_30_70", commission_per_trade=0, slippage_bps=0, **kwargs)
    Run the backtester and return a :class:`BacktestResult`. Optional
    commission (subtracted from equity on each position change) and slippage
    (basis points; applied as a cost on the bar where position changes).

rsi_strategy(close, timeperiod=14, oversold=30, overbought=70)
    Built-in RSI oversold/overbought strategy; returns a signal array.

sma_crossover_strategy(close, fast=10, slow=30)
    Built-in SMA crossover strategy; returns a signal array.

macd_crossover_strategy(close, fastperiod=12, slowperiod=26, signalperiod=9)
    Built-in MACD line/signal crossover strategy; returns a signal array.

BacktestResult
    Dataclass-like container with signals, positions, returns, equity arrays.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import CommissionModel
from ferro_ta._ferro_ta import Currency as _RustCurrency
from ferro_ta._ferro_ta import backtest_core as _rust_backtest_core
from ferro_ta._ferro_ta import (
    backtest_multi_asset_core as _rust_backtest_multi_asset_core,
)
from ferro_ta._ferro_ta import backtest_ohlcv_core as _rust_backtest_ohlcv_core
from ferro_ta._ferro_ta import compute_performance_metrics as _rust_compute_perf_metrics
from ferro_ta._ferro_ta import drawdown_series as _rust_drawdown_series
from ferro_ta._ferro_ta import extract_trades_ohlcv as _rust_extract_trades
from ferro_ta._ferro_ta import kelly_fraction as _rust_kelly_fraction
from ferro_ta._ferro_ta import macd_crossover_signals as _rust_macd_crossover_signals
from ferro_ta._ferro_ta import monte_carlo_bootstrap as _rust_monte_carlo_bootstrap
from ferro_ta._ferro_ta import rsi_threshold_signals as _rust_rsi_threshold_signals
from ferro_ta._ferro_ta import sma_crossover_signals as _rust_sma_crossover_signals
from ferro_ta._ferro_ta import walk_forward_indices as _rust_walk_forward_indices
from ferro_ta.core.exceptions import FerroTAInputError, FerroTAValueError

# ---------------------------------------------------------------------------
# Currency system (backed by Rust via ferro_ta._ferro_ta.Currency)
# ---------------------------------------------------------------------------

# Re-export the Rust-backed Currency class as the public API.

Currency = _RustCurrency

# Built-in currency constants
INR: _RustCurrency = Currency.INR()
USD: _RustCurrency = Currency.USD()
EUR: _RustCurrency = Currency.EUR()
GBP: _RustCurrency = Currency.GBP()
JPY: _RustCurrency = Currency.JPY()
USDT: _RustCurrency = Currency.USDT()

_CURRENCIES: dict[str, _RustCurrency] = {
    "INR": INR,
    "USD": USD,
    "EUR": EUR,
    "GBP": GBP,
    "JPY": JPY,
    "USDT": USDT,
}


def format_currency(amount: float, currency: _RustCurrency | None = None) -> str:
    """Format *amount* using *currency*'s display style.

    Uses Indian lakh/crore grouping for INR, standard grouping for others.

    >>> format_currency(123456.78)
    '₹1,23,456.78'
    >>> format_currency(1234567.89, USD)
    '$1,234,567.89'
    """
    if currency is None:
        currency = INR
    return currency.format(amount)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


class BacktestResult:
    """Container for backtesting output.

    Attributes
    ----------
    signals : NDArray[np.float64]
        Array of +1 (long), -1 (short), or 0 (flat) for every bar.
    positions : NDArray[np.float64]
        Lagged signals — position held *during* each bar (shift by 1 to
        avoid look-ahead bias).
    bar_returns : NDArray[np.float64]
        Per-bar return of the underlying close price (pct change).
    strategy_returns : NDArray[np.float64]
        ``positions * bar_returns`` — strategy return at each bar.
    equity : NDArray[np.float64]
        Cumulative equity curve starting at 1.0.
    n_trades : int
        Number of position changes.
    final_equity : float
        Terminal equity value.
    """

    __slots__ = (
        "signals",
        "positions",
        "bar_returns",
        "strategy_returns",
        "equity",
        "n_trades",
        "final_equity",
    )

    def __init__(
        self,
        signals: NDArray[np.float64],
        positions: NDArray[np.float64],
        bar_returns: NDArray[np.float64],
        strategy_returns: NDArray[np.float64],
        equity: NDArray[np.float64],
    ) -> None:
        self.signals = signals
        self.positions = positions
        self.bar_returns = bar_returns
        self.strategy_returns = strategy_returns
        self.equity = equity
        self.n_trades = int(np.sum(np.diff(positions) != 0))
        self.final_equity = float(equity[-1]) if len(equity) > 0 else 1.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BacktestResult("
            f"bars={len(self.signals)}, "
            f"trades={self.n_trades}, "
            f"final_equity={self.final_equity:.4f})"
        )


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


def rsi_strategy(
    close: ArrayLike,
    timeperiod: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> NDArray[np.float64]:
    """RSI oversold / overbought signal generator.

    Returns
    -------
    signals : ndarray of float64
        +1 where RSI <= oversold (buy signal), -1 where RSI >= overbought
        (sell signal), 0 otherwise.  NaN during the RSI warm-up period.

    Parameters
    ----------
    close : array-like
        Close prices.
    timeperiod : int
        RSI look-back period (default 14).
    oversold : float
        RSI level below which a long (+1) signal is generated (default 30).
    overbought : float
        RSI level above which a short (-1) signal is generated (default 70).
    """
    if timeperiod < 1:
        raise FerroTAValueError(f"timeperiod must be >= 1, got {timeperiod}")

    c = np.asarray(close, dtype=np.float64)
    return np.asarray(
        _rust_rsi_threshold_signals(
            c, int(timeperiod), float(oversold), float(overbought)
        ),
        dtype=np.float64,
    )


def sma_crossover_strategy(
    close: ArrayLike,
    fast: int = 10,
    slow: int = 30,
) -> NDArray[np.float64]:
    """SMA fast/slow crossover strategy.

    Returns
    -------
    signals : ndarray of float64
        +1 when fast SMA > slow SMA (uptrend), -1 when fast SMA < slow SMA
        (downtrend), NaN during the warm-up window.

    Parameters
    ----------
    close : array-like
        Close prices.
    fast : int
        Fast SMA period (default 10).
    slow : int
        Slow SMA period (default 30).
    """
    if fast < 1:
        raise FerroTAValueError(f"fast must be >= 1, got {fast}")
    if slow < 1:
        raise FerroTAValueError(f"slow must be >= 1, got {slow}")
    if fast >= slow:
        raise FerroTAValueError(f"fast ({fast}) must be less than slow ({slow})")

    c = np.asarray(close, dtype=np.float64)
    return np.asarray(
        _rust_sma_crossover_signals(c, int(fast), int(slow)),
        dtype=np.float64,
    )


def macd_crossover_strategy(
    close: ArrayLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> NDArray[np.float64]:
    """MACD line / signal line crossover strategy.

    Returns
    -------
    signals : ndarray of float64
        +1 when MACD line > signal line (uptrend), -1 when MACD line < signal line
        (downtrend), NaN during the MACD warm-up window.

    Parameters
    ----------
    close : array-like
        Close prices.
    fastperiod : int
        Fast EMA period (default 12).
    slowperiod : int
        Slow EMA period (default 26).
    signalperiod : int
        Signal line EMA period (default 9).
    """
    if fastperiod < 1 or slowperiod < 1 or signalperiod < 1:
        raise FerroTAValueError("MACD periods must be >= 1")
    if fastperiod >= slowperiod:
        raise FerroTAValueError(
            f"fastperiod ({fastperiod}) must be less than slowperiod ({slowperiod})"
        )

    c = np.asarray(close, dtype=np.float64)
    return np.asarray(
        _rust_macd_crossover_signals(
            c, int(fastperiod), int(slowperiod), int(signalperiod)
        ),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Built-in strategy registry
# ---------------------------------------------------------------------------

_BENCHMARK_METRICS = frozenset(
    (
        "alpha",
        "beta",
        "tracking_error",
        "information_ratio",
        "benchmark_cagr",
        "benchmark_sharpe",
    )
)

_BUILTIN_STRATEGIES: dict[str, Callable[..., NDArray[np.float64]]] = {
    "rsi_30_70": rsi_strategy,
    "sma_crossover": sma_crossover_strategy,
    "macd_crossover": macd_crossover_strategy,
}


# ---------------------------------------------------------------------------
# Main backtest entry point
# ---------------------------------------------------------------------------


def backtest(
    close: ArrayLike,
    *,
    high: Optional[ArrayLike] = None,
    low: Optional[ArrayLike] = None,
    open: Optional[ArrayLike] = None,
    volume: Optional[ArrayLike] = None,
    strategy: Union[str, Callable[..., NDArray[np.float64]]] = "rsi_30_70",
    commission_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    **strategy_kwargs: object,
) -> BacktestResult:
    """Run a vectorized backtest on *close* prices using *strategy*.

    Parameters
    ----------
    close : array-like
        Close prices (required).
    high, low, open, volume : array-like, optional
        Additional OHLCV data.  Passed to the strategy function if it accepts
        them (via ``**strategy_kwargs``); currently unused by the built-in
        strategies.
    strategy : str or callable
        Either a name of a built-in strategy (``"rsi_30_70"``,
        ``"sma_crossover"``, or ``"macd_crossover"``) or a callable with
        signature ``(close, **kwargs) -> ndarray`` that returns a signal array.
    commission_per_trade : float, optional
        Fixed commission deducted from equity on each position change (default 0).
    slippage_bps : float, optional
        Slippage in basis points (1 bps = 0.01%) applied as a cost on the bar
        where the position changes (default 0).
    **strategy_kwargs
        Extra keyword arguments forwarded to the strategy function
        (e.g. ``timeperiod=14``, ``oversold=30``).

    Returns
    -------
    BacktestResult
        Container with signals, positions, equity curve, and trade count.

    Raises
    ------
    FerroTAValueError
        If a named strategy is unknown.
    FerroTAInputError
        If ``close`` is too short (< 2 bars) or contains non-finite values.

    Notes
    -----
    Commission is subtracted from equity immediately after each position change.
    Slippage is applied by reducing the strategy return on the bar where the
    position changes by ``slippage_bps / 10000`` (one-way).
    """
    c = np.asarray(close, dtype=np.float64)
    if c.ndim != 1:
        raise FerroTAInputError("close must be a 1-D array.")
    if len(c) < 2:
        raise FerroTAInputError(f"close must have at least 2 bars, got {len(c)}.")

    # ------------------------------------------------------------------
    # Resolve strategy & compute signals
    # ------------------------------------------------------------------
    strategy_fn = _resolve_strategy(strategy)
    signals = np.asarray(strategy_fn(c, **strategy_kwargs), dtype=np.float64)
    positions, bar_returns, strategy_returns, equity = _rust_backtest_core(
        c,
        signals,
        commission_per_trade=float(commission_per_trade),
        slippage_bps=float(slippage_bps),
    )

    return BacktestResult(
        signals=signals,
        positions=np.asarray(positions, dtype=np.float64),
        bar_returns=np.asarray(bar_returns, dtype=np.float64),
        strategy_returns=np.asarray(strategy_returns, dtype=np.float64),
        equity=np.asarray(equity, dtype=np.float64),
    )


# ===========================================================================
# Advanced API — AdvancedBacktestResult, BacktestEngine, walk_forward, monte_carlo
# ===========================================================================


class AdvancedBacktestResult(BacktestResult):
    """Extended backtest result with full metrics, trade log, and drawdown series.

    All ``BacktestResult`` attributes are preserved (``isinstance`` checks work).

    Additional Attributes
    ---------------------
    metrics : dict[str, float]
        Full performance metrics: cagr, sharpe, sortino, calmar, max_drawdown,
        avg_drawdown, max_drawdown_duration_bars, ulcer_index, omega_ratio,
        win_rate, profit_factor, r_expectancy, tail_ratio, skewness, kurtosis, etc.
    trades : Any
        Trade log as ``pd.DataFrame`` (if pandas is installed) with columns:
        entry_bar, exit_bar, direction, entry_price, exit_price, pnl_pct,
        duration_bars, mae, mfe. None if no trades were extracted.
    drawdown_series : NDArray[np.float64]
        Per-bar drawdown (always <= 0).
    fill_prices : NDArray[np.float64]
        Actual fill prices per bar (NaN when flat). NaN array in close-only mode.
    """

    __slots__ = BacktestResult.__slots__ + (
        "metrics",
        "trades",
        "drawdown_series",
        "fill_prices",
        "currency",
        "initial_capital",
        "equity_abs",
    )

    def __init__(
        self,
        signals: NDArray,
        positions: NDArray,
        bar_returns: NDArray,
        strategy_returns: NDArray,
        equity: NDArray,
        metrics: dict,
        trades: Any,
        drawdown_series: NDArray,
        fill_prices: NDArray,
        currency: _RustCurrency = INR,
        initial_capital: float = 100_000.0,
    ) -> None:
        super().__init__(signals, positions, bar_returns, strategy_returns, equity)
        self.metrics = metrics
        self.trades = trades
        self.drawdown_series = drawdown_series
        self.fill_prices = fill_prices
        self.currency = currency
        self.initial_capital = float(initial_capital)
        self.equity_abs = equity * self.initial_capital

    def __repr__(self) -> str:  # pragma: no cover
        m = self.metrics
        final = (
            float(self.equity_abs[-1])
            if len(self.equity_abs) > 0
            else self.initial_capital
        )
        return (
            f"AdvancedBacktestResult("
            f"bars={len(self.signals)}, "
            f"trades={self.n_trades}, "
            f"sharpe={m.get('sharpe', float('nan')):.3f}, "
            f"max_dd={m.get('max_drawdown', float('nan')):.1%}, "
            f"final={self.currency.format(final)})"
        )

    def to_equity_dataframe(self, freq: str = "B") -> Any:
        """Return equity and drawdown as a ``pd.DataFrame`` indexed by date.

        Parameters
        ----------
        freq : str
            pandas date-offset alias for the synthetic DatetimeIndex (default ``"B"``).

        Returns
        -------
        pd.DataFrame with columns ``equity``, ``equity_abs``, ``strategy_returns``,
        ``drawdown``, indexed by a synthetic ``pd.DatetimeIndex`` starting 2000-01-03.
        Raises ``ImportError`` if pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_equity_dataframe()") from exc
        n = len(self.equity)
        idx = pd.date_range("2000-01-03", periods=n, freq=freq)
        return pd.DataFrame(
            {
                "equity": self.equity,
                "equity_abs": self.equity_abs,
                "strategy_returns": self.strategy_returns,
                "drawdown": self.drawdown_series,
            },
            index=idx,
        )

    def summary(self) -> dict:
        """Return a concise performance summary dict.

        Includes the 9 most commonly cited metrics plus n_trades,
        initial_capital, final_capital, absolute_pnl, and currency.
        """
        m = self.metrics
        keys = (
            "total_return",
            "cagr",
            "annualized_vol",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        )
        result = {k: m.get(k, float("nan")) for k in keys}
        result["n_trades"] = self.n_trades
        result["initial_capital"] = self.initial_capital
        final_capital = (
            float(self.equity_abs[-1])
            if len(self.equity_abs) > 0
            else self.initial_capital
        )
        result["final_capital"] = final_capital
        result["absolute_pnl"] = final_capital - self.initial_capital
        result["currency"] = self.currency.code
        # Include benchmark metrics if available
        for key in _BENCHMARK_METRICS:
            if key in m:
                result[key] = m[key]
        return result


def _resolve_strategy(
    strategy: Union[str, Callable],
) -> Callable[..., NDArray]:
    if isinstance(strategy, str):
        if strategy not in _BUILTIN_STRATEGIES:
            raise FerroTAValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {sorted(_BUILTIN_STRATEGIES)}"
            )
        return _BUILTIN_STRATEGIES[strategy]
    elif callable(strategy):
        return strategy
    raise FerroTAValueError("strategy must be a string name or a callable.")


def _pct_change(arr: NDArray) -> NDArray:
    """Percentage change with zero-price guard. Returns array of length len(arr)-1."""
    return np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1.0)


def _kelly_stats(strategy_returns: NDArray) -> tuple[float, float, float]:
    """Extract (win_rate, avg_win, avg_loss) from strategy returns.

    Returns (0, 0, 0) when there are no active trades.
    """
    active = strategy_returns[np.isfinite(strategy_returns) & (strategy_returns != 0.0)]
    if len(active) == 0:
        return 0.0, 0.0, 0.0
    wins = active[active > 0.0]
    losses = active[active < 0.0]
    win_rate = len(wins) / len(active)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(np.abs(losses).mean()) if len(losses) > 0 else 0.0
    return win_rate, avg_win, avg_loss


def _build_trades_df(
    positions: NDArray,
    fill_prices: NDArray,
    high: NDArray,
    low: NDArray,
    initial_capital: float = 100_000.0,
) -> Any:
    """Extract trade log; returns pd.DataFrame if pandas available, else None."""
    try:
        import pandas as pd
    except ImportError:
        return None
    eb, xb, d, ep, xp, pnl, dur, mae, mfe = _rust_extract_trades(
        positions, fill_prices, high, low
    )
    if len(eb) == 0:
        return pd.DataFrame(
            columns=[
                "entry_bar",
                "exit_bar",
                "direction",
                "entry_price",
                "exit_price",
                "pnl_pct",
                "pnl_abs",
                "duration_bars",
                "mae",
                "mfe",
            ]
        )
    df = pd.DataFrame(
        {
            "entry_bar": eb,
            "exit_bar": xb,
            "direction": d,
            "entry_price": ep,
            "exit_price": xp,
            "pnl_pct": pnl,
            "duration_bars": dur,
            "mae": mae,
            "mfe": mfe,
        }
    )
    df["pnl_abs"] = df["pnl_pct"] * initial_capital
    return df


class BacktestEngine:
    """Composable backtesting engine with a fluent builder interface.

    Example
    -------
    >>> import numpy as np
    >>> from ferro_ta.analysis.backtest import BacktestEngine
    >>> close = np.cumprod(1 + np.random.randn(200) * 0.01) * 100
    >>> high = close * 1.01; low = close * 0.99; open_ = close * 0.999
    >>> result = (
    ...     BacktestEngine()
    ...     .with_commission(0.001)
    ...     .with_slippage(5.0)
    ...     .with_ohlcv(high=high, low=low, open_=open_)
    ...     .with_stop_loss(0.03)
    ...     .run(close, strategy="rsi_30_70")
    ... )
    >>> print(result.metrics["sharpe"])
    """

    def __init__(self) -> None:
        self._commission: float = 0.0
        self._commission_model: Optional[CommissionModel] = None
        self._currency: _RustCurrency = INR
        self._initial_capital: float = 100_000.0
        self._slippage_bps: float = 0.0
        self._slippage_pct_range: float = 0.0
        self._position_sizing: str = "fixed"
        self._fixed_fraction: float = 1.0
        self._vol_window: int = 20
        self._target_vol: float = 0.10
        self._high: Optional[NDArray] = None
        self._low: Optional[NDArray] = None
        self._open: Optional[NDArray] = None
        self._stop_loss_pct: float = 0.0
        self._take_profit_pct: float = 0.0
        self._trailing_stop_pct: float = 0.0
        self._fill_mode: str = "market_open"
        self._periods_per_year: float = 252.0
        self._risk_free_rate: float = 0.0
        self._benchmark_close: Optional[NDArray] = None
        self._limit_prices: Optional[NDArray] = None
        self._max_hold_bars: int = 0
        self._breakeven_pct: float = 0.0
        # Phase 2: Portfolio & Risk
        self._margin_ratio: float = 0.0
        self._margin_call_pct: float = 0.5
        self._daily_loss_limit: float = 0.0
        self._total_loss_limit: float = 0.0
        self._max_asset_weight: float = 1.0
        self._max_gross_exposure: float = 0.0
        self._max_net_exposure: float = 0.0

    def with_commission(self, rate: float) -> BacktestEngine:
        """Backward-compat: set a flat per-order fee (in base currency units)."""
        self._commission = float(rate)
        return self

    def with_commission_model(self, model: CommissionModel) -> BacktestEngine:
        """Set a full commission+tax model (takes precedence over ``with_commission``)."""
        self._commission_model = model
        return self

    def with_currency(
        self, currency: str | _RustCurrency | None = None
    ) -> BacktestEngine:
        """Set display currency (default: INR)."""
        if currency is None:
            currency = INR
        if isinstance(currency, str):
            try:
                currency = Currency.from_code(currency)
            except Exception:
                raise FerroTAValueError(
                    f"Unknown currency code '{currency}'. "
                    f"Supported: {sorted(_CURRENCIES)}"
                )
        self._currency = currency
        return self

    def with_initial_capital(self, capital: float) -> BacktestEngine:
        """Set starting capital in base currency (default: ₹1,00,000)."""
        self._initial_capital = float(capital)
        return self

    def with_benchmark(self, benchmark_close: ArrayLike) -> BacktestEngine:
        """Set benchmark close prices for alpha/beta/tracking error computation."""
        self._benchmark_close = np.asarray(benchmark_close, dtype=np.float64)
        return self

    def with_trailing_stop(self, pct: float) -> BacktestEngine:
        """Set trailing stop distance as a fraction (e.g. 0.02 = 2%). 0 = disabled."""
        self._trailing_stop_pct = float(pct)
        return self

    def with_slippage(self, bps: float) -> BacktestEngine:
        self._slippage_bps = float(bps)
        return self

    def with_ohlcv(
        self,
        *,
        high: ArrayLike,
        low: ArrayLike,
        open_: ArrayLike,
    ) -> BacktestEngine:
        self._high = np.asarray(high, dtype=np.float64)
        self._low = np.asarray(low, dtype=np.float64)
        self._open = np.asarray(open_, dtype=np.float64)
        return self

    def with_stop_loss(self, pct: float) -> BacktestEngine:
        self._stop_loss_pct = float(pct)
        return self

    def with_take_profit(self, pct: float) -> BacktestEngine:
        self._take_profit_pct = float(pct)
        return self

    def with_fill_mode(self, mode: str) -> BacktestEngine:
        if mode not in ("market_open", "market_close"):
            raise FerroTAValueError("fill_mode must be 'market_open' or 'market_close'")
        self._fill_mode = mode
        return self

    def with_position_sizing(
        self,
        method: str,
        fraction: float = 1.0,
        vol_window: int = 20,
        target_vol: float = 0.10,
    ) -> BacktestEngine:
        valid = (
            "fixed",
            "kelly",
            "half_kelly",
            "fixed_fractional",
            "volatility_target",
        )
        if method not in valid:
            raise FerroTAValueError(f"position_sizing must be one of {valid}")
        if method == "fixed_fractional" and not (0.0 < fraction <= 1.0):
            raise FerroTAValueError("fixed_fractional fraction must be in (0, 1]")
        self._vol_window = int(vol_window)
        self._target_vol = float(target_vol)
        self._position_sizing = method
        self._fixed_fraction = float(fraction)
        return self

    def with_calendar(self, periods_per_year: float) -> BacktestEngine:
        self._periods_per_year = float(periods_per_year)
        return self

    def with_risk_free_rate(self, rate: float) -> BacktestEngine:
        self._risk_free_rate = float(rate)
        return self

    def with_limit_orders(self, prices: ArrayLike) -> BacktestEngine:
        """Set limit prices for entry/exit orders (requires OHLCV data via with_ohlcv).

        Parameters
        ----------
        prices : array-like, shape (n_bars,)
            Limit price for each signal bar. NaN (or 0) entries use market-order fill.
            Buy limit: fill only when bar low <= limit_price (execute at limit_price).
            Sell limit: fill only when bar high >= limit_price (execute at limit_price).
        """
        self._limit_prices = np.asarray(prices, dtype=np.float64)
        return self

    def with_max_hold(self, n_bars: int) -> BacktestEngine:
        """Force exit after *n_bars* bars in trade regardless of signal (requires OHLCV).

        0 = disabled (default). Useful for mean-reversion strategies.
        """
        if int(n_bars) < 0:
            raise FerroTAValueError("max_hold n_bars must be >= 0")
        self._max_hold_bars = int(n_bars)
        return self

    def with_slippage_pct_range(self, pct: float) -> BacktestEngine:
        """Set slippage as a fraction of the bar's high-low range (requires OHLCV).

        Overrides ``with_slippage`` when both are set. Typical values: 0.05–0.20.
        Example: pct=0.10 means slippage = 10% of bar's (high - low).
        """
        self._slippage_pct_range = float(pct)
        return self

    def with_breakeven_stop(self, pct: float) -> BacktestEngine:
        """Move stop to entry price once profit reaches *pct* fraction (e.g. 0.02 = 2%). 0 = disabled."""
        self._breakeven_pct = float(pct)
        return self

    def with_leverage(
        self, margin_ratio: float, margin_call_pct: float = 0.5
    ) -> BacktestEngine:
        """Enable margin/leverage modeling. margin_ratio=0.2 means 20% margin (5x leverage).
        margin_call_pct=0.5 triggers a margin call when equity falls to 50% of initial margin."""
        self._margin_ratio = float(margin_ratio)
        self._margin_call_pct = float(margin_call_pct)
        return self

    def with_loss_limits(
        self, daily: float = 0.0, total: float = 0.0
    ) -> BacktestEngine:
        """Set circuit breakers. daily=0.02 halts after a 2% per-bar loss. total=0.20 halts after 20% drawdown."""
        self._daily_loss_limit = float(daily)
        self._total_loss_limit = float(total)
        return self

    def with_portfolio_constraints(
        self,
        max_asset_weight: float = 1.0,
        max_gross_exposure: float = 0.0,
        max_net_exposure: float = 0.0,
    ) -> BacktestEngine:
        """Set portfolio-level constraints for multi-asset backtests."""
        self._max_asset_weight = float(max_asset_weight)
        self._max_gross_exposure = float(max_gross_exposure)
        self._max_net_exposure = float(max_net_exposure)
        return self

    def run(
        self,
        close: ArrayLike,
        strategy: Union[str, Callable] = "rsi_30_70",
        **strategy_kwargs: object,
    ) -> AdvancedBacktestResult:
        """Run the backtest and return an AdvancedBacktestResult."""
        c = np.asarray(close, dtype=np.float64)
        if c.ndim != 1:
            raise FerroTAInputError("close must be a 1-D array.")
        if len(c) < 2:
            raise FerroTAInputError(f"close must have at least 2 bars, got {len(c)}.")

        strategy_fn = _resolve_strategy(strategy)
        signals = np.asarray(strategy_fn(c, **strategy_kwargs), dtype=np.float64)

        cm = self._commission_model
        commission_scalar = self._commission if cm is None else 0.0
        ic = self._initial_capital

        if self._position_sizing == "fixed_fractional":
            signals = signals * self._fixed_fraction

        if self._position_sizing == "volatility_target":
            proxy_rets = _pct_change(c)
            w = self._vol_window
            # Naive rolling window is O(n·w); cumsum-of-squares is O(n) with no per-bar allocation.
            # Safe for financial returns (centred near zero → no catastrophic cancellation).
            cs = np.cumsum(proxy_rets)
            cs2 = np.cumsum(proxy_rets**2)
            pad = np.zeros(1)
            s1 = cs[w - 1 :] - np.concatenate([pad, cs[: len(cs) - w]])
            s2 = cs2[w - 1 :] - np.concatenate([pad, cs2[: len(cs2) - w]])
            var = np.maximum(s2 / w - (s1 / w) ** 2, 0.0)
            rolling_vol = np.concatenate([np.full(w, np.nan), np.sqrt(var)]) * np.sqrt(
                self._periods_per_year
            )
            rolling_vol = np.concatenate([[np.nan], rolling_vol[: len(signals) - 1]])
            with np.errstate(divide="ignore", invalid="ignore"):
                # NaN positions (warm-up) have rolling_vol<=0 → else-branch produces 1.0
                scale = np.where(
                    rolling_vol > 0,
                    np.clip(self._target_vol / rolling_vol, 0.0, 3.0),
                    1.0,
                )
            signals = signals * scale

        use_ohlcv = (
            self._high is not None and self._low is not None and self._open is not None
        )

        def _execute_run(sigs: NDArray) -> tuple:
            if use_ohlcv:
                pos, fp, br, sr, eq = _rust_backtest_ohlcv_core(
                    self._open,
                    self._high,
                    self._low,
                    c,
                    sigs,
                    self._fill_mode,
                    self._stop_loss_pct,
                    self._take_profit_pct,
                    self._trailing_stop_pct,
                    cm,
                    self._slippage_bps,
                    ic,
                    commission_scalar,
                    self._limit_prices,
                    self._max_hold_bars,
                    self._slippage_pct_range,
                    self._breakeven_pct,
                    self._periods_per_year,
                    self._margin_ratio,
                    self._margin_call_pct,
                    self._daily_loss_limit,
                    self._total_loss_limit,
                )
                return (
                    np.asarray(pos),
                    np.asarray(fp),
                    np.asarray(br),
                    np.asarray(sr),
                    np.asarray(eq),
                )
            pos, br, sr, eq = _rust_backtest_core(
                c,
                sigs,
                cm,
                self._slippage_bps,
                ic,
                commission_scalar,
            )
            return (
                np.asarray(pos),
                np.full(len(c), np.nan, dtype=np.float64),
                np.asarray(br),
                np.asarray(sr),
                np.asarray(eq),
            )

        bench_returns_arr = None
        if self._benchmark_close is not None and len(self._benchmark_close) == len(c):
            bc = self._benchmark_close
            bench_returns_arr = np.concatenate([[0.0], _pct_change(bc)])

        def _compute_metrics(sr: NDArray, eq: NDArray) -> dict:
            return dict(
                _rust_compute_perf_metrics(
                    sr,
                    eq,
                    self._periods_per_year,
                    self._risk_free_rate,
                    bench_returns_arr,
                )
            )

        # Kelly / half-Kelly: estimate fraction from a preliminary run, then re-run scaled
        _kelly_kf: float = 0.0
        if self._position_sizing in ("kelly", "half_kelly"):
            positions, fill_prices, bar_returns, strategy_returns, equity = (
                _execute_run(signals)
            )
            wr, aw, al = _kelly_stats(strategy_returns)
            if aw > 0.0:
                try:
                    _kelly_kf = _rust_kelly_fraction(wr, aw, al)
                    fraction = (
                        _kelly_kf
                        if self._position_sizing == "kelly"
                        else _kelly_kf / 2.0
                    )
                    signals = signals * fraction
                except Exception as exc:
                    warnings.warn(
                        f"Kelly sizing failed, falling back to unit signals: {exc}",
                        stacklevel=2,
                    )

        positions, fill_prices, bar_returns, strategy_returns, equity = _execute_run(
            signals
        )
        metrics = _compute_metrics(strategy_returns, equity)

        # Annotate Kelly info (reuse pre-computed fraction, avoid re-scanning returns)
        if _kelly_kf > 0.0 and "kelly_fraction" not in metrics:
            metrics["kelly_fraction"] = _kelly_kf
            metrics["half_kelly_fraction"] = _kelly_kf / 2.0
            metrics["position_size_fraction"] = (
                _kelly_kf if self._position_sizing == "kelly" else _kelly_kf / 2.0
            )

        high_arr: NDArray = self._high if use_ohlcv and self._high is not None else c
        low_arr: NDArray = self._low if use_ohlcv and self._low is not None else c
        trades = _build_trades_df(positions, fill_prices, high_arr, low_arr, ic)

        dd_arr, _ = _rust_drawdown_series(equity)
        drawdown_series = np.asarray(dd_arr)

        return AdvancedBacktestResult(
            signals=signals,
            positions=positions,
            bar_returns=bar_returns,
            strategy_returns=strategy_returns,
            equity=equity,
            metrics=metrics,
            trades=trades,
            drawdown_series=drawdown_series,
            fill_prices=fill_prices,
            currency=self._currency,
            initial_capital=ic,
        )


# ---------------------------------------------------------------------------
# Additional built-in strategies
# ---------------------------------------------------------------------------


def adx_trend_follow_strategy(
    close: ArrayLike,
    high: Optional[ArrayLike] = None,
    low: Optional[ArrayLike] = None,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    sma_period: int = 50,
    **kwargs: object,
) -> NDArray:
    """ADX trend-following: +1 when ADX>threshold AND close>SMA, else -1."""
    from ferro_ta._ferro_ta import adx as _adx
    from ferro_ta._ferro_ta import sma as _sma

    c = np.asarray(close, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64) if high is not None else c * 1.001
    low_arr = np.asarray(low, dtype=np.float64) if low is not None else c * 0.999

    adx_vals = np.asarray(_adx(h, low_arr, c, adx_period), dtype=np.float64)
    sma_vals = np.asarray(_sma(c, sma_period), dtype=np.float64)

    out = np.where(
        np.isnan(adx_vals) | np.isnan(sma_vals),
        np.nan,
        np.where((adx_vals > adx_threshold) & (c > sma_vals), 1.0, -1.0),
    )
    return out


def bb_mean_revert_strategy(
    close: ArrayLike,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    **kwargs: object,
) -> NDArray:
    """Bollinger Band mean reversion: +1 near lower band, -1 near upper band."""
    from ferro_ta._ferro_ta import bbands as _bbands

    c = np.asarray(close, dtype=np.float64)
    upper, middle, lower = _bbands(c, timeperiod, nbdevup, nbdevdn)
    upper = np.asarray(upper, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)

    out = np.where(
        np.isnan(upper) | np.isnan(lower),
        np.nan,
        np.where(c <= lower, 1.0, np.where(c >= upper, -1.0, 0.0)),
    )
    return out


def rsi_sma_combo_strategy(
    close: ArrayLike,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    sma_period: int = 50,
    **kwargs: object,
) -> NDArray:
    """RSI signal filtered by SMA trend: RSI oversold/overbought only in trend direction."""
    from ferro_ta._ferro_ta import sma as _sma

    c = np.asarray(close, dtype=np.float64)
    rsi_signals = rsi_strategy(c, rsi_period, oversold, overbought)
    sma_vals = np.asarray(_sma(c, sma_period), dtype=np.float64)

    trend = np.where(np.isnan(sma_vals), np.nan, np.where(c > sma_vals, 1.0, -1.0))
    # Only take RSI long signals in uptrend, RSI short signals in downtrend
    out = np.where(
        np.isnan(rsi_signals) | np.isnan(trend),
        np.nan,
        np.where(
            (rsi_signals == 1.0) & (trend == 1.0),
            1.0,
            np.where((rsi_signals == -1.0) & (trend == -1.0), -1.0, 0.0),
        ),
    )
    return out


# Register additional built-in strategies
_BUILTIN_STRATEGIES["adx_trend_follow"] = adx_trend_follow_strategy
_BUILTIN_STRATEGIES["bb_mean_revert"] = bb_mean_revert_strategy
_BUILTIN_STRATEGIES["rsi_sma_combo"] = rsi_sma_combo_strategy


# ---------------------------------------------------------------------------
# WalkForwardResult + walk_forward()
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WalkForwardResult:
    """Results from walk-forward analysis.

    Attributes
    ----------
    fold_results : list[AdvancedBacktestResult]
        Out-of-sample backtest result for each fold.
    fold_indices : NDArray[np.int64]
        Shape (n_folds, 4): [train_start, train_end, test_start, test_end].
    best_params_per_fold : list[dict]
        Parameter dict that scored highest in each fold's training period.
    oos_equity : NDArray[np.float64]
        Concatenated out-of-sample equity curve (chained, not spliced raw).
    oos_metrics : dict[str, float]
        Performance metrics computed on the full OOS equity curve.
    param_stability : dict[str, Any]
        For each param name, the most-chosen value and its selection frequency.
    """

    fold_results: list
    fold_indices: NDArray
    best_params_per_fold: list
    oos_equity: NDArray
    oos_metrics: dict
    param_stability: dict


def walk_forward(
    close: ArrayLike,
    strategy_fn: Callable,
    param_grid: list,
    train_bars: int,
    test_bars: int,
    *,
    metric: str = "sharpe",
    anchored: bool = False,
    step_bars: int = 0,
    commission_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    periods_per_year: float = 252.0,
) -> WalkForwardResult:
    """Walk-forward analysis with grid search on each training fold.

    Parameters
    ----------
    close : array-like
        Full close price series.
    strategy_fn : callable
        Signal-generating function ``(close, **params) -> signals``.
    param_grid : list[dict]
        List of parameter dicts to test on the training set.
    train_bars : int
        Number of bars in each training window.
    test_bars : int
        Number of bars in each test (out-of-sample) window.
    metric : str
        Metric name from ``compute_performance_metrics`` to optimise (default "sharpe").
    anchored : bool
        If True, training window always starts from bar 0 (expanding window).
    step_bars : int
        Step between folds. 0 → non-overlapping (step = test_bars).
    commission_per_trade, slippage_bps : float
        Applied in both training (for metric computation) and test.
    periods_per_year : float
        Annualisation factor for metrics (default 252).

    Returns
    -------
    WalkForwardResult
    """
    if metric in _BENCHMARK_METRICS:
        raise FerroTAValueError(
            f"metric '{metric}' requires a benchmark and is not supported in walk_forward(). "
            f"Use a non-benchmark metric such as 'sharpe', 'cagr', or 'sortino'."
        )

    c = np.asarray(close, dtype=np.float64)
    n = len(c)

    fold_idx = np.asarray(
        _rust_walk_forward_indices(n, train_bars, test_bars, anchored, step_bars),
        dtype=np.int64,
    )

    fold_results: list = []
    best_params_per_fold: list = []
    oos_returns_parts: list = []

    engine_base = (
        BacktestEngine()
        .with_commission(commission_per_trade)
        .with_slippage(slippage_bps)
        .with_calendar(periods_per_year)
    )

    for fold in fold_idx:
        tr_start, tr_end, te_start, te_end = (
            int(fold[0]),
            int(fold[1]),
            int(fold[2]),
            int(fold[3]),
        )
        c_train = c[tr_start:tr_end]
        c_test = c[te_start:te_end]

        # Grid search on training set
        best_params: dict = param_grid[0] if param_grid else {}
        best_score = float("-inf")

        for params in param_grid:
            try:
                signals_train = np.asarray(
                    strategy_fn(c_train, **params), dtype=np.float64
                )
                _, _, sr_train, eq_train = _rust_backtest_core(
                    c_train,
                    signals_train,
                    commission_per_trade=commission_per_trade,
                    slippage_bps=slippage_bps,
                )
                sr_train = np.asarray(sr_train, dtype=np.float64)
                eq_train = np.asarray(eq_train, dtype=np.float64)
                fold_metrics = dict(
                    _rust_compute_perf_metrics(
                        sr_train, eq_train, periods_per_year, 0.0
                    )
                )
                score = fold_metrics.get(metric, float("-inf"))
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as exc:
                warnings.warn(
                    f"walk_forward: training fold param evaluation failed: {exc}",
                    stacklevel=2,
                )
                continue

        best_params_per_fold.append(best_params)

        # Test with best params
        try:
            test_result = engine_base.run(c_test, strategy_fn, **best_params)
        except Exception as exc:
            warnings.warn(
                f"walk_forward: test fold failed, using flat equity: {exc}",
                stacklevel=2,
            )
            dummy = np.ones(len(c_test))
            test_result = AdvancedBacktestResult(
                signals=dummy,
                positions=dummy,
                bar_returns=dummy,
                strategy_returns=np.zeros(len(c_test)),
                equity=dummy,
                metrics={},
                trades=None,
                drawdown_series=np.zeros(len(c_test)),
                fill_prices=np.full(len(c_test), np.nan),
            )

        fold_results.append(test_result)
        oos_returns_parts.append(test_result.strategy_returns)

    # Chain OOS equity curves from per-fold equity (preserves commission deductions)
    if fold_results:
        oos_equity_parts: list[NDArray] = []
        oos_returns = np.concatenate(oos_returns_parts)
        cumulative = 1.0
        for fr in fold_results:
            fold_eq = np.asarray(fr.equity, dtype=np.float64)
            # Renormalize: fold equity starts at 1.0, scale to chain from prior fold
            oos_equity_parts.append(fold_eq * cumulative)
            cumulative *= float(fold_eq[-1]) if len(fold_eq) > 0 else 1.0
        oos_equity = np.concatenate(oos_equity_parts)
    else:
        oos_returns = np.array([0.0])
        oos_equity = np.array([1.0])

    # OOS metrics on full concatenated curve
    try:
        oos_metrics = dict(
            _rust_compute_perf_metrics(oos_returns, oos_equity, periods_per_year, 0.0)
        )
    except Exception:
        oos_metrics = {}

    # Parameter stability: how often each param value was chosen
    param_stability: dict = {}
    if best_params_per_fold:
        all_keys = set().union(*[p.keys() for p in best_params_per_fold])
        for key in all_keys:
            vals = [p.get(key) for p in best_params_per_fold if key in p]
            counts = Counter(vals)
            most_common_val, most_common_count = counts.most_common(1)[0]
            param_stability[key] = {
                "most_chosen": most_common_val,
                "frequency": most_common_count / len(best_params_per_fold),
                "counts": dict(counts),
            }

    return WalkForwardResult(
        fold_results=fold_results,
        fold_indices=fold_idx,
        best_params_per_fold=best_params_per_fold,
        oos_equity=oos_equity,
        oos_metrics=oos_metrics,
        param_stability=param_stability,
    )


# ---------------------------------------------------------------------------
# MonteCarloResult + monte_carlo()
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MonteCarloResult:
    """Results from Monte Carlo bootstrap simulation.

    Attributes
    ----------
    equity_curves : NDArray[np.float64]
        Shape (n_sims, n_bars) — simulated equity curves.
    terminal_equity : NDArray[np.float64]
        Shape (n_sims,) — final equity value per simulation.
    confidence_lower : NDArray[np.float64]
        Lower confidence band per bar.
    confidence_upper : NDArray[np.float64]
        Upper confidence band per bar.
    median_curve : NDArray[np.float64]
        Median equity curve across simulations.
    var : float
        Value-at-Risk: worst ``(1-confidence)`` percentile of terminal equity.
    cvar : float
        Conditional VaR: mean of worst ``(1-confidence)`` fraction of terminal equity.
    prob_profit : float
        Fraction of simulations where terminal equity > 1.0.
    n_sims : int
    confidence : float
    """

    equity_curves: NDArray
    terminal_equity: NDArray
    confidence_lower: NDArray
    confidence_upper: NDArray
    median_curve: NDArray
    var: float
    cvar: float
    prob_profit: float
    n_sims: int
    confidence: float


def monte_carlo(
    result_or_returns: Union[BacktestResult, NDArray],
    n_sims: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    block_size: int = 1,
) -> MonteCarloResult:
    """Run Monte Carlo bootstrap simulation on strategy returns.

    Parameters
    ----------
    result_or_returns : BacktestResult or array-like
        Either a ``BacktestResult`` (uses its ``strategy_returns``) or
        a 1-D array of returns directly.
    n_sims : int
        Number of bootstrap simulations (default 1000).
    confidence : float
        Confidence level for bands and VaR (default 0.95).
    seed : int
        Random seed for reproducibility.
    block_size : int
        Block size for stationary block bootstrap (1 = IID resample).

    Returns
    -------
    MonteCarloResult
    """
    if isinstance(result_or_returns, BacktestResult):
        returns = np.asarray(result_or_returns.strategy_returns, dtype=np.float64)
    else:
        returns = np.asarray(result_or_returns, dtype=np.float64)

    equity_curves = np.asarray(
        _rust_monte_carlo_bootstrap(returns, int(n_sims), int(seed), int(block_size)),
        dtype=np.float64,
    )

    terminal_equity = equity_curves[:, -1]

    lower_pct = (1.0 - confidence) / 2.0 * 100.0
    upper_pct = (1.0 + confidence) / 2.0 * 100.0

    pct_results = np.percentile(equity_curves, [lower_pct, upper_pct, 50.0], axis=0)
    confidence_lower = pct_results[0]
    confidence_upper = pct_results[1]
    median_curve = pct_results[2]

    var_threshold = np.percentile(terminal_equity, (1.0 - confidence) * 100.0)
    tail = terminal_equity[terminal_equity <= var_threshold]
    cvar = float(np.mean(tail)) if len(tail) > 0 else float(var_threshold)

    prob_profit = float(np.mean(terminal_equity > 1.0))

    return MonteCarloResult(
        equity_curves=equity_curves,
        terminal_equity=terminal_equity,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        median_curve=median_curve,
        var=float(var_threshold),
        cvar=cvar,
        prob_profit=prob_profit,
        n_sims=int(n_sims),
        confidence=float(confidence),
    )


# ---------------------------------------------------------------------------
# Portfolio backtest
# ---------------------------------------------------------------------------


def backtest_portfolio(
    close_2d: ArrayLike,
    weights_2d: ArrayLike,
    *,
    commission_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    periods_per_year: float = 252.0,
    parallel: bool = True,
    max_asset_weight: float = 1.0,
    max_gross_exposure: float = 0.0,
    max_net_exposure: float = 0.0,
) -> PortfolioBacktestResult:
    """Backtest a portfolio of N assets in parallel.

    Parameters
    ----------
    close_2d : array-like, shape (n_bars, n_assets)
        Close prices for each asset.
    weights_2d : array-like, shape (n_bars, n_assets)
        Desired position per asset per bar (lagged internally like signals).
    commission_per_trade : float
        Per-position-change commission (default 0).
    slippage_bps : float
        Slippage in basis points (default 0).
    periods_per_year : float
        Annualisation factor for metrics (default 252).
    parallel : bool
        Use rayon parallelism (default True).

    Returns
    -------
    PortfolioBacktestResult
    """
    c2d = np.ascontiguousarray(close_2d, dtype=np.float64)
    w2d = np.ascontiguousarray(weights_2d, dtype=np.float64)

    asset_returns, portfolio_returns, portfolio_equity = (
        _rust_backtest_multi_asset_core(
            c2d,
            w2d,
            commission_per_trade,
            slippage_bps,
            parallel,
            max_asset_weight,
            max_gross_exposure,
            max_net_exposure,
        )
    )
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)
    portfolio_equity = np.asarray(portfolio_equity, dtype=np.float64)

    metrics = dict(
        _rust_compute_perf_metrics(
            portfolio_returns, portfolio_equity, periods_per_year, 0.0
        )
    )

    return PortfolioBacktestResult(
        asset_returns=asset_returns,
        portfolio_returns=portfolio_returns,
        portfolio_equity=portfolio_equity,
        metrics=metrics,
    )


@dataclasses.dataclass
class PortfolioBacktestResult:
    """Result from a multi-asset portfolio backtest.

    Attributes
    ----------
    asset_returns : NDArray[np.float64]
        Shape (n_bars, n_assets) — per-asset strategy returns.
    portfolio_returns : NDArray[np.float64]
        Shape (n_bars,) — combined portfolio returns.
    portfolio_equity : NDArray[np.float64]
        Shape (n_bars,) — cumulative portfolio equity.
    metrics : dict[str, float]
        Full performance metrics on the portfolio equity curve.
    """

    asset_returns: NDArray
    portfolio_returns: NDArray
    portfolio_equity: NDArray
    metrics: dict
