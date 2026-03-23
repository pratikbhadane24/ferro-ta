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

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta.core.exceptions import FerroTAInputError, FerroTAValueError

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
    from ferro_ta import RSI  # local import to avoid circular dep

    if timeperiod < 1:
        raise FerroTAValueError(f"timeperiod must be >= 1, got {timeperiod}")

    c = np.asarray(close, dtype=np.float64)
    rsi = np.asarray(RSI(c, timeperiod=timeperiod), dtype=np.float64)
    signals = np.where(rsi <= oversold, 1.0, np.where(rsi >= overbought, -1.0, 0.0))
    signals[np.isnan(rsi)] = np.nan
    return signals


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
    from ferro_ta import SMA  # local import

    if fast < 1:
        raise FerroTAValueError(f"fast must be >= 1, got {fast}")
    if slow < 1:
        raise FerroTAValueError(f"slow must be >= 1, got {slow}")
    if fast >= slow:
        raise FerroTAValueError(f"fast ({fast}) must be less than slow ({slow})")

    c = np.asarray(close, dtype=np.float64)
    sma_fast = np.asarray(SMA(c, timeperiod=fast), dtype=np.float64)
    sma_slow = np.asarray(SMA(c, timeperiod=slow), dtype=np.float64)
    signals = np.where(sma_fast > sma_slow, 1.0, -1.0).astype(np.float64)
    # Warm-up: NaN where either MA is NaN
    warmup = np.isnan(sma_fast) | np.isnan(sma_slow)
    signals[warmup] = np.nan
    return signals


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
    from ferro_ta import MACD  # local import

    if fastperiod < 1 or slowperiod < 1 or signalperiod < 1:
        raise FerroTAValueError("MACD periods must be >= 1")
    if fastperiod >= slowperiod:
        raise FerroTAValueError(
            f"fastperiod ({fastperiod}) must be less than slowperiod ({slowperiod})"
        )

    c = np.asarray(close, dtype=np.float64)
    macd_line, signal_line, _ = MACD(
        c, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
    )
    macd_line = np.asarray(macd_line, dtype=np.float64)
    signal_line = np.asarray(signal_line, dtype=np.float64)
    signals = np.where(macd_line > signal_line, 1.0, -1.0).astype(np.float64)
    warmup = np.isnan(macd_line) | np.isnan(signal_line)
    signals[warmup] = np.nan
    return signals


# ---------------------------------------------------------------------------
# Built-in strategy registry
# ---------------------------------------------------------------------------

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
    # Resolve strategy
    # ------------------------------------------------------------------
    if isinstance(strategy, str):
        if strategy not in _BUILTIN_STRATEGIES:
            raise FerroTAValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {sorted(_BUILTIN_STRATEGIES)}"
            )
        strategy_fn: Callable[..., NDArray[np.float64]] = _BUILTIN_STRATEGIES[strategy]
    elif callable(strategy):
        strategy_fn = strategy
    else:
        raise FerroTAValueError("strategy must be a string name or a callable.")

    # ------------------------------------------------------------------
    # Compute signals
    # ------------------------------------------------------------------
    signals = np.asarray(strategy_fn(c, **strategy_kwargs), dtype=np.float64)

    # ------------------------------------------------------------------
    # Positions: lag signals by 1 bar to avoid look-ahead bias
    # ------------------------------------------------------------------
    positions = np.empty_like(signals)
    positions[0] = 0.0
    positions[1:] = signals[:-1]
    # Replace NaN in positions with 0 (flat)
    positions = np.nan_to_num(positions, nan=0.0)

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------
    bar_returns: np.ndarray = np.empty(len(c), dtype=np.float64)
    bar_returns[0] = 0.0
    bar_returns[1:] = np.diff(c) / c[:-1]

    strategy_returns = positions * bar_returns
    position_changed = np.concatenate([[False], positions[1:] != positions[:-1]])

    # Slippage: on each position change, reduce return by slippage_bps/10000 (one-way)
    if slippage_bps > 0:
        strategy_returns = strategy_returns.copy()
        strategy_returns[position_changed] -= slippage_bps / 10_000.0

    # Cumulative equity: with optional commission per trade
    if commission_per_trade <= 0:
        equity = np.cumprod(1.0 + strategy_returns)
    else:
        gross_equity = np.cumprod(1.0 + strategy_returns)
        if np.any(gross_equity == 0.0):
            equity = np.empty(len(c), dtype=np.float64)
            equity[0] = 1.0
            for i in range(1, len(c)):
                equity[i] = equity[i - 1] * (1.0 + strategy_returns[i])
                if position_changed[i]:
                    equity[i] -= commission_per_trade
        else:
            commissions = position_changed.astype(np.float64) * commission_per_trade
            discounted_commissions = np.cumsum(commissions / gross_equity)
            equity = gross_equity * (1.0 - discounted_commissions)

    return BacktestResult(
        signals=signals,
        positions=positions,
        bar_returns=bar_returns,
        strategy_returns=strategy_returns,
        equity=np.asarray(equity, dtype=np.float64),
    )
