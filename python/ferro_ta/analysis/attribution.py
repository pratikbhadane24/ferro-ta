"""
ferro_ta.attribution — Performance attribution and trade analysis.
=================================================================

Compute trade-level statistics and attribute equity-curve performance to
individual signals or time periods.  Designed to work with the output of
``ferro_ta.backtest.backtest()``.

Functions
---------
trade_stats(pnl, hold_bars)
    Compute win rate, avg win/loss, profit factor, and avg hold duration.

from_backtest(result)
    Extract the trade list (PnL per trade, hold duration) from a
    :class:`~ferro_ta.backtest.BacktestResult`.

attribution_by_month(bar_returns, timestamps)
    Attribute per-bar returns to calendar months.

attribution_by_signal(bar_returns, signal_labels)
    Attribute per-bar returns to signal labels.

TradeStats
    Named-tuple-style result container returned by ``trade_stats``.

Rust backend
------------
    ferro_ta._ferro_ta.trade_stats
    ferro_ta._ferro_ta.monthly_contribution
    ferro_ta._ferro_ta.signal_attribution
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import (
    extract_trades as _rust_extract_trades,
)
from ferro_ta._ferro_ta import (
    monthly_contribution as _rust_monthly_contribution,
)
from ferro_ta._ferro_ta import (
    signal_attribution as _rust_signal_attribution,
)
from ferro_ta._ferro_ta import (
    trade_stats as _rust_trade_stats,
)
from ferro_ta._utils import _to_f64

__all__ = [
    "TradeStats",
    "trade_stats",
    "from_backtest",
    "attribution_by_month",
    "attribution_by_signal",
]


# ---------------------------------------------------------------------------
# TradeStats container
# ---------------------------------------------------------------------------


class TradeStats:
    """Container for trade-level statistics.

    Attributes
    ----------
    win_rate      : float — fraction of trades with PnL > 0
    avg_win       : float — mean PnL of winning trades (0 if none)
    avg_loss      : float — mean PnL of losing trades (negative; 0 if none)
    profit_factor : float — gross profit / |gross loss|  (inf if no losses)
    avg_hold_bars : float — mean hold duration in bars
    n_trades      : int   — total number of trades
    """

    __slots__ = (
        "win_rate",
        "avg_win",
        "avg_loss",
        "profit_factor",
        "avg_hold_bars",
        "n_trades",
    )

    def __init__(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        profit_factor: float,
        avg_hold_bars: float,
        n_trades: int,
    ) -> None:
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.profit_factor = profit_factor
        self.avg_hold_bars = avg_hold_bars
        self.n_trades = n_trades

    def __repr__(self) -> str:
        return (
            f"TradeStats(n_trades={self.n_trades}, "
            f"win_rate={self.win_rate:.2%}, "
            f"profit_factor={self.profit_factor:.2f}, "
            f"avg_hold={self.avg_hold_bars:.1f} bars)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stats as a plain dict."""
        return {
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "avg_hold_bars": self.avg_hold_bars,
        }


# ---------------------------------------------------------------------------
# trade_stats
# ---------------------------------------------------------------------------


def trade_stats(
    pnl: ArrayLike,
    hold_bars: Optional[ArrayLike] = None,
) -> TradeStats:
    """Compute trade-level performance statistics.

    Parameters
    ----------
    pnl       : array-like — per-trade PnL (positive = win, negative = loss)
    hold_bars : array-like, optional — hold duration in bars for each trade.
        If ``None``, defaults to an array of ones (hold duration unknown).

    Returns
    -------
    :class:`TradeStats`

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.attribution import trade_stats
    >>> pnl = np.array([10.0, -5.0, 8.0, -3.0, 15.0, -2.0])
    >>> hold = np.array([5.0, 3.0, 7.0, 2.0, 10.0, 1.0])
    >>> ts = trade_stats(pnl, hold)
    >>> print(ts)
    TradeStats(n_trades=6, win_rate=50.00%, profit_factor=...)
    """
    p = _to_f64(pnl)
    n = len(p)
    if n == 0:
        raise ValueError("pnl must be non-empty")
    if hold_bars is None:
        h = np.ones(n, dtype=np.float64)
    else:
        h = _to_f64(hold_bars)

    win_rate, avg_win, avg_loss, profit_factor, avg_hold = _rust_trade_stats(p, h)
    return TradeStats(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_hold_bars=avg_hold,
        n_trades=n,
    )


# ---------------------------------------------------------------------------
# from_backtest
# ---------------------------------------------------------------------------


def from_backtest(result: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract per-trade PnL and hold durations from a BacktestResult.

    Scans the ``positions`` and ``strategy_returns`` arrays of *result* to
    find trade entries and exits, then computes per-trade PnL and duration.

    Parameters
    ----------
    result : :class:`~ferro_ta.backtest.BacktestResult`

    Returns
    -------
    tuple ``(pnl, hold_bars)`` — 1-D float64 arrays of length n_trades.

    Notes
    -----
    A "trade" is defined as a continuous run of non-zero position.  PnL is
    the sum of ``strategy_returns`` during that period.  Hold duration is
    the number of bars in the run.
    """
    pos = np.asarray(result.positions, dtype=np.float64)
    ret = np.asarray(result.strategy_returns, dtype=np.float64)
    pnl, hold = _rust_extract_trades(pos, ret)
    return (
        np.asarray(pnl, dtype=np.float64),
        np.asarray(hold, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# attribution_by_month
# ---------------------------------------------------------------------------


def attribution_by_month(
    bar_returns: ArrayLike,
    timestamps: Optional[ArrayLike] = None,
) -> dict[str, float]:
    """Attribute per-bar returns to calendar months.

    Parameters
    ----------
    bar_returns : array-like — per-bar strategy returns
    timestamps  : array-like of int64, optional — UTC timestamps in
        nanoseconds (e.g. ``pandas.DatetimeIndex.astype('int64')``).
        If ``None``, bars are grouped into calendar-agnostic monthly buckets
        of 21 bars (approximate trading month).

    Returns
    -------
    dict mapping month label (str ``'YYYY-MM'`` or ``'period_N'``) to
    total return for that month.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.attribution import attribution_by_month
    >>> rng = np.random.default_rng(0)
    >>> ret = rng.normal(0, 0.01, 252)
    >>> contrib = attribution_by_month(ret)
    >>> list(contrib.keys())[:3]
    ['period_0', 'period_1', 'period_2']
    """
    ret = _to_f64(bar_returns)
    n = len(ret)

    if timestamps is not None:
        # Convert ns timestamps → month index
        ts = np.asarray(timestamps, dtype=np.int64)
        # Month = year*12 + month_of_year (0-based)
        # ns → seconds → datetime calculation (fast path without pandas)
        try:
            import pandas as pd

            dti = pd.to_datetime(ts, unit="ns", utc=True)
            month_idx = (dti.year * 12 + dti.month - 1).astype(np.int64)  # type: ignore[union-attr]
            offset = int(month_idx[0])
            month_idx = (month_idx - offset).values.astype(np.int64)
        except ImportError:
            # Fallback: 21-bar buckets
            month_idx = np.arange(n, dtype=np.int64) // 21
    else:
        month_idx = np.arange(n, dtype=np.int64) // 21

    months_arr, contrib_arr = _rust_monthly_contribution(ret, month_idx)
    months = np.asarray(months_arr, dtype=np.int64)
    contribs = np.asarray(contrib_arr, dtype=np.float64)

    if timestamps is not None:
        try:
            import pandas as pd

            ts = np.asarray(timestamps, dtype=np.int64)
            dti = pd.to_datetime(ts, unit="ns", utc=True)
            month_idx_full = (dti.year * 12 + dti.month - 1).astype(np.int64).values  # type: ignore[union-attr]
            offset = int(month_idx_full[0])
            labels = {}
            for m, c in zip(months, contribs):
                abs_month = int(m) + offset
                year = abs_month // 12
                month_of_year = abs_month % 12 + 1
                labels[f"{year:04d}-{month_of_year:02d}"] = float(c)
            return labels
        except ImportError:
            pass

    return {f"period_{int(m)}": float(c) for m, c in zip(months, contribs)}


# ---------------------------------------------------------------------------
# attribution_by_signal
# ---------------------------------------------------------------------------


def attribution_by_signal(
    bar_returns: ArrayLike,
    signal_labels: ArrayLike,
) -> dict[str, float]:
    """Attribute per-bar returns to signal labels.

    Parameters
    ----------
    bar_returns   : array-like — per-bar strategy returns
    signal_labels : array-like of int — signal label per bar.
        Use ``-1`` for flat (no position) bars.

    Returns
    -------
    dict mapping signal label (str) to total attributed return.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.attribution import attribution_by_signal
    >>> rng = np.random.default_rng(0)
    >>> ret = rng.normal(0, 0.01, 100)
    >>> labels = np.where(np.arange(100) < 50, 0, 1)  # signal 0 or signal 1
    >>> contrib = attribution_by_signal(ret, labels)
    >>> sorted(contrib.keys())
    ['signal_0', 'signal_1']
    """
    ret = _to_f64(bar_returns)
    lbl = np.asarray(signal_labels, dtype=np.int64)
    labels_arr, contrib_arr = _rust_signal_attribution(ret, lbl)
    labels = np.asarray(labels_arr, dtype=np.int64)
    contribs = np.asarray(contrib_arr, dtype=np.float64)
    return {f"signal_{int(lbl)}": float(c) for lbl, c in zip(labels, contribs)}
