"""
ferro_ta.crypto — Crypto and 24/7 market helpers.
=================================================

Helpers designed for continuous (24/7) markets such as cryptocurrency or FX.

Functions
---------
funding_pnl(position_size, funding_rate)
    Compute the cumulative PnL from periodic funding rate payments.

continuous_bar_labels(n_bars, period_bars)
    Assign integer period labels to bars without calendar-based sessions.

session_boundaries(timestamps_ns)
    Return bar indices at the start of each UTC-day session boundary.

resample_continuous(ohlcv, period_bars)
    Resample a continuous OHLCV series by grouping every *period_bars* input
    bars into one output bar (no session filtering).

Rust backend
------------
    ferro_ta._ferro_ta.funding_cumulative_pnl
    ferro_ta._ferro_ta.continuous_bar_labels
    ferro_ta._ferro_ta.mark_session_boundaries
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import (
    continuous_bar_labels as _rust_continuous_bar_labels,
)
from ferro_ta._ferro_ta import (
    funding_cumulative_pnl as _rust_funding_cumulative_pnl,
)
from ferro_ta._ferro_ta import (
    mark_session_boundaries as _rust_mark_session_boundaries,
)
from ferro_ta._ferro_ta import (
    ohlcv_agg as _rust_ohlcv_agg,
)
from ferro_ta._utils import _to_f64

__all__ = [
    "funding_pnl",
    "continuous_bar_labels",
    "session_boundaries",
    "resample_continuous",
]

# type alias
OHLCVTuple = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


def funding_pnl(
    position_size: ArrayLike,
    funding_rate: ArrayLike,
) -> NDArray[np.float64]:
    """Compute cumulative PnL from periodic funding rate payments.

    Crypto perpetual contracts charge a periodic funding rate to position
    holders.  A long position pays when the funding rate is positive; a short
    position receives.

    PnL at period *i* = ``-position_size[i] * funding_rate[i]``
    Returned array is the cumulative sum of those per-period PnLs.

    Parameters
    ----------
    position_size : array-like — signed position size per funding period.
        Positive = long, negative = short.
    funding_rate  : array-like — periodic funding rate in decimal notation
        (e.g. 0.0001 = 0.01%).  Must have the same length as *position_size*.

    Returns
    -------
    numpy.ndarray of float64 — cumulative funding PnL.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.crypto import funding_pnl
    >>> pos = np.ones(5)            # long 1 contract
    >>> rate = np.array([0.0001, 0.0002, -0.0001, 0.0001, 0.0001])
    >>> pnl = funding_pnl(pos, rate)
    >>> pnl.round(6)
    array([-0.0001, -0.0003,  0.    , -0.0001, -0.0002])
    """
    return np.asarray(
        _rust_funding_cumulative_pnl(_to_f64(position_size), _to_f64(funding_rate)),
        dtype=np.float64,
    )


def continuous_bar_labels(
    n_bars: int,
    period_bars: int,
) -> NDArray[np.int64]:
    """Assign sequential integer labels to bars in equal-size buckets.

    Useful for grouping continuous data (no session gaps) into periods without
    relying on calendar logic.  Bars 0…(period_bars-1) get label 0,
    bars period_bars…(2·period_bars-1) get label 1, etc.

    Parameters
    ----------
    n_bars      : int — total number of bars
    period_bars : int — number of bars per period (e.g. 24 for hourly → daily)

    Returns
    -------
    numpy.ndarray of int64 — period label per bar.

    Examples
    --------
    >>> from ferro_ta.analysis.crypto import continuous_bar_labels
    >>> continuous_bar_labels(10, 3)
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    """
    return np.asarray(
        _rust_continuous_bar_labels(int(n_bars), int(period_bars)),
        dtype=np.int64,
    )


def session_boundaries(
    timestamps_ns: ArrayLike,
) -> NDArray[np.int64]:
    """Return bar indices at the start of each UTC-day boundary.

    Intended for 24/7 data where no exchange session gaps exist.  Useful for
    building daily OHLCV bars from intraday continuous data.

    Parameters
    ----------
    timestamps_ns : array-like of int64 — UTC timestamps in nanoseconds
        (e.g. ``pandas.DatetimeIndex.astype('int64')``).

    Returns
    -------
    numpy.ndarray of int64 — indices of the first bar in each UTC day
    (always includes index 0).

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.crypto import session_boundaries
    >>> # Two UTC days of hourly bars: day 0 = bars 0-23, day 1 = bars 24-47
    >>> base_ns = np.int64(1_700_000_000_000_000_000)  # some UTC timestamp
    >>> ns_per_hour = np.int64(3_600_000_000_000)
    >>> ts = base_ns + np.arange(48, dtype=np.int64) * ns_per_hour
    >>> bounds = session_boundaries(ts)
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    return np.asarray(
        _rust_mark_session_boundaries(ts),
        dtype=np.int64,
    )


def resample_continuous(
    ohlcv: Union[
        tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike],
        object,  # pandas.DataFrame
    ],
    period_bars: int,
) -> OHLCVTuple:
    """Resample a continuous OHLCV series by grouping *period_bars* input bars.

    Unlike time-based resampling, this function requires no calendar or
    session information.  Every *period_bars* consecutive input bars are
    aggregated into one output bar.  Ideal for 24/7 crypto data.

    Parameters
    ----------
    ohlcv : tuple ``(open, high, low, close, volume)`` of array-like,
        **or** a ``pandas.DataFrame`` with columns ``open/high/low/close/volume``
        (case-insensitive).
    period_bars : int — number of input bars per output bar (must be >= 1).

    Returns
    -------
    tuple ``(open, high, low, close, volume)`` of numpy.ndarray — resampled bars.

    Notes
    -----
    The last output bar may aggregate fewer than *period_bars* input bars if
    ``len(close) % period_bars != 0``.
    """
    try:
        import pandas as pd

        if isinstance(ohlcv, pd.DataFrame):
            cols = {c.lower(): c for c in ohlcv.columns}  # type: ignore[union-attr]
            o = _to_f64(ohlcv[cols["open"]].values)  # type: ignore[index]
            h = _to_f64(ohlcv[cols["high"]].values)  # type: ignore[index]
            lo = _to_f64(ohlcv[cols["low"]].values)  # type: ignore[index]
            c = _to_f64(ohlcv[cols["close"]].values)  # type: ignore[index]
            v = _to_f64(ohlcv[cols["volume"]].values)  # type: ignore[index]
        else:
            o, h, lo, c, v = [_to_f64(x) for x in ohlcv]  # type: ignore[union-attr]
    except ImportError:
        o, h, lo, c, v = [_to_f64(x) for x in ohlcv]  # type: ignore[union-attr]

    n = len(c)
    if period_bars < 1:
        raise ValueError("period_bars must be >= 1")
    # Build bar-group labels
    labels = np.asarray(
        _rust_continuous_bar_labels(n, int(period_bars)),
        dtype=np.int64,
    )
    ro, rh, rl, rc, rv = _rust_ohlcv_agg(o, h, lo, c, v, labels)
    return (
        np.asarray(ro, dtype=np.float64),
        np.asarray(rh, dtype=np.float64),
        np.asarray(rl, dtype=np.float64),
        np.asarray(rc, dtype=np.float64),
        np.asarray(rv, dtype=np.float64),
    )
