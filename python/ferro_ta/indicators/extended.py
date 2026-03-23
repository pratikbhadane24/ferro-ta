"""
Extended Indicators — Popular indicators not in the TA-Lib standard set.

All indicator logic is implemented in Rust (PyO3) for maximum performance.
This module provides the public Python API with:
- Input validation
- ``_to_f64`` conversion
- pandas/polars-compatible return values (numpy arrays)

Functions
---------
VWAP               — Volume Weighted Average Price (cumulative or rolling)
SUPERTREND         — ATR-based trend-following signal
ICHIMOKU           — Ichimoku Cloud
DONCHIAN           — Donchian Channels
PIVOT_POINTS       — Classic / Fibonacci / Camarilla pivot levels
KELTNER_CHANNELS   — EMA ± ATR bands
HULL_MA            — Hull Moving Average (WMA-based)
CHANDELIER_EXIT    — ATR-based stop-loss / exit levels
VWMA               — Volume Weighted Moving Average
CHOPPINESS_INDEX   — Market choppiness / trending strength index

Rust backend
------------
All computations delegate to Rust functions in the ``_ferro_ta`` extension::

    from ferro_ta._ferro_ta import supertrend, donchian, vwap, ...
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Import Rust implementations
# ---------------------------------------------------------------------------
from ferro_ta._ferro_ta import (
    chandelier_exit as _rust_chandelier_exit,
)
from ferro_ta._ferro_ta import (
    choppiness_index as _rust_choppiness_index,
)
from ferro_ta._ferro_ta import (
    donchian as _rust_donchian,
)
from ferro_ta._ferro_ta import (
    hull_ma as _rust_hull_ma,
)
from ferro_ta._ferro_ta import (
    ichimoku as _rust_ichimoku,
)
from ferro_ta._ferro_ta import (
    keltner_channels as _rust_keltner_channels,
)
from ferro_ta._ferro_ta import (
    pivot_points as _rust_pivot_points,
)
from ferro_ta._ferro_ta import (
    supertrend as _rust_supertrend,
)
from ferro_ta._ferro_ta import (
    vwap as _rust_vwap,
)
from ferro_ta._ferro_ta import (
    vwma as _rust_vwma,
)
from ferro_ta._utils import _to_f64


def VWAP(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 0,
) -> np.ndarray:
    """Volume Weighted Average Price.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    volume : array-like
        Sequence of volumes.
    timeperiod : int, optional
        Rolling window length. ``0`` (default) computes a cumulative VWAP
        from bar 0 (session VWAP). Any value ``>= 1`` uses a rolling window
        of that length; the first ``timeperiod - 1`` values are ``NaN``.

    Returns
    -------
    numpy.ndarray
        Array of VWAP values.

    Notes
    -----
    Typical price is used: ``(high + low + close) / 3``.
    Implemented in Rust for maximum performance.
    """
    if timeperiod < 0:
        from ferro_ta.core.exceptions import FerroTAValueError

        raise FerroTAValueError("timeperiod must be >= 0 for VWAP")
    h = _to_f64(high)
    lo = _to_f64(low)
    c = _to_f64(close)
    v = _to_f64(volume)
    return np.asarray(_rust_vwap(h, lo, c, v, timeperiod))


def SUPERTREND(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 7,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Supertrend indicator.

    An ATR-based trend-following indicator. Returns the Supertrend line and a
    direction array.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        ATR period (default 7).
    multiplier : float, optional
        ATR multiplier for band width (default 3.0).

    Returns
    -------
    supertrend : numpy.ndarray
        The Supertrend line values. ``NaN`` during the warmup period.
    direction : numpy.ndarray
        ``1`` = uptrend (price above Supertrend), ``-1`` = downtrend.
        ``0`` during warmup.

    Notes
    -----
    Implemented in Rust — the sequential band-adjustment loop that was
    previously a Python bottleneck now runs at native speed.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta import SUPERTREND
    >>> h = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0,
    ...               12.0, 13.0, 14.0, 13.0, 12.0])
    >>> l = h - 1.0
    >>> c = (h + l) / 2.0
    >>> st, dir_ = SUPERTREND(h, l, c)
    """
    h = _to_f64(high)
    lo = _to_f64(low)
    c = _to_f64(close)
    st, d = _rust_supertrend(h, lo, c, timeperiod, multiplier)
    return np.asarray(st), np.asarray(d)


def ICHIMOKU(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ichimoku Cloud (Ichimoku Kinko Hyo).

    Parameters
    ----------
    high : array-like
    low : array-like
    close : array-like
    tenkan_period : int, default 9
        Conversion line (Tenkan-sen) period.
    kijun_period : int, default 26
        Base line (Kijun-sen) period.
    senkou_b_period : int, default 52
        Leading Span B period.
    displacement : int, default 26
        Displacement / cloud offset for Senkou A & B.

    Returns
    -------
    tenkan, kijun, senkou_a, senkou_b, chikou : numpy.ndarray
        Each is a 1-D float64 array of the same length as the inputs.

    Notes
    -----
    Implemented in Rust with O(n) monotonic deque for all rolling windows.
    """
    h = _to_f64(high)
    lo = _to_f64(low)
    c = _to_f64(close)
    t, k, sa, sb, ch = _rust_ichimoku(
        h, lo, c, tenkan_period, kijun_period, senkou_b_period, displacement
    )
    return (
        np.asarray(t),
        np.asarray(k),
        np.asarray(sa),
        np.asarray(sb),
        np.asarray(ch),
    )


def DONCHIAN(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donchian Channels — rolling highest high / lowest low.

    Parameters
    ----------
    high : array-like
    low : array-like
    timeperiod : int, default 20

    Returns
    -------
    upper, middle, lower : numpy.ndarray
        Rolling highest high, midpoint, and lowest low.

    Notes
    -----
    Implemented in Rust with O(n) monotonic deque (no Python loop).
    """
    h = _to_f64(high)
    lo = _to_f64(low)
    upper, middle, lower = _rust_donchian(h, lo, timeperiod)
    return np.asarray(upper), np.asarray(middle), np.asarray(lower)


def PIVOT_POINTS(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    method: str = "classic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot Points — support / resistance levels.

    Computes pivot points for each bar using the *previous bar's* H/L/C.
    The first bar output is NaN.

    Parameters
    ----------
    high : array-like
    low : array-like
    close : array-like
    method : {'classic', 'fibonacci', 'camarilla'}, default 'classic'

    Returns
    -------
    pivot, r1, s1, r2, s2 : numpy.ndarray

    Notes
    -----
    **Classic**: P=(H+L+C)/3; R1=2P−L; S1=2P−H; R2=P+(H−L); S2=P−(H−L)

    **Fibonacci**: P=(H+L+C)/3; R1=P+0.382*(H−L); S1=P−0.382*(H−L);
    R2=P+0.618*(H−L); S2=P−0.618*(H−L)

    **Camarilla**: P=(H+L+C)/3; R1=C+1.1*(H−L)/12; S1=C−1.1*(H−L)/12;
    R2=C+1.1*(H−L)/6; S2=C−1.1*(H−L)/6
    """
    valid_methods = {"classic", "fibonacci", "camarilla"}
    if method.lower() not in valid_methods:
        raise ValueError(
            f"Unknown pivot method '{method}'. Use 'classic', 'fibonacci', or 'camarilla'."
        )
    h = _to_f64(high)
    lo = _to_f64(low)
    c = _to_f64(close)
    pivot, r1, s1, r2, s2 = _rust_pivot_points(h, lo, c, method)
    return (
        np.asarray(pivot),
        np.asarray(r1),
        np.asarray(s1),
        np.asarray(r2),
        np.asarray(s2),
    )


def KELTNER_CHANNELS(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channels — EMA ± (multiplier × ATR).

    Parameters
    ----------
    high : array-like
    low : array-like
    close : array-like
    timeperiod : int, default 20
        EMA period for the middle band.
    atr_period : int, default 10
        ATR period for band width.
    multiplier : float, default 2.0
        ATR multiplier.

    Returns
    -------
    upper, middle, lower : numpy.ndarray

    Notes
    -----
    Implemented in Rust — EMA and ATR computed inline without Python calls.
    """
    h = _to_f64(high)
    lo = _to_f64(low)
    c = _to_f64(close)
    upper, middle, lower = _rust_keltner_channels(
        h, lo, c, timeperiod, atr_period, multiplier
    )
    return np.asarray(upper), np.asarray(middle), np.asarray(lower)


def HULL_MA(
    close: ArrayLike,
    timeperiod: int = 16,
) -> np.ndarray:
    """Hull Moving Average (HMA).

    A fast-responding moving average that reduces lag.

    Parameters
    ----------
    close : array-like
    timeperiod : int, default 16

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Formula: ``HMA(n) = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))``

    Implemented in Rust — all WMA computations are in-process.
    """
    c = _to_f64(close)
    return np.asarray(_rust_hull_ma(c, timeperiod))


def CHANDELIER_EXIT(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 22,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Chandelier Exit — ATR-based trailing stop levels.

    Parameters
    ----------
    high : array-like
    low : array-like
    close : array-like
    timeperiod : int, default 22
        Lookback period for highest high / lowest low and ATR.
    multiplier : float, default 3.0
        ATR multiplier.

    Returns
    -------
    long_exit, short_exit : numpy.ndarray

    Notes
    -----
    Implemented in Rust with O(n) monotonic deque for rolling max/min.
    """
    h = _to_f64(high)
    lo = _to_f64(low)
    c = _to_f64(close)
    long_exit, short_exit = _rust_chandelier_exit(h, lo, c, timeperiod, multiplier)
    return np.asarray(long_exit), np.asarray(short_exit)


def VWMA(
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 20,
) -> np.ndarray:
    """Volume Weighted Moving Average.

    Parameters
    ----------
    close : array-like
    volume : array-like
    timeperiod : int, default 20

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    ``VWMA = sum(close * volume, n) / sum(volume, n)``
    Implemented in Rust with O(n) prefix-sum approach.
    """
    c = _to_f64(close)
    v = _to_f64(volume)
    return np.asarray(_rust_vwma(c, v, timeperiod))


def CHOPPINESS_INDEX(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Choppiness Index — measures market choppiness (range-bound vs trending).

    Parameters
    ----------
    high : array-like
    low : array-like
    close : array-like
    timeperiod : int, default 14

    Returns
    -------
    numpy.ndarray
        Values in ``[0, 100]``. Values near 100 indicate choppy/range-bound
        markets; values near 0 indicate strong trends.

    Notes
    -----
    ``CI = 100 * log10(sum(ATR(1), n) / (highest_high − lowest_low)) / log10(n)``

    Implemented in Rust with O(n) monotonic deques (no Python loop).
    """
    h = _to_f64(high)
    lo = _to_f64(low)
    c = _to_f64(close)
    return np.asarray(_rust_choppiness_index(h, lo, c, timeperiod))


__all__ = [
    "VWAP",
    "SUPERTREND",
    "ICHIMOKU",
    "DONCHIAN",
    "PIVOT_POINTS",
    "KELTNER_CHANNELS",
    "HULL_MA",
    "CHANDELIER_EXIT",
    "VWMA",
    "CHOPPINESS_INDEX",
]
