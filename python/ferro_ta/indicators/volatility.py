"""
Volatility Indicators — Measure the magnitude of price fluctuations.

Functions
---------
ATR   — Average True Range
NATR  — Normalized Average True Range
TRANGE — True Range
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    atr as _atr,
)
from ferro_ta._ferro_ta import (
    natr as _natr,
)
from ferro_ta._ferro_ta import (
    trange as _trange,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def ATR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Average True Range.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of ATR values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _atr(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def NATR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Normalized Average True Range.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of NATR values (percentage); leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _natr(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def TRANGE(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """True Range.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    numpy.ndarray
        Array of True Range values.
    """
    try:
        return _trange(_to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = ["ATR", "NATR", "TRANGE"]
