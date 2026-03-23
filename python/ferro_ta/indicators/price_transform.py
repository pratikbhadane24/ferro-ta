"""
Price Transformations — Helper functions to synthesize OHLC arrays into single arrays.

Functions
---------
AVGPRICE — Average Price: (Open + High + Low + Close) / 4
MEDPRICE — Median Price: (High + Low) / 2
TYPPRICE — Typical Price: (High + Low + Close) / 3
WCLPRICE — Weighted Close Price: (High + Low + Close * 2) / 4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    avgprice as _avgprice,
)
from ferro_ta._ferro_ta import (
    medprice as _medprice,
)
from ferro_ta._ferro_ta import (
    typprice as _typprice,
)
from ferro_ta._ferro_ta import (
    wclprice as _wclprice,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def AVGPRICE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Average Price: (Open + High + Low + Close) / 4.

    Parameters
    ----------
    open : array-like
        Sequence of open prices.
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    numpy.ndarray
        Array of AVGPRICE values.
    """
    try:
        return _avgprice(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def MEDPRICE(high: ArrayLike, low: ArrayLike) -> np.ndarray:
    """Median Price: (High + Low) / 2.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.

    Returns
    -------
    numpy.ndarray
        Array of MEDPRICE values.
    """
    try:
        return _medprice(_to_f64(high), _to_f64(low))
    except ValueError as e:
        _normalize_rust_error(e)


def TYPPRICE(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> np.ndarray:
    """Typical Price: (High + Low + Close) / 3.

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
        Array of TYPPRICE values.
    """
    try:
        return _typprice(_to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def WCLPRICE(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> np.ndarray:
    """Weighted Close Price: (High + Low + Close * 2) / 4.

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
        Array of WCLPRICE values.
    """
    try:
        return _wclprice(_to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
