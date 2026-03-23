"""
Batch Execution API — run indicators on multiple series in a single call.

This module provides a 2-D batch API that accepts a 2-D numpy array
(n_samples × n_series) and applies an indicator to every column, returning
a 2-D output array of the same shape.

For the most common indicators — SMA, EMA, RSI — the 2-D path is handled
entirely in Rust (a single GIL release for all columns). The generic
``batch_apply`` is available for other indicators that do not have a Rust
batch implementation.

Functions
---------
batch_sma    — SMA on every column of a 2-D array  (Rust fast path for 2-D)
batch_ema    — EMA on every column of a 2-D array  (Rust fast path for 2-D)
batch_rsi    — RSI on every column of a 2-D array  (Rust fast path for 2-D)
batch_apply  — Generic batch wrapper (Python loop) for any arbitrary indicator

Usage
-----
>>> import numpy as np
>>> from ferro_ta.data.batch import batch_sma
>>> data = np.random.rand(100, 5)   # 100 bars, 5 symbols
>>> result = batch_sma(data, timeperiod=14)
>>> result.shape
(100, 5)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    batch_adx as _rust_batch_adx,
)
from ferro_ta._ferro_ta import (
    batch_atr as _rust_batch_atr,
)
from ferro_ta._ferro_ta import (
    batch_ema as _rust_batch_ema,
)
from ferro_ta._ferro_ta import (
    batch_rsi as _rust_batch_rsi,
)
from ferro_ta._ferro_ta import (
    batch_sma as _rust_batch_sma,
)
from ferro_ta._ferro_ta import (
    batch_stoch as _rust_batch_stoch,
)
from ferro_ta.indicators.momentum import RSI
from ferro_ta.indicators.overlap import EMA, SMA

__all__ = [
    "batch_sma",
    "batch_ema",
    "batch_rsi",
    "batch_apply",
]


def batch_apply(
    data: ArrayLike,
    fn: Callable[..., np.ndarray],
    **kwargs,
) -> np.ndarray:
    """Apply any single-series indicator *fn* to every column of *data*.

    This is the generic fallback batch executor — it calls *fn* once per
    column in a Python loop.  For the common indicators SMA, EMA, and RSI
    prefer the dedicated :func:`batch_sma`, :func:`batch_ema`, and
    :func:`batch_rsi` functions, which use a Rust-side loop and avoid
    per-column Python round-trips.

    Parameters
    ----------
    data : array-like, shape (n_samples,) or (n_samples, n_series)
        Input data.  If 1-D, the function is called directly on the array
        and the result is returned without adding a column dimension.
    fn : callable
        Single-series indicator function (e.g. ``SMA``, ``EMA``, ``RSI``).
        It must accept a 1-D array as first positional argument and return
        a 1-D array of the same length.
    **kwargs
        Extra keyword arguments forwarded to *fn* (e.g. ``timeperiod=14``).

    Returns
    -------
    numpy.ndarray
        Same shape as *data*.  Leading values are ``NaN`` for the warm-up
        period, identical to calling *fn* on each column individually.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta import SMA
    >>> from ferro_ta.data.batch import batch_apply
    >>> data = np.random.rand(50, 3)
    >>> out = batch_apply(data, SMA, timeperiod=5)
    >>> out.shape
    (50, 3)
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        return fn(arr, **kwargs)
    if arr.ndim != 2:
        raise ValueError(f"batch_apply expects 1-D or 2-D input; got {arr.ndim}-D")

    n_samples, n_series = arr.shape
    result = np.empty((n_samples, n_series), dtype=np.float64)
    for j in range(n_series):
        result[:, j] = fn(arr[:, j], **kwargs)
    return result


def batch_sma(
    data: ArrayLike,
    timeperiod: int = 30,
    parallel: bool = True,
) -> np.ndarray:
    """Simple Moving Average on every column of *data*.

    For 2-D inputs uses a Rust-side column loop (single GIL release).
    When *parallel* is ``True`` (default), columns are processed in parallel
    via Rayon across all available CPU cores.
    1-D input is passed directly to the single-series SMA.

    Parameters
    ----------
    data : array-like, shape (n_samples,) or (n_samples, n_series)
    timeperiod : int, default 30
    parallel : bool, default True
        Enable multi-threaded parallel column processing via Rayon.
        Set to ``False`` for small inputs where thread overhead dominates.

    Returns
    -------
    numpy.ndarray — same shape as *data*.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.data.batch import batch_sma
    >>> data = np.arange(1.0, 101.0).reshape(100, 1).repeat(3, axis=1)
    >>> out = batch_sma(data, timeperiod=10)
    >>> out.shape
    (100, 3)
    """
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim == 1:
        return SMA(arr, timeperiod=timeperiod)
    if arr.ndim != 2:
        raise ValueError(f"batch_sma expects 1-D or 2-D input; got {arr.ndim}-D")
    return np.asarray(_rust_batch_sma(arr, timeperiod, parallel))


def batch_ema(
    data: ArrayLike,
    timeperiod: int = 30,
    parallel: bool = True,
) -> np.ndarray:
    """Exponential Moving Average on every column of *data*.

    For 2-D inputs uses a Rust-side column loop (single GIL release).
    When *parallel* is ``True`` (default), columns are processed in parallel
    via Rayon across all available CPU cores.

    Parameters
    ----------
    data : array-like, shape (n_samples,) or (n_samples, n_series)
    timeperiod : int, default 30
    parallel : bool, default True
        Enable multi-threaded parallel column processing via Rayon.

    Returns
    -------
    numpy.ndarray — same shape as *data*.
    """
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim == 1:
        return EMA(arr, timeperiod=timeperiod)
    if arr.ndim != 2:
        raise ValueError(f"batch_ema expects 1-D or 2-D input; got {arr.ndim}-D")
    return np.asarray(_rust_batch_ema(arr, timeperiod, parallel))


def batch_rsi(
    data: ArrayLike,
    timeperiod: int = 14,
    parallel: bool = True,
) -> np.ndarray:
    """Relative Strength Index on every column of *data*.

    For 2-D inputs uses a Rust-side column loop (single GIL release).
    When *parallel* is ``True`` (default), columns are processed in parallel
    via Rayon across all available CPU cores.

    Parameters
    ----------
    data : array-like, shape (n_samples,) or (n_samples, n_series)
    timeperiod : int, default 14
    parallel : bool, default True
        Enable multi-threaded parallel column processing via Rayon.

    Returns
    -------
    numpy.ndarray — same shape as *data*.  Values in [0, 100].
    """
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim == 1:
        return RSI(arr, timeperiod=timeperiod)
    if arr.ndim != 2:
        raise ValueError(f"batch_rsi expects 1-D or 2-D input; got {arr.ndim}-D")
    return np.asarray(_rust_batch_rsi(arr, timeperiod, parallel))


def batch_atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
    parallel: bool = True,
) -> np.ndarray:
    h = np.ascontiguousarray(high, dtype=np.float64)
    low_arr = np.ascontiguousarray(low, dtype=np.float64)
    c = np.ascontiguousarray(close, dtype=np.float64)
    return np.asarray(_rust_batch_atr(h, low_arr, c, timeperiod, parallel))


def batch_stoch(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
    parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.ascontiguousarray(high, dtype=np.float64)
    low_arr = np.ascontiguousarray(low, dtype=np.float64)
    c = np.ascontiguousarray(close, dtype=np.float64)
    k, d = _rust_batch_stoch(
        h, low_arr, c, fastk_period, slowk_period, slowd_period, parallel
    )
    return np.asarray(k), np.asarray(d)


def batch_adx(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
    parallel: bool = True,
) -> np.ndarray:
    h = np.ascontiguousarray(high, dtype=np.float64)
    low_arr = np.ascontiguousarray(low, dtype=np.float64)
    c = np.ascontiguousarray(close, dtype=np.float64)
    return np.asarray(_rust_batch_adx(h, low_arr, c, timeperiod, parallel))
