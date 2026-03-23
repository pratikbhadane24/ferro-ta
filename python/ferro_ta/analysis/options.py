"""
ferro_ta.options — Options and Implied Volatility Helpers
=========================================================

Optional module that provides helpers for options/IV analysis when supplied
with an implied-volatility series (IV series as input).  All heavy compute
delegates to Rust via ``ferro_ta`` core; this module is a thin orchestration
layer.

.. note::
    Options support is **optional** and does not require any additional
    third-party libraries beyond ``numpy``.  For advanced option-pricing
    functionality (e.g. Black-Scholes, Greeks) install the optional
    ``ferro_ta[options]`` extra which may pull in additional dependencies.

See ``docs/options-volatility.md`` for the full design doc.

Quick start
-----------
>>> import numpy as np
>>> from ferro_ta.analysis.options import iv_rank, iv_percentile
>>>
>>> # Synthetic IV series (e.g. VIX or single-name IV)
>>> rng = np.random.default_rng(42)
>>> iv = rng.uniform(10, 40, 252)
>>>
>>> rank = iv_rank(iv, window=252)
>>> pct = iv_percentile(iv, window=252)

API
---
iv_rank(iv_series, window)
    Rolling IV rank: where is today's IV relative to min/max over *window* bars?
    Returns values in [0, 1] (NaN during warm-up).

iv_percentile(iv_series, window)
    Rolling IV percentile: fraction of observations over *window* bars that are
    ≤ today's IV.  Returns values in [0, 1] (NaN during warm-up).

iv_zscore(iv_series, window)
    Rolling IV z-score: (IV - rolling_mean) / rolling_std over *window* bars.
    Returns z-score values (NaN during warm-up).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from ferro_ta.core.exceptions import FerroTAInputError, FerroTAValueError

__all__ = [
    "iv_rank",
    "iv_percentile",
    "iv_zscore",
]


def _validate_iv(iv_series: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Validate and convert iv_series; check window."""
    arr = np.asarray(iv_series, dtype=np.float64)
    if arr.ndim != 1:
        raise FerroTAInputError("iv_series must be a 1-D array.")
    if len(arr) == 0:
        raise FerroTAInputError("iv_series must not be empty.")
    if window < 1:
        raise FerroTAValueError(f"window must be >= 1, got {window}.")
    return arr


def iv_rank(
    iv_series: ArrayLike,
    window: int = 252,
) -> NDArray[np.float64]:
    """Compute rolling IV rank.

    IV rank measures where today's IV sits relative to the min/max of IV over
    the look-back *window*.  A value of 1.0 means current IV is at its
    highest, 0.0 means it is at its lowest.

    Parameters
    ----------
    iv_series : array-like
        1-D series of implied volatility values (e.g. VIX daily closes or
        single-name option IV).  Any positive numeric values are accepted.
    window : int
        Look-back period in bars (default 252 ≈ 1 trading year).

    Returns
    -------
    ndarray of float64
        Rolling IV rank in [0, 1]. NaN for bars where the window is not yet
        full (i.e. the first ``window - 1`` bars).

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.options import iv_rank
    >>> iv = np.array([20.0, 25.0, 30.0, 15.0, 22.0])
    >>> iv_rank(iv, window=3)
    array([       nan,        nan, 1.        , 0.        , 0.46666667])
    """
    arr = _validate_iv(np.asarray(iv_series, dtype=np.float64), window)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if window > n:
        return out

    windows = sliding_window_view(arr, window_shape=window)
    lower = np.nanmin(windows, axis=1)
    upper = np.nanmax(windows, axis=1)
    current = arr[window - 1 :]
    spread = upper - lower
    out[window - 1 :] = np.where(spread == 0.0, 0.0, (current - lower) / spread)

    return out


def iv_percentile(
    iv_series: ArrayLike,
    window: int = 252,
) -> NDArray[np.float64]:
    """Compute rolling IV percentile.

    IV percentile measures the fraction of days over the look-back *window*
    for which IV was *at or below* today's level.  Unlike IV rank (which only
    considers min/max), IV percentile uses the full distribution of values.

    Parameters
    ----------
    iv_series : array-like
        1-D series of implied volatility values.
    window : int
        Look-back period in bars (default 252).

    Returns
    -------
    ndarray of float64
        Rolling IV percentile in [0, 1]. NaN for bars before the window fills.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.options import iv_percentile
    >>> iv = np.array([20.0, 25.0, 30.0, 15.0, 22.0])
    >>> iv_percentile(iv, window=3)
    array([       nan,        nan, 1.        , 0.        , 0.33333333])
    """
    arr = _validate_iv(np.asarray(iv_series, dtype=np.float64), window)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if window > n:
        return out

    windows = sliding_window_view(arr, window_shape=window)
    current = arr[window - 1 :, None]
    out[window - 1 :] = np.sum(windows <= current, axis=1, dtype=np.int64) / window

    return out


def iv_zscore(
    iv_series: ArrayLike,
    window: int = 252,
) -> NDArray[np.float64]:
    """Compute rolling IV z-score.

    Measures how many standard deviations today's IV is above (positive) or
    below (negative) the rolling mean over *window* bars.

    Parameters
    ----------
    iv_series : array-like
        1-D series of implied volatility values.
    window : int
        Look-back period in bars (default 252).

    Returns
    -------
    ndarray of float64
        Rolling z-score.  NaN during warm-up (first ``window - 1`` bars) and
        when the rolling standard deviation is zero.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.options import iv_zscore
    >>> iv = np.array([20.0, 25.0, 30.0, 15.0, 22.0])
    >>> z = iv_zscore(iv, window=3)
    >>> z[2]  # (30 - 25) / std([20, 25, 30])
    np.float64(1.2247...)
    """
    arr = _validate_iv(np.asarray(iv_series, dtype=np.float64), window)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if window > n:
        return out

    windows = sliding_window_view(arr, window_shape=window)
    mean = np.nanmean(windows, axis=1)
    std = np.nanstd(windows, axis=1, ddof=0)
    current = arr[window - 1 :]
    out[window - 1 :] = np.where(std == 0.0, np.nan, (current - mean) / std)

    return out
