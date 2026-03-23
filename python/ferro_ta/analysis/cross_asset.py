"""
ferro_ta.cross_asset — Cross-asset and relative strength analytics.

Provides helpers for relative value and pair-trading workflows:
- relative_strength(asset_returns, benchmark_returns)
- spread(a, b, hedge=1.0)
- ratio(a, b)
- zscore(x, window)
- rolling_beta(a, b, window)

Compute-intensive work delegates to Rust (via ferro_ta._ferro_ta).

Functions
---------
relative_strength(asset_returns, benchmark_returns)
    Cumulative-return ratio (asset / benchmark), starting at 1.

spread(a, b, hedge=1.0)
    Spread series: a - hedge * b.

ratio(a, b)
    Ratio series: a / b.

zscore(x, window)
    Rolling Z-score of series *x* over a sliding window.

rolling_beta(a, b, window)
    Rolling beta (hedge ratio) of series *a* vs *b*.

Rust backend
------------
    ferro_ta._ferro_ta.relative_strength
    ferro_ta._ferro_ta.spread
    ferro_ta._ferro_ta.zscore_series
    ferro_ta._ferro_ta.rolling_beta
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import relative_strength as _rust_rel_strength
from ferro_ta._ferro_ta import rolling_beta as _rust_rolling_beta
from ferro_ta._ferro_ta import spread as _rust_spread
from ferro_ta._ferro_ta import zscore_series as _rust_zscore
from ferro_ta._utils import _to_f64

__all__ = [
    "relative_strength",
    "spread",
    "ratio",
    "zscore",
    "rolling_beta",
]


# ---------------------------------------------------------------------------
# relative_strength
# ---------------------------------------------------------------------------


def relative_strength(
    asset_returns: ArrayLike,
    benchmark_returns: ArrayLike,
) -> NDArray[np.float64]:
    """Compute relative strength of an asset versus a benchmark.

    Returns the ratio of cumulative returns::

        RS[i] = (1 + r_asset[0]) * … * (1 + r_asset[i]) /
                ((1 + r_bench[0]) * … * (1 + r_bench[i]))

    starting from RS[0] ≈ 1.

    Parameters
    ----------
    asset_returns, benchmark_returns : array-like
        Fractional returns per bar (e.g. 0.01 for +1%).  Equal length.

    Returns
    -------
    numpy.ndarray of same length — relative strength series.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.cross_asset import relative_strength
    >>> r_a = np.array([0.01, 0.02, -0.01, 0.005])
    >>> r_b = np.array([0.005, 0.01, -0.005, 0.002])
    >>> rs = relative_strength(r_a, r_b)
    >>> rs[0] > 1  # asset outperformed at bar 0
    True
    """
    a = _to_f64(asset_returns)
    b = _to_f64(benchmark_returns)
    return _rust_rel_strength(a, b)


# ---------------------------------------------------------------------------
# spread
# ---------------------------------------------------------------------------


def spread(
    a: ArrayLike,
    b: ArrayLike,
    hedge: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the spread between two series.

    ``spread[i] = a[i] - hedge * b[i]``

    Parameters
    ----------
    a, b : array-like  (equal length)
    hedge : float  — hedge ratio (default 1.0)

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.cross_asset import spread
    >>> a = np.array([10.0, 11.0, 12.0])
    >>> b = np.array([9.0, 10.0, 11.0])
    >>> list(spread(a, b))
    [1.0, 1.0, 1.0]
    """
    return _rust_spread(_to_f64(a), _to_f64(b), float(hedge))


# ---------------------------------------------------------------------------
# ratio
# ---------------------------------------------------------------------------


def ratio(
    a: ArrayLike,
    b: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the ratio of two series: a / b.

    Zeros in *b* produce ``NaN`` in the result.

    Parameters
    ----------
    a, b : array-like  (equal length)

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.cross_asset import ratio
    >>> a = np.array([10.0, 12.0, 15.0])
    >>> b = np.array([5.0, 4.0, 5.0])
    >>> list(ratio(a, b))
    [2.0, 3.0, 3.0]
    """
    av = _to_f64(a)
    bv = _to_f64(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(bv == 0, np.nan, av / bv)
    return result


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------


def zscore(
    x: ArrayLike,
    window: int,
) -> NDArray[np.float64]:
    """Compute the rolling Z-score of series *x*.

    ``z[i] = (x[i] - mean(x[i-window+1..i])) / std(x[i-window+1..i])``

    Parameters
    ----------
    x : array-like
    window : int  — must be >= 2

    Returns
    -------
    numpy.ndarray — NaN for first ``window-1`` positions.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.cross_asset import zscore
    >>> x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    >>> z = zscore(x, window=3)
    >>> np.isnan(z[0]) and np.isnan(z[1])
    True
    """
    return _rust_zscore(_to_f64(x), int(window))


# ---------------------------------------------------------------------------
# rolling_beta
# ---------------------------------------------------------------------------


def rolling_beta(
    a: ArrayLike,
    b: ArrayLike,
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling beta (hedge ratio) of series *a* vs *b*.

    Parameters
    ----------
    a, b : array-like  (equal length)
    window : int  — rolling window size (must be >= 2)

    Returns
    -------
    numpy.ndarray — NaN for first ``window-1`` positions.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.cross_asset import rolling_beta
    >>> rng = np.random.default_rng(42)
    >>> b = rng.normal(0, 1, 50)
    >>> a = 0.8 * b + rng.normal(0, 0.1, 50)
    >>> rb = rolling_beta(a, b, window=20)
    >>> np.isnan(rb[18])
    True
    >>> abs(rb[-1] - 0.8) < 0.3
    True
    """
    return _rust_rolling_beta(_to_f64(a), _to_f64(b), int(window))
