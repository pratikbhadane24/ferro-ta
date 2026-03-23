"""
ferro_ta.portfolio — Portfolio and multi-asset analytics.

Compute-intensive portfolio metrics (correlation, volatility, beta, drawdown)
are implemented in Rust; this module provides the Python-facing API.

Functions
---------
correlation_matrix(returns_df_or_array)
    Compute the pairwise Pearson correlation matrix for a returns table.

portfolio_volatility(returns, weights)
    Compute portfolio volatility sqrt(w' Σ w) from a returns table and
    weights (or pass a covariance matrix directly).

beta(asset_returns, benchmark_returns, *, window=None)
    Compute beta of one asset vs a benchmark, full-sample or rolling.

drawdown(equity, *, as_series=True)
    Compute the drawdown series and max drawdown for an equity curve.

Rust backend
------------
All compute delegates to::

    ferro_ta._ferro_ta.correlation_matrix
    ferro_ta._ferro_ta.portfolio_volatility
    ferro_ta._ferro_ta.beta_full
    ferro_ta._ferro_ta.rolling_beta
    ferro_ta._ferro_ta.drawdown_series
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import beta_full as _rust_beta_full
from ferro_ta._ferro_ta import correlation_matrix as _rust_corr
from ferro_ta._ferro_ta import drawdown_series as _rust_drawdown
from ferro_ta._ferro_ta import portfolio_volatility as _rust_port_vol
from ferro_ta._ferro_ta import rolling_beta as _rust_rolling_beta
from ferro_ta._utils import _to_f64

__all__ = [
    "correlation_matrix",
    "portfolio_volatility",
    "beta",
    "drawdown",
]


# ---------------------------------------------------------------------------
# correlation_matrix
# ---------------------------------------------------------------------------


def correlation_matrix(returns: Any) -> Any:
    """Compute the pairwise Pearson correlation matrix.

    Parameters
    ----------
    returns : pandas.DataFrame or 2-D array-like, shape (n_bars, n_assets)
        Returns per bar and asset.  Assets are columns.

    Returns
    -------
    numpy.ndarray of shape (n_assets, n_assets), or pandas.DataFrame
    with same column/index names if a DataFrame was passed.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.portfolio import correlation_matrix
    >>> rng = np.random.default_rng(0)
    >>> r = rng.normal(0, 0.01, (100, 3))
    >>> corr = correlation_matrix(r)
    >>> corr.shape
    (3, 3)
    >>> abs(corr[0, 0] - 1.0) < 1e-10
    True
    """
    try:
        import pandas as pd

        if isinstance(returns, pd.DataFrame):
            cols = returns.columns.tolist()
            arr = returns.values.astype(np.float64, copy=False)
            arr = np.ascontiguousarray(arr)
            result = _rust_corr(arr)
            return pd.DataFrame(result, index=cols, columns=cols)
    except ImportError:
        pass
    arr = np.ascontiguousarray(returns, dtype=np.float64)
    return _rust_corr(arr)


# ---------------------------------------------------------------------------
# portfolio_volatility
# ---------------------------------------------------------------------------


def portfolio_volatility(
    returns: Any,
    weights: ArrayLike,
    *,
    annualise: Optional[float] = None,
) -> float:
    """Compute portfolio volatility sqrt(w' Σ w).

    Parameters
    ----------
    returns : pandas.DataFrame or 2-D array-like, shape (n_bars, n_assets)
        Returns per bar/asset.  The covariance matrix is computed from this.
    weights : array-like of length n_assets
        Portfolio weights (do not need to sum to 1).
    annualise : float, optional
        If given, the result is multiplied by ``sqrt(annualise)`` (e.g.
        ``252`` for daily returns annualised to yearly).

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.portfolio import portfolio_volatility
    >>> rng = np.random.default_rng(1)
    >>> r = rng.normal(0, 0.01, (252, 3))
    >>> vol = portfolio_volatility(r, weights=[1/3, 1/3, 1/3])
    >>> vol > 0
    True
    """
    try:
        import pandas as pd

        if isinstance(returns, pd.DataFrame):
            arr = returns.values.astype(np.float64, copy=False)
        else:
            arr = np.asarray(returns, dtype=np.float64)
    except ImportError:
        arr = np.asarray(returns, dtype=np.float64)

    arr = np.ascontiguousarray(arr)
    cov = np.cov(arr.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = np.ascontiguousarray(cov)
    w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    vol = _rust_port_vol(cov, w)
    if annualise is not None:
        vol *= float(annualise) ** 0.5
    return vol


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------


def beta(
    asset_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    *,
    window: Optional[int] = None,
) -> Union[float, NDArray[np.float64]]:
    """Compute beta of an asset vs a benchmark.

    Parameters
    ----------
    asset_returns, benchmark_returns : array-like
        Fractional returns per bar (equal length, >= 2 elements).
    window : int, optional
        If given, compute rolling beta over a sliding window of this size.
        Returns a 1-D array with ``NaN`` for the first ``window-1`` bars.
        If ``None`` (default), return the full-sample scalar beta.

    Returns
    -------
    float or numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.portfolio import beta
    >>> rng = np.random.default_rng(2)
    >>> bench = rng.normal(0, 0.01, 100)
    >>> asset = 1.2 * bench + rng.normal(0, 0.001, 100)
    >>> abs(beta(asset, bench) - 1.2) < 0.05
    True
    """
    a = _to_f64(asset_returns)
    b = _to_f64(benchmark_returns)
    if window is not None:
        return _rust_rolling_beta(a, b, int(window))
    return _rust_beta_full(a, b)


# ---------------------------------------------------------------------------
# drawdown
# ---------------------------------------------------------------------------


def drawdown(
    equity: ArrayLike,
    *,
    as_series: bool = True,
) -> Union[tuple[NDArray[np.float64], float], float]:
    """Compute the drawdown series and maximum drawdown.

    Parameters
    ----------
    equity : array-like
        Equity or price series (e.g. portfolio equity curve).
    as_series : bool
        If ``True`` (default), return ``(drawdown_array, max_drawdown)``.
        If ``False``, return only the scalar max_drawdown.

    Returns
    -------
    (numpy.ndarray, float) when *as_series* is True;
    float when *as_series* is False.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.portfolio import drawdown
    >>> eq = np.array([100.0, 110.0, 105.0, 90.0, 95.0])
    >>> dd, max_dd = drawdown(eq)
    >>> round(max_dd, 4)
    -0.1818
    """
    eq = _to_f64(equity)
    dd_arr, max_dd = _rust_drawdown(eq)
    if as_series:
        return dd_arr, max_dd
    return max_dd
