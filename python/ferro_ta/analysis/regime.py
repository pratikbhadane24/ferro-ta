"""
ferro_ta.regime — Regime detection and structural breaks.
=========================================================

Detect market regimes (trending vs ranging) and structural breaks in price or
indicator series using existing ferro-ta indicators plus rule-based methods.

Functions
---------
regime(ohlcv, method='adx', **kwargs)
    Label each bar as trending (1), ranging (0), or warm-up (-1).
    Supported methods: ``'adx'``, ``'combined'``.

structural_breaks(series, method='cusum', **kwargs)
    Detect structural breaks.  Returns a binary mask (1 = break).
    Supported methods: ``'cusum'``, ``'variance'``.

regime_adx(adx, threshold=25.0)
    Low-level: label bars using an ADX array directly.

regime_combined(adx, atr, close, adx_threshold=25.0, atr_pct_threshold=0.005)
    Low-level: ADX + ATR-ratio labelling.

detect_breaks_cusum(series, window=20, threshold=3.0, slack=0.5)
    Low-level: CUSUM-based structural break detection.

rolling_variance_break(series, short_window=10, long_window=50, threshold=2.0)
    Low-level: rolling variance ratio break detection.

Rust backend
------------
    ferro_ta._ferro_ta.regime_adx
    ferro_ta._ferro_ta.regime_combined
    ferro_ta._ferro_ta.detect_breaks_cusum
    ferro_ta._ferro_ta.rolling_variance_break
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import (
    detect_breaks_cusum as _rust_detect_breaks_cusum,
)
from ferro_ta._ferro_ta import (
    regime_adx as _rust_regime_adx,
)
from ferro_ta._ferro_ta import (
    regime_combined as _rust_regime_combined,
)
from ferro_ta._ferro_ta import (
    rolling_variance_break as _rust_rolling_variance_break,
)
from ferro_ta._utils import _to_f64

__all__ = [
    "regime",
    "structural_breaks",
    "regime_adx",
    "regime_combined",
    "detect_breaks_cusum",
    "rolling_variance_break",
]

# type alias for OHLCV tuple
OHLCVTuple = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]


# ---------------------------------------------------------------------------
# Low-level wrappers
# ---------------------------------------------------------------------------


def regime_adx(
    adx: ArrayLike,
    threshold: float = 25.0,
) -> NDArray[np.int8]:
    """Label each bar as trend (1), range (0), or warm-up (-1) using ADX.

    Parameters
    ----------
    adx       : array-like — ADX values (NaN during warm-up)
    threshold : float — ADX level above which a bar is "trending" (default 25)

    Returns
    -------
    numpy.ndarray of int8 — ``1`` trend, ``0`` range, ``-1`` warm-up (NaN)
    """
    return np.asarray(
        _rust_regime_adx(_to_f64(adx), float(threshold)),
        dtype=np.int8,
    )


def regime_combined(
    adx: ArrayLike,
    atr: ArrayLike,
    close: ArrayLike,
    adx_threshold: float = 25.0,
    atr_pct_threshold: float = 0.005,
) -> NDArray[np.int8]:
    """Label bars using ADX + ATR-as-%-of-close rule.

    A bar is "trending" when both:
    - ``adx[i] > adx_threshold``
    - ``atr[i] / close[i] > atr_pct_threshold``

    Parameters
    ----------
    adx               : array-like — ADX values
    atr               : array-like — ATR values
    close             : array-like — close prices
    adx_threshold     : float — ADX threshold (default 25.0)
    atr_pct_threshold : float — minimum ATR/close ratio (default 0.005)

    Returns
    -------
    numpy.ndarray of int8 — ``1`` trend, ``0`` range, ``-1`` NaN
    """
    return np.asarray(
        _rust_regime_combined(
            _to_f64(adx),
            _to_f64(atr),
            _to_f64(close),
            float(adx_threshold),
            float(atr_pct_threshold),
        ),
        dtype=np.int8,
    )


def detect_breaks_cusum(
    series: ArrayLike,
    window: int = 20,
    threshold: float = 3.0,
    slack: float = 0.5,
) -> NDArray[np.int8]:
    """Detect structural breaks using CUSUM (cumulative sum) approach.

    Parameters
    ----------
    series    : array-like — price or indicator series to monitor
    window    : int — lookback window for mean/std estimation (>= 2, default 20)
    threshold : float — CUSUM threshold in units of std (default 3.0)
    slack     : float — allowance term (default 0.5)

    Returns
    -------
    numpy.ndarray of int8 — ``1`` at break bars, ``0`` elsewhere
    """
    return np.asarray(
        _rust_detect_breaks_cusum(
            _to_f64(series),
            int(window),
            float(threshold),
            float(slack),
        ),
        dtype=np.int8,
    )


def rolling_variance_break(
    series: ArrayLike,
    short_window: int = 10,
    long_window: int = 50,
    threshold: float = 2.0,
) -> NDArray[np.int8]:
    """Detect volatility regime breaks using a rolling variance ratio test.

    Parameters
    ----------
    series       : array-like — returns or price series
    short_window : int — recent variance lookback (>= 2, default 10)
    long_window  : int — baseline variance lookback (> short_window, default 50)
    threshold    : float — ratio short_var/long_var above which a break fires
        (default 2.0)

    Returns
    -------
    numpy.ndarray of int8 — ``1`` at break bars, ``0`` elsewhere
    """
    return np.asarray(
        _rust_rolling_variance_break(
            _to_f64(series),
            int(short_window),
            int(long_window),
            float(threshold),
        ),
        dtype=np.int8,
    )


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def regime(
    ohlcv: Union[OHLCVTuple, object],  # also accepts pandas.DataFrame
    method: str = "adx",
    adx_threshold: float = 25.0,
    atr_pct_threshold: float = 0.005,
    adx_timeperiod: int = 14,
    atr_timeperiod: int = 14,
) -> NDArray[np.int8]:
    """Label each bar as trending (1) or ranging (0) using existing indicators.

    Parameters
    ----------
    ohlcv : tuple ``(open, high, low, close, volume)`` or pandas DataFrame
    method : str
        - ``'adx'`` (default) — uses ADX > *adx_threshold*
        - ``'combined'`` — uses ADX + ATR/close ratio
    adx_threshold     : float — ADX level threshold (default 25.0)
    atr_pct_threshold : float — minimum ATR/close ratio for ``'combined'``
        (default 0.005 = 0.5%)
    adx_timeperiod : int — ADX period (default 14)
    atr_timeperiod : int — ATR period for combined method (default 14)

    Returns
    -------
    numpy.ndarray of int8 — ``1`` trend, ``0`` range, ``-1`` warm-up

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.regime import regime
    >>> rng = np.random.default_rng(1)
    >>> n = 200
    >>> close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    >>> open_ = close * rng.uniform(0.998, 1.002, n)
    >>> high = np.maximum(close, open_) + rng.uniform(0, 0.5, n)
    >>> low = np.minimum(close, open_) - rng.uniform(0, 0.5, n)
    >>> vol = rng.uniform(1000, 5000, n)
    >>> labels = regime((open_, high, low, close, vol))
    >>> # Count trending bars (excluding warm-up)
    >>> valid = labels[labels >= 0]
    >>> trend_pct = (valid == 1).sum() / len(valid)
    """
    from ferro_ta import ADX, ATR  # local import to avoid circular dependency

    try:
        import pandas as pd

        if isinstance(ohlcv, pd.DataFrame):
            cols = {c.lower(): c for c in ohlcv.columns}  # type: ignore[union-attr]
            high_arr = _to_f64(ohlcv[cols["high"]].values)  # type: ignore[index]
            low_arr = _to_f64(ohlcv[cols["low"]].values)  # type: ignore[index]
            close_arr = _to_f64(ohlcv[cols["close"]].values)  # type: ignore[index]
        else:
            _, high_arr, low_arr, close_arr, _ = [_to_f64(x) for x in ohlcv]  # type: ignore[union-attr]
    except ImportError:
        _, high_arr, low_arr, close_arr, _ = [_to_f64(x) for x in ohlcv]  # type: ignore[union-attr]

    adx_vals = np.asarray(
        ADX(high_arr, low_arr, close_arr, timeperiod=adx_timeperiod), dtype=np.float64
    )

    if method == "adx":
        return regime_adx(adx_vals, threshold=adx_threshold)
    elif method == "combined":
        atr_vals = np.asarray(
            ATR(high_arr, low_arr, close_arr, timeperiod=atr_timeperiod),
            dtype=np.float64,
        )
        return regime_combined(
            adx_vals,
            atr_vals,
            close_arr,
            adx_threshold=adx_threshold,
            atr_pct_threshold=atr_pct_threshold,
        )
    else:
        raise ValueError(f"Unknown regime method '{method}'. Use 'adx' or 'combined'.")


def structural_breaks(
    series: ArrayLike,
    method: str = "cusum",
    window: int = 20,
    threshold: float = 3.0,
    slack: float = 0.5,
    short_window: int = 10,
    long_window: int = 50,
    variance_threshold: float = 2.0,
) -> NDArray[np.int8]:
    """Detect structural breaks in a series.

    Parameters
    ----------
    series   : array-like — price or returns series to monitor
    method   : str
        - ``'cusum'`` (default) — CUSUM-based break detection
        - ``'variance'`` — rolling variance ratio break detection
    window   : int — CUSUM lookback window (default 20)
    threshold: float — CUSUM threshold in std units (default 3.0)
    slack    : float — CUSUM slack term (default 0.5)
    short_window    : int — short variance window for ``'variance'`` (default 10)
    long_window     : int — long variance window for ``'variance'`` (default 50)
    variance_threshold : float — variance ratio threshold (default 2.0)

    Returns
    -------
    numpy.ndarray of int8 — ``1`` at break bars, ``0`` elsewhere

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.regime import structural_breaks
    >>> rng = np.random.default_rng(42)
    >>> # Create a series with a structural break in the middle
    >>> s1 = rng.normal(0, 1, 100)
    >>> s2 = rng.normal(5, 3, 100)   # different mean/variance
    >>> series = np.concatenate([s1, s2])
    >>> breaks = structural_breaks(series, method='cusum')
    >>> int(breaks[100:115].any())  # break near index 100
    1
    """
    if method == "cusum":
        return detect_breaks_cusum(
            series, window=window, threshold=threshold, slack=slack
        )
    elif method == "variance":
        return rolling_variance_break(
            series,
            short_window=short_window,
            long_window=long_window,
            threshold=variance_threshold,
        )
    else:
        raise ValueError(
            f"Unknown structural_breaks method '{method}'. Use 'cusum' or 'variance'."
        )
