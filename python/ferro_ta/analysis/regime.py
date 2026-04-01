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


# ---------------------------------------------------------------------------
# Phase 4: Volatility/Trend regime detection (pure NumPy)
# ---------------------------------------------------------------------------


try:
    from ferro_ta._ferro_ta import sma as _rust_sma
except ImportError:
    _rust_sma = None


def _rolling_sma_pure(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling SMA — delegates to the Rust SMA when available."""
    if _rust_sma is not None:
        return np.asarray(_rust_sma(arr, window), dtype=np.float64)
    # Fallback: O(n) rolling SMA using cumsum
    n = len(arr)
    out = np.full(n, np.nan)
    if window > n:
        return out
    cs = np.cumsum(arr)
    out[window - 1] = cs[window - 1] / window
    if window < n:
        out[window:] = (cs[window:] - cs[: n - window]) / window
    return out


def _rolling_std_pure(arr: np.ndarray, window: int) -> np.ndarray:
    """O(n) rolling std using cumsum-of-squares on the valid (non-NaN) portion.

    Handles leading NaN values (e.g., log returns where arr[0] is NaN).
    NaN is returned for warm-up bars.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    if window < 2 or window > n:
        return out

    # Find the first non-NaN index
    first_valid = 0
    while first_valid < n and np.isnan(arr[first_valid]):
        first_valid += 1

    if first_valid >= n:
        return out  # all NaN

    # Work on the valid slice
    valid_slice = arr[first_valid:]
    m = len(valid_slice)
    if window > m:
        return out

    cs = np.cumsum(valid_slice)
    cs2 = np.cumsum(valid_slice**2)

    n_windows = m - window + 1
    s = np.empty(n_windows)
    s2 = np.empty(n_windows)
    s[0] = cs[window - 1]
    s2[0] = cs2[window - 1]
    if n_windows > 1:
        s[1:] = cs[window:] - cs[: m - window]
        s2[1:] = cs2[window:] - cs2[: m - window]

    mean = s / window
    var = np.maximum(s2 / window - mean**2, 0.0)
    stds = np.sqrt(var)

    # Place back into output (first result is at index first_valid + window - 1)
    start_out = first_valid + window - 1
    out[start_out : start_out + n_windows] = stds
    return out


def detect_volatility_regime(
    close: ArrayLike,
    window: int = 20,
    n_regimes: int = 3,
) -> NDArray:
    """Label bars by rolling volatility percentile bucket (0 = lowest vol regime).

    Uses rolling standard deviation of log returns. NaN for warm-up bars
    (returned as -1 in the integer output).

    Parameters
    ----------
    close : array-like
        Close price series.
    window : int
        Rolling window for std computation (default 20).
    n_regimes : int
        Number of volatility regimes (default 3: low/mid/high = 0/1/2).

    Returns
    -------
    NDArray[int64]
        Integer array where each element is in {-1, 0, ..., n_regimes-1}.
        -1 indicates NaN (warm-up) bars.
    """
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    out = np.full(n, -1, dtype=np.int64)

    log_ret = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(c[1:] / c[:-1])

    rolling_vol = _rolling_std_pure(log_ret, window)

    valid = ~np.isnan(rolling_vol)
    if not np.any(valid):
        return out

    vol_vals = rolling_vol[valid]
    pcts = [100.0 * k / n_regimes for k in range(1, n_regimes)]
    boundaries = np.percentile(vol_vals, pcts) if pcts else np.array([])

    labels = np.digitize(vol_vals, boundaries).astype(np.int64)

    out[valid] = labels
    return out


def detect_trend_regime(
    close: ArrayLike,
    fast: int = 50,
    slow: int = 200,
) -> NDArray:
    """Label bars: 1=bull (fast SMA > slow SMA), -1=bear, 0=sideways/NaN warmup.

    Parameters
    ----------
    close : array-like
        Close price series.
    fast : int
        Fast SMA period (default 50).
    slow : int
        Slow SMA period (default 200).

    Returns
    -------
    NDArray[int64]
        Integer array with values in {-1, 0, 1}.
        0 for warm-up bars where either SMA is NaN.
    """
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    out = np.zeros(n, dtype=np.int64)

    fast_sma = _rolling_sma_pure(c, fast)
    slow_sma = _rolling_sma_pure(c, slow)

    valid = ~np.isnan(fast_sma) & ~np.isnan(slow_sma)
    out[valid & (fast_sma > slow_sma)] = 1
    out[valid & (fast_sma < slow_sma)] = -1
    return out


def detect_combined_regime(
    close: ArrayLike,
    vol_window: int = 20,
    fast: int = 50,
    slow: int = 200,
) -> NDArray:
    """Combine trend + vol into 6-state integer regime label.

    States: 0=bull+low-vol, 1=bull+mid-vol, 2=bull+high-vol,
            3=bear+low-vol, 4=bear+mid-vol, 5=bear+high-vol.
    NaN bars (warm-up or sideways) → -1.

    Parameters
    ----------
    close : array-like
        Close price series.
    vol_window : int
        Rolling window for volatility regime detection.
    fast, slow : int
        SMA periods for trend regime detection.

    Returns
    -------
    NDArray[int64]
        Integer array with values in {-1, 0, 1, 2, 3, 4, 5}.
    """
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    out = np.full(n, -1, dtype=np.int64)

    trend = detect_trend_regime(c, fast=fast, slow=slow)
    vol = detect_volatility_regime(c, window=vol_window, n_regimes=3)

    bull_valid = (trend == 1) & (vol >= 0)
    bear_valid = (trend == -1) & (vol >= 0)

    out[bull_valid] = vol[bull_valid]  # 0, 1, or 2
    out[bear_valid] = 3 + vol[bear_valid]  # 3, 4, or 5

    return out


class RegimeFilter:
    """Filter trading signals to only fire in allowed market regimes.

    Parameters
    ----------
    allowed_regimes : list[int]
        Which regime labels to trade in. Signals in other regimes are zeroed out.
    vol_window : int
        Rolling window for volatility regime detection.
    fast, slow : int
        SMA periods for trend regime detection.
    """

    def __init__(
        self,
        allowed_regimes: list[int],
        vol_window: int = 20,
        fast: int = 50,
        slow: int = 200,
    ) -> None:
        self.allowed_regimes = list(allowed_regimes)
        self._allowed_regimes_arr = np.array(allowed_regimes, dtype=np.int64)
        self.vol_window = int(vol_window)
        self.fast = int(fast)
        self.slow = int(slow)

    def filter(self, signals: ArrayLike, close: ArrayLike) -> NDArray:
        """Zero out signals where regime is not in allowed_regimes.

        Parameters
        ----------
        signals : array-like
            Signal array (+1, -1, 0, or NaN).
        close : array-like
            Close price series (same length as signals).

        Returns
        -------
        NDArray[float64]
            Filtered signal array — signals in disallowed regimes are set to 0.
        """
        s = np.asarray(signals, dtype=np.float64).copy()
        regimes = detect_combined_regime(
            close,
            vol_window=self.vol_window,
            fast=self.fast,
            slow=self.slow,
        )
        in_allowed = np.isin(regimes, self._allowed_regimes_arr)
        s[~in_allowed] = 0.0
        return s


# ---------------------------------------------------------------------------
# (original structural_breaks below)
# ---------------------------------------------------------------------------


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
