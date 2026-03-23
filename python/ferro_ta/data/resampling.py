"""
ferro_ta.resampling — OHLCV resampling and multi-timeframe API.

Provides functions to resample OHLCV data into coarser time bars or volume
bars, and a multi-timeframe helper that runs an indicator on two or more
resampled timeframes in one call.

The heavy OHLCV aggregation logic lives in the Rust backend
(``_ferro_ta.volume_bars`` and ``_ferro_ta.ohlcv_agg``); this module provides
the Python-facing API with:
- Time-based resampling via pandas (requires ``pandas``).
- Volume-bar resampling via Rust (no extra dependencies).
- Multi-timeframe helper that returns a dict of DataFrames.

Functions
---------
resample(ohlcv, rule, *, label='right', closed='right')
    Resample a pandas OHLCV DataFrame by a time rule (e.g. ``'5min'``,
    ``'1h'``).  Requires pandas.

volume_bars(ohlcv, volume_threshold)
    Aggregate OHLCV data into volume bars using the Rust backend.
    Accepts a pandas DataFrame or separate numpy arrays.

multi_timeframe(ohlcv, rules, *, indicator=None, indicator_kwargs=None)
    Resample OHLCV to multiple timeframes and optionally run an indicator
    on each.  Returns a dict mapping each rule to a DataFrame (or to an
    indicator result when *indicator* is given).

Rust backend
------------
All bar-accumulation logic delegates to::

    ferro_ta._ferro_ta.volume_bars
    ferro_ta._ferro_ta.ohlcv_agg
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from ferro_ta._ferro_ta import volume_bars as _rust_volume_bars
from ferro_ta._utils import _to_f64

__all__ = [
    "resample",
    "volume_bars",
    "multi_timeframe",
]


# ---------------------------------------------------------------------------
# resample — time-based resampling (pandas required)
# ---------------------------------------------------------------------------


def resample(
    ohlcv: Any,
    rule: str,
    *,
    label: str = "right",
    closed: str = "right",
) -> Any:
    """Resample an OHLCV DataFrame to a coarser time rule.

    Uses ``pandas.DataFrame.resample`` under the hood; the index must be a
    ``DatetimeIndex`` (timezone-aware or naive).

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Must have columns ``open``, ``high``, ``low``, ``close``, ``volume``
        (case-sensitive; use the column-name helpers in :mod:`ferro_ta._utils`
        if your column names differ).  Index must be a ``DatetimeIndex``.
    rule : str
        Pandas offset alias (e.g. ``'5min'``, ``'1h'``, ``'1D'``).
    label : str
        Which bin edge to label the bucket with (``'left'`` or ``'right'``).
        Default ``'right'``.
    closed : str
        Which side of the interval is closed (``'left'`` or ``'right'``).
        Default ``'right'``.

    Returns
    -------
    pandas.DataFrame
        Resampled OHLCV DataFrame with the same column names.

    Raises
    ------
    ImportError
        If pandas is not installed.
    ValueError
        If required columns are missing or the index is not a DatetimeIndex.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from ferro_ta.data.resampling import resample
    >>> idx = pd.date_range("2024-01-01", periods=60, freq="1min")
    >>> df = pd.DataFrame({
    ...     "open": np.random.rand(60) + 100,
    ...     "high": np.random.rand(60) + 101,
    ...     "low":  np.random.rand(60) + 99,
    ...     "close": np.random.rand(60) + 100,
    ...     "volume": np.random.randint(100, 1000, 60).astype(float),
    ... }, index=idx)
    >>> df5 = resample(df, "5min")
    >>> df5.shape[0]
    12
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for time-based resampling.  "
            "Install it with: pip install pandas"
        ) from exc

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError(
            "ohlcv.index must be a pandas DatetimeIndex for time-based resampling."
        )

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return ohlcv.resample(rule, label=label, closed=closed).agg(agg).dropna(how="all")


# ---------------------------------------------------------------------------
# volume_bars — volume-based resampling (Rust backend)
# ---------------------------------------------------------------------------


def volume_bars(
    ohlcv: Any,
    volume_threshold: float,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> Any:
    """Aggregate OHLCV data into volume bars using the Rust backend.

    Each output bar accumulates input bars until ``volume_threshold`` units of
    volume have been consumed.

    Parameters
    ----------
    ohlcv : pandas.DataFrame or tuple of arrays
        Either a pandas DataFrame with OHLCV columns, or a tuple
        ``(open, high, low, close, volume)`` of array-like objects.
    volume_threshold : float
        Target volume per output bar (must be > 0).
    open_col, high_col, low_col, close_col, volume_col : str
        Column names when ``ohlcv`` is a DataFrame.

    Returns
    -------
    pandas.DataFrame or tuple of numpy arrays
        If a DataFrame was passed in, returns a DataFrame with the same column
        names.  Otherwise returns a tuple
        ``(open, high, low, close, volume)`` of numpy arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.data.resampling import volume_bars
    >>> n = 100
    >>> o = np.random.rand(n) + 100
    >>> h = o + np.random.rand(n)
    >>> l = o - np.random.rand(n)
    >>> c = np.random.rand(n) + 100
    >>> v = np.random.randint(50, 150, n).astype(float)
    >>> bars = volume_bars((o, h, l, c, v), volume_threshold=500)
    >>> len(bars[0]) > 0
    True
    """
    if isinstance(ohlcv, tuple):
        o, h, low, c, v = (_to_f64(x) for x in ohlcv)
        return _rust_volume_bars(o, h, low, c, v, float(volume_threshold))

    # pandas DataFrame path
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required when passing a DataFrame") from exc

    o = _to_f64(ohlcv[open_col].values)
    h = _to_f64(ohlcv[high_col].values)
    low = _to_f64(ohlcv[low_col].values)
    c = _to_f64(ohlcv[close_col].values)
    v = _to_f64(ohlcv[volume_col].values)
    ro, rh, rl, rc, rv = _rust_volume_bars(o, h, low, c, v, float(volume_threshold))
    return pd.DataFrame(
        {
            open_col: ro,
            high_col: rh,
            low_col: rl,
            close_col: rc,
            volume_col: rv,
        }
    )


# ---------------------------------------------------------------------------
# multi_timeframe — run indicator on multiple resampled timeframes
# ---------------------------------------------------------------------------


def multi_timeframe(
    ohlcv: Any,
    rules: list[str],
    *,
    indicator: Optional[Callable[..., Any]] = None,
    indicator_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Resample OHLCV to multiple timeframes and optionally run an indicator.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        OHLCV data with a ``DatetimeIndex``.
    rules : list of str
        Pandas offset aliases, e.g. ``['5min', '1h']``.
    indicator : callable, optional
        A function ``indicator(close, **kwargs) -> array`` (or multi-output).
        When provided it is called on the resampled ``close`` column for each
        rule, and the result is stored in the returned dict instead of the
        full DataFrame.
    indicator_kwargs : dict, optional
        Keyword arguments forwarded to *indicator*.

    Returns
    -------
    dict
        Mapping from each rule string to:
        - a resampled pandas DataFrame when *indicator* is ``None``, or
        - the indicator output (numpy array or tuple) when *indicator* is given.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from ferro_ta import RSI
    >>> from ferro_ta.data.resampling import multi_timeframe
    >>> idx = pd.date_range("2024-01-01", periods=200, freq="1min")
    >>> close = np.cumprod(1 + np.random.randn(200) * 0.001) * 100
    >>> df = pd.DataFrame({
    ...     "open": close, "high": close * 1.001, "low": close * 0.999,
    ...     "close": close, "volume": np.ones(200) * 1000,
    ... }, index=idx)
    >>> result = multi_timeframe(df, ["5min", "15min"], indicator=RSI,
    ...                          indicator_kwargs={"timeperiod": 14})
    >>> sorted(result.keys())
    ['15min', '5min']
    """
    kw = indicator_kwargs or {}
    out: dict[str, Any] = {}
    for rule in rules:
        df_r = resample(ohlcv, rule)
        if indicator is not None:
            out[rule] = indicator(_to_f64(df_r["close"].values), **kw)
        else:
            out[rule] = df_r
    return out
