"""
Shared utility helpers for ferro_ta Python wrappers.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

# Default OHLCV column names for DataFrame contract
DEFAULT_OHLCV_COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


@functools.lru_cache(maxsize=1)
def _optional_pandas_module():
    """Import pandas lazily once and cache absence for low-overhead hot paths."""
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd


@functools.lru_cache(maxsize=1)
def _optional_polars_module():
    """Import polars lazily once and cache absence for low-overhead hot paths."""
    try:
        import polars as pl
    except ImportError:
        return None
    return pl


def _to_f64(data: ArrayLike) -> np.ndarray:
    """Convert any array-like to a contiguous 1-D float64 NumPy array.

    Transparently accepts ``pandas.Series`` and ``polars.Series`` — the values
    are extracted and the index/metadata is discarded (use :func:`pandas_wrap`
    or :func:`polars_wrap` to preserve it).

    Fast path: if *data* is already a 1-D C-contiguous ``float64`` NumPy array
    it is returned as-is without any copy or allocation.
    """
    # Fast path: already a 1-D contiguous float64 numpy array — no copy needed.
    if (
        isinstance(data, np.ndarray)
        and data.dtype == np.float64
        and data.ndim == 1
        and data.flags["C_CONTIGUOUS"]
    ):
        return data
    # Accept pandas Series/DataFrame without requiring pandas at import time
    if hasattr(data, "to_numpy"):
        try:
            data = data.to_numpy(dtype=np.float64)  # type: ignore[union-attr]
        except TypeError:
            # Some libraries (e.g. polars) have to_numpy() but don't accept dtype
            data = np.asarray(data.to_numpy(), dtype=np.float64)  # type: ignore[union-attr]
    # Accept polars Series via to_numpy() (available since polars 0.13)
    elif hasattr(data, "to_list") and type(data).__name__ == "Series":
        # polars Series doesn't have to_numpy with dtype kwarg; use cast+to_numpy
        try:
            data = data.cast(float).to_numpy()  # type: ignore[union-attr]
        except Exception:
            data = np.array(data.to_list(), dtype=np.float64)  # type: ignore[union-attr]
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1-D array or list of prices.")
    return arr


def get_ohlcv(
    df: Any,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: Optional[str] = "volume",
) -> tuple[Any, Any, Any, Any, Any]:
    """Extract OHLCV arrays or Series from a DataFrame with configurable column names.

    Use this when you have a single DataFrame with OHLCV columns (possibly with
    different names) and want to call indicators that expect separate arrays.
    Index is preserved when the input is a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least columns for open, high, low, close (and optionally volume).
    open_col, high_col, low_col, close_col, volume_col : str
        Column names to use. Defaults are ``'open'``, ``'high'``, ``'low'``,
        ``'close'``, ``'volume'``.

    Returns
    -------
    tuple of (open, high, low, close, volume)
        Each element is a 1-D array or pandas Series (same type as DataFrame columns)
        with the same index as ``df``. Missing columns raise KeyError.

    Examples
    --------
    >>> import pandas as pd
    >>> from ferro_ta import ATR, RSI
    >>> from ferro_ta._utils import get_ohlcv
    >>> df = pd.DataFrame({
    ...     'Open': [1, 2, 3], 'High': [1.1, 2.1, 3.1],
    ...     'Low': [0.9, 1.9, 2.9], 'Close': [1.05, 2.05, 3.05]
    ... })
    >>> o, h, l, c, v = get_ohlcv(df, open_col='Open', high_col='High',
    ...                            low_col='Low', close_col='Close', volume_col=None)
    >>> atr = ATR(h, l, c, timeperiod=2)  # index preserved if pandas
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("get_ohlcv requires pandas. Install with: pip install pandas")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("get_ohlcv expects a pandas.DataFrame")

    def _get(name: Optional[str]) -> Any:
        if name is None:
            return np.full(len(df), np.nan)
        if name not in df.columns:
            raise KeyError(
                f"Column '{name}' not found in DataFrame. Columns: {list(df.columns)}"
            )
        return df[name]

    vol_col = volume_col if (volume_col and volume_col in df.columns) else None
    return (
        _get(open_col),
        _get(high_col),
        _get(low_col),
        _get(close_col),
        _get(vol_col) if vol_col else np.full(len(df), np.nan),
    )


def pandas_wrap(func):
    """Decorator — transparent ``pandas.Series`` / ``DataFrame`` support.

    When at least one positional argument is a ``pandas.Series`` or
    ``pandas.DataFrame`` column, the wrapper:

    1. Extracts the NumPy arrays from all pandas inputs.
    2. Captures the index from the *first* pandas input.
    3. Calls the original function with plain NumPy arrays.
    4. Wraps every ``numpy.ndarray`` in the result back into a
       ``pandas.Series`` (or tuple of Series) with the captured index.

    If ``pandas`` is not installed the decorator is a no-op pass-through so
    the NumPy API is unaffected.

    Parameters that are already NumPy arrays (or lists) are passed through
    unchanged.  Scalar keyword arguments (e.g. ``timeperiod``) are always
    passed through unchanged.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from ferro_ta import SMA
    >>> s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = SMA(s, timeperiod=3)
    >>> isinstance(result, pd.Series)
    True
    >>> list(result.index) == list(s.index)
    True
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pd = _optional_pandas_module()
        if pd is None:
            return func(*args, **kwargs)

        pd_index = None
        new_args: list[Any] = []

        for arg in args:
            if isinstance(arg, pd.Series):
                if pd_index is None:
                    pd_index = arg.index
                new_args.append(arg.to_numpy(dtype=np.float64))
            elif isinstance(arg, pd.DataFrame):
                if pd_index is None:
                    pd_index = arg.index
                # Pass each column as a 1-D array (single-column DataFrames)
                if arg.shape[1] == 1:
                    new_args.append(arg.iloc[:, 0].to_numpy(dtype=np.float64))
                else:
                    new_args.append(arg)
            else:
                new_args.append(arg)

        result = func(*new_args, **kwargs)

        if pd_index is not None:
            if isinstance(result, tuple):
                return tuple(
                    pd.Series(r, index=pd_index) if isinstance(r, np.ndarray) else r
                    for r in result
                )
            elif isinstance(result, np.ndarray):
                return pd.Series(result, index=pd_index)

        return result

    # Mark so callers can detect wrapped functions
    wrapper._pandas_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def polars_wrap(func):
    """Decorator — transparent ``polars.Series`` support.

    When at least one positional argument is a ``polars.Series``, the wrapper:

    1. Converts all polars Series inputs to NumPy arrays.
    2. Captures the name from the *first* polars input (used as the result
       series name).
    3. Calls the original function with plain NumPy arrays.
    4. Wraps every ``numpy.ndarray`` in the result back into a
       ``polars.Series`` with the same name.

    If ``polars`` is not installed the decorator is a no-op pass-through so
    the NumPy API is unaffected.

    Parameters that are already NumPy arrays (or lists) are passed through
    unchanged.  Scalar keyword arguments (e.g. ``timeperiod``) are always
    passed through unchanged.

    Examples
    --------
    >>> import polars as pl
    >>> from ferro_ta import SMA
    >>> s = pl.Series("close", [1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = SMA(s, timeperiod=3)
    >>> isinstance(result, pl.Series)
    True
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pl = _optional_polars_module()
        if pl is None:
            return func(*args, **kwargs)

        pl_name: Optional[str] = None
        new_args: list[Any] = []

        for arg in args:
            if isinstance(arg, pl.Series):
                if pl_name is None:
                    pl_name = arg.name
                try:
                    new_args.append(arg.cast(pl.Float64).to_numpy())
                except Exception:
                    new_args.append(np.array(arg.to_list(), dtype=np.float64))
            else:
                new_args.append(arg)

        result = func(*new_args, **kwargs)

        if pl_name is not None:
            if isinstance(result, tuple):
                return tuple(
                    pl.Series(pl_name, r) if isinstance(r, np.ndarray) else r
                    for r in result
                )
            elif isinstance(result, np.ndarray):
                return pl.Series(pl_name, result)

        return result

    wrapper._polars_wrapped = True  # type: ignore[attr-defined]
    return wrapper
