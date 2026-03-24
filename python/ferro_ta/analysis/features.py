"""
ferro_ta.features — Feature matrix and ML readiness.

Exports a feature matrix (indicators as columns, bars as rows) suitable for
sklearn or other ML pipelines.

Functions
---------
feature_matrix(ohlcv, indicators, *, nan_policy='keep', close_col='close', ...)
    Compute all requested indicators on the OHLCV data and return a single
    DataFrame with bars as rows and indicator names as columns.

Rust backend
------------
Individual indicator calls delegate to existing Rust-backed ferro_ta functions
via the registry.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ferro_ta._ferro_ta import forward_fill_nan as _rust_forward_fill_nan
from ferro_ta._utils import _to_f64
from ferro_ta.data.batch import compute_many

__all__ = [
    "feature_matrix",
]


def _forward_fill_nan(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        _rust_forward_fill_nan(np.ascontiguousarray(arr, dtype=np.float64))
    )


# ---------------------------------------------------------------------------
# feature_matrix
# ---------------------------------------------------------------------------


def feature_matrix(
    ohlcv: Any,
    indicators: list[Union[str, tuple[str, dict[str, Any]]]],
    *,
    nan_policy: str = "keep",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    volume_col: str = "volume",
) -> Any:
    """Compute multiple indicators on OHLCV data and return a feature matrix.

    Parameters
    ----------
    ohlcv : pandas.DataFrame or dict of arrays
        OHLCV data.  Must contain at least a ``close`` column/key.
    indicators : list of (str | tuple)
        Each element is either:
        - A string indicator name (e.g. ``'RSI'``), using default params.
        - A ``(name, kwargs)`` tuple, e.g. ``('RSI', {'timeperiod': 14})``.
        - A ``(name, kwargs, output_key)`` 3-tuple to name a specific output
          of a multi-output indicator (0-indexed int or output key).

        The column name in the output matrix is ``<name>`` for single-output
        indicators or ``<name>_<output_key>`` for multi-output ones.

    nan_policy : str
        How to handle NaN values (warmup rows):
        - ``'keep'`` (default) — keep NaN rows as-is.
        - ``'drop'``  — drop any row that contains at least one NaN.
        - ``'fill'``  — forward-fill NaN values.

    close_col, high_col, low_col, open_col, volume_col : str
        Column names when *ohlcv* is a DataFrame.

    Returns
    -------
    pandas.DataFrame or dict of numpy arrays
        If pandas is available, returns a DataFrame with one column per
        indicator.  Otherwise returns a dict {name: array}.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.features import feature_matrix
    >>> rng = np.random.default_rng(0)
    >>> n = 50
    >>> close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    >>> ohlcv = {"close": close, "high": close * 1.01, "low": close * 0.99,
    ...          "open": close, "volume": np.ones(n) * 1000}
    >>> fm = feature_matrix(ohlcv, [("SMA", {"timeperiod": 10}),
    ...                              ("RSI", {"timeperiod": 14})])
    >>> list(fm.keys())
    ['SMA', 'RSI']
    """

    # --- Extract arrays ---
    def _get(col: str) -> Optional[NDArray[np.float64]]:
        try:
            import pandas as pd

            if isinstance(ohlcv, pd.DataFrame):
                return _to_f64(ohlcv[col].to_numpy()) if col in ohlcv.columns else None
        except ImportError:
            pass
        if isinstance(ohlcv, dict):
            return _to_f64(ohlcv[col]) if col in ohlcv else None
        return None

    close = _get(close_col)
    high = _get(high_col)
    low = _get(low_col)
    _open = _get(open_col)  # noqa: F841 - reserved for future OHLCV indicators
    volume = _get(volume_col)

    if close is None:
        raise ValueError(f"close column '{close_col}' not found in ohlcv")

    n = len(close)
    columns: dict[str, NDArray[np.float64]] = {}

    results = compute_many(
        indicators,
        close=close,
        high=high if high is not None else None,
        low=low if low is not None else None,
        volume=volume if volume is not None else None,
    )

    for spec, result in zip(indicators, results):
        if isinstance(spec, str):
            name = spec
            out_key: Optional[Any] = None
        elif len(spec) == 2:
            name, _ = spec  # type: ignore[misc]
            out_key = None
        else:
            name, _, out_key = spec  # type: ignore[misc]

        if isinstance(result, tuple):
            if out_key is not None:
                if isinstance(out_key, int):
                    col_name = f"{name}_{out_key}"
                    columns[col_name] = np.asarray(result[out_key], dtype=np.float64)
                else:
                    col_name = f"{name}_{out_key}"
                    columns[col_name] = np.asarray(
                        result[int(out_key)], dtype=np.float64
                    )
            else:
                for ki, arr in enumerate(result):
                    columns[f"{name}_{ki}"] = np.asarray(arr, dtype=np.float64)
        else:
            columns[name] = np.asarray(result, dtype=np.float64)

    # --- NaN policy ---
    try:
        import pandas as pd

        index = None
        if isinstance(ohlcv, pd.DataFrame):
            index = ohlcv.index
        df = pd.DataFrame(columns, index=index)
        if nan_policy == "drop":
            df = df.dropna()
        elif nan_policy == "fill":
            df = df.ffill()
        return df
    except ImportError:
        if nan_policy == "drop":
            mask = np.ones(n, dtype=bool)
            for arr in columns.values():
                mask &= ~np.isnan(arr)
            return {k: v[mask] for k, v in columns.items()}
        elif nan_policy == "fill":
            for key, arr in columns.items():
                columns[key] = _forward_fill_nan(arr)
        return columns
