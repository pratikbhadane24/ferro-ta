"""
Batch Execution API — run indicators on multiple series in a single call.

This module provides a 2-D batch API that accepts a 2-D numpy array
(n_samples × n_series) and applies an indicator to every column, returning
a 2-D output array of the same shape.

For the most common indicators — SMA, EMA, RSI — the 2-D path is handled
entirely in Rust (a single GIL release for all columns). ``batch_apply``
also dispatches these indicators to Rust when possible; other indicators
use the generic Python fallback path.

Functions
---------
batch_sma    — SMA on every column of a 2-D array  (Rust fast path for 2-D)
batch_ema    — EMA on every column of a 2-D array  (Rust fast path for 2-D)
batch_rsi    — RSI on every column of a 2-D array  (Rust fast path for 2-D)
batch_apply  — Generic batch wrapper with Rust fast-path for SMA/EMA/RSI

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

from collections.abc import Callable, Sequence

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
from ferro_ta._ferro_ta import (
    run_close_indicators as _rust_run_close_indicators,
)
from ferro_ta._ferro_ta import (
    run_hlc_indicators as _rust_run_hlc_indicators,
)
from ferro_ta.core.registry import run as _registry_run
from ferro_ta.indicators.momentum import RSI
from ferro_ta.indicators.overlap import EMA, SMA

__all__ = [
    "batch_sma",
    "batch_ema",
    "batch_rsi",
    "batch_apply",
    "compute_many",
]

_CLOSE_FASTPATH_DEFAULTS: dict[str, int] = {
    "SMA": 30,
    "EMA": 30,
    "RSI": 14,
    "STDDEV": 5,
    "VAR": 5,
    "LINEARREG": 14,
    "LINEARREG_SLOPE": 14,
    "LINEARREG_INTERCEPT": 14,
    "LINEARREG_ANGLE": 14,
    "TSF": 14,
}

_HLC_FASTPATH_DEFAULTS: dict[str, int] = {
    "ATR": 14,
    "NATR": 14,
    "ADX": 14,
    "ADXR": 14,
    "CCI": 14,
    "WILLR": 14,
}

_BATCH_FASTPATH_DEFAULTS: dict[str, int] = {
    "SMA": 30,
    "EMA": 30,
    "RSI": 14,
}


def _resolve_batch_fastpath(
    fn: Callable[..., np.ndarray],
    kwargs: dict[str, object],
) -> tuple[str, int] | None:
    name = getattr(fn, "__name__", "").upper()
    if name not in _BATCH_FASTPATH_DEFAULTS:
        return None
    if set(kwargs) - {"timeperiod"}:
        return None
    raw = kwargs.get("timeperiod", _BATCH_FASTPATH_DEFAULTS[name])
    if not isinstance(raw, int):
        return None
    return name, int(raw)


def _normalize_indicator_spec(
    spec: str | tuple[str, dict[str, object]] | tuple[str, dict[str, object], object],
) -> tuple[str, dict[str, object], object | None]:
    if isinstance(spec, str):
        return spec, {}, None
    if len(spec) == 2:
        name, kwargs = spec
        return name, kwargs, None
    name, kwargs, out_key = spec
    return name, kwargs, out_key


def _extract_timeperiod(
    name: str, kwargs: dict[str, object], defaults: dict[str, int]
) -> int | None:
    if name not in defaults:
        return None
    extra_keys = set(kwargs) - {"timeperiod"}
    if extra_keys:
        return None
    raw_value = kwargs.get("timeperiod", defaults[name])
    if not isinstance(raw_value, int):
        return None
    return raw_value


def compute_many(
    indicators: Sequence[
        str | tuple[str, dict[str, object]] | tuple[str, dict[str, object], object]
    ],
    *,
    close: ArrayLike,
    high: ArrayLike | None = None,
    low: ArrayLike | None = None,
    volume: ArrayLike | None = None,
    parallel: bool = True,
) -> list[object]:
    """Compute multiple indicators over the same arrays with grouped Rust calls.

    Supported single-output indicators are grouped into one Rust boundary crossing
    per input-shape family (`close` only or `high/low/close`). Unsupported specs
    fall back to the regular registry path, preserving behavior.
    """

    close_arr = np.ascontiguousarray(close, dtype=np.float64)
    high_arr = None if high is None else np.ascontiguousarray(high, dtype=np.float64)
    low_arr = None if low is None else np.ascontiguousarray(low, dtype=np.float64)
    volume_arr = (
        None if volume is None else np.ascontiguousarray(volume, dtype=np.float64)
    )

    normalized = [_normalize_indicator_spec(spec) for spec in indicators]
    results: list[object | None] = [None] * len(normalized)

    close_indices: list[int] = []
    close_names: list[str] = []
    close_periods: list[int] = []

    hlc_indices: list[int] = []
    hlc_names: list[str] = []
    hlc_periods: list[int] = []

    for idx, (name, kwargs, out_key) in enumerate(normalized):
        if out_key is None:
            close_period = _extract_timeperiod(name, kwargs, _CLOSE_FASTPATH_DEFAULTS)
            if close_period is not None:
                close_indices.append(idx)
                close_names.append(name)
                close_periods.append(close_period)
                continue

            hlc_period = _extract_timeperiod(name, kwargs, _HLC_FASTPATH_DEFAULTS)
            if hlc_period is not None and high_arr is not None and low_arr is not None:
                hlc_indices.append(idx)
                hlc_names.append(name)
                hlc_periods.append(hlc_period)
                continue

    if close_names:
        grouped = _rust_run_close_indicators(
            close_arr, close_names, close_periods, parallel
        )
        for idx, value in zip(close_indices, grouped):
            results[idx] = np.asarray(value, dtype=np.float64)

    if hlc_names and high_arr is not None and low_arr is not None:
        grouped = _rust_run_hlc_indicators(
            high_arr, low_arr, close_arr, hlc_names, hlc_periods, parallel
        )
        for idx, value in zip(hlc_indices, grouped):
            results[idx] = np.asarray(value, dtype=np.float64)

    for idx, (name, kwargs, _) in enumerate(normalized):
        if results[idx] is not None:
            continue
        try:
            results[idx] = _registry_run(name, close_arr, **kwargs)
            continue
        except (TypeError, Exception):
            pass

        if high_arr is not None and low_arr is not None:
            try:
                results[idx] = _registry_run(
                    name, high_arr, low_arr, close_arr, **kwargs
                )
                continue
            except Exception:
                pass
            if volume_arr is not None:
                try:
                    results[idx] = _registry_run(
                        name, high_arr, low_arr, close_arr, volume_arr, **kwargs
                    )
                    continue
                except Exception:
                    pass

        raise ValueError(
            f"Cannot call indicator '{name}': insufficient data columns or incompatible parameters."
        )

    return [result for result in results]


def batch_apply(
    data: ArrayLike,
    fn: Callable[..., np.ndarray],
    **kwargs,
) -> np.ndarray:
    """Apply any single-series indicator *fn* to every column of *data*.

    For recognized close-only indicators (SMA/EMA/RSI with default or
    ``timeperiod`` argument only), this function dispatches to the Rust
    batch kernels.  Otherwise it falls back to a Python per-column loop.

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

    fastpath = _resolve_batch_fastpath(fn, kwargs)
    if fastpath is not None:
        indicator, timeperiod = fastpath
        contiguous = np.ascontiguousarray(arr)
        if indicator == "SMA":
            return np.asarray(_rust_batch_sma(contiguous, timeperiod, True))
        if indicator == "EMA":
            return np.asarray(_rust_batch_ema(contiguous, timeperiod, True))
        return np.asarray(_rust_batch_rsi(contiguous, timeperiod, True))

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
