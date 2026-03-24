"""
ferro_ta.chunked — Chunked / out-of-core processing.
====================================================

Run ferro-ta indicators on data that is too large to fit in memory by
processing it in overlapping chunks.  Each chunk contains a warm-up prefix
(``overlap`` bars) from the previous chunk so that indicator state is
correct.  After computing the indicator, the warm-up prefix is discarded and
the resulting arrays are concatenated.

Functions
---------
chunk_apply(fn, series, chunk_size, overlap, **fn_kwargs)
    Run a single-input indicator function on a large series in chunks.

make_chunk_ranges(n, chunk_size, overlap)
    Return (start, end) index pairs for chunked processing.

trim_overlap(chunk_out, overlap)
    Discard the first *overlap* elements from an array.

stitch_chunks(chunks)
    Concatenate trimmed chunk outputs into one array.

Rust backend
------------
    ferro_ta._ferro_ta.make_chunk_ranges
    ferro_ta._ferro_ta.trim_overlap
    ferro_ta._ferro_ta.stitch_chunks
    ferro_ta._ferro_ta.chunk_apply_close_indicator

Notes
-----
Indicators that rely on the full history (e.g. HT_TRENDLINE) cannot
produce exact results in chunked mode; the approximation improves with
larger ``overlap`` values.  Indicators with a finite look-back period
(SMA, EMA, RSI, etc.) are exact when ``overlap >= timeperiod - 1``.

For very large datasets or distributed execution, the optional Dask
integration (``dask.dataframe.map_partitions``) can be used directly
by passing any ferro-ta indicator function.  See the example in the
docstring of ``chunk_apply``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import (
    chunk_apply_close_indicator as _rust_chunk_apply_close_indicator,
)
from ferro_ta._ferro_ta import (
    make_chunk_ranges as _rust_make_chunk_ranges,
)
from ferro_ta._ferro_ta import (
    stitch_chunks as _rust_stitch_chunks,
)
from ferro_ta._ferro_ta import (
    trim_overlap as _rust_trim_overlap,
)
from ferro_ta._utils import _to_f64

__all__ = [
    "chunk_apply",
    "make_chunk_ranges",
    "trim_overlap",
    "stitch_chunks",
]

_FASTPATH_DEFAULT_PERIODS: dict[str, int] = {
    "SMA": 30,
    "EMA": 30,
    "RSI": 14,
}


def _resolve_chunk_fastpath(
    fn: Callable[..., Any], fn_kwargs: dict[str, Any]
) -> tuple[str, int] | None:
    name = getattr(fn, "__name__", "").upper()
    if name not in _FASTPATH_DEFAULT_PERIODS:
        return None
    if set(fn_kwargs) - {"timeperiod"}:
        return None
    raw = fn_kwargs.get("timeperiod", _FASTPATH_DEFAULT_PERIODS[name])
    if not isinstance(raw, int):
        return None
    return name, int(raw)


def make_chunk_ranges(
    n: int,
    chunk_size: int,
    overlap: int,
) -> NDArray[np.int64]:
    """Compute start/end index pairs for chunked processing.

    Parameters
    ----------
    n          : int — total length of the series
    chunk_size : int — desired output bars per chunk (>= 1)
    overlap    : int — warm-up bars prepended to each chunk (>= 0)

    Returns
    -------
    numpy.ndarray of int64 with shape (n_chunks, 2) — each row is
    ``[start_index, end_index)`` of the slice to pass to the indicator.

    Examples
    --------
    >>> from ferro_ta.data.chunked import make_chunk_ranges
    >>> make_chunk_ranges(10, 4, 2)
    array([[ 0,  6],
           [ 4, 10]])
    """
    raw = np.asarray(
        _rust_make_chunk_ranges(int(n), int(chunk_size), int(overlap)),
        dtype=np.int64,
    )
    if len(raw) == 0:
        return raw.reshape(0, 2)
    return raw.reshape(-1, 2)


def trim_overlap(
    chunk_out: ArrayLike,
    overlap: int,
) -> NDArray[np.float64]:
    """Discard the first *overlap* elements from a chunk's indicator output.

    Parameters
    ----------
    chunk_out : array-like — indicator output for a chunk
    overlap   : int — number of leading warm-up elements to discard

    Returns
    -------
    numpy.ndarray of float64 — the remaining elements
    """
    arr = np.ascontiguousarray(_to_f64(chunk_out))
    return np.asarray(_rust_trim_overlap(arr, int(overlap)), dtype=np.float64)


def stitch_chunks(
    chunks: list[ArrayLike],
) -> NDArray[np.float64]:
    """Concatenate trimmed chunk outputs into a single array.

    Parameters
    ----------
    chunks : list of array-like — trimmed indicator outputs

    Returns
    -------
    numpy.ndarray of float64 — full concatenated result
    """
    converted = [np.ascontiguousarray(_to_f64(c)) for c in chunks]
    return np.asarray(_rust_stitch_chunks(converted), dtype=np.float64)


def chunk_apply(
    fn: Callable[..., Any],
    series: ArrayLike,
    chunk_size: int = 10_000,
    overlap: int = 100,
    **fn_kwargs: Any,
) -> NDArray[np.float64]:
    """Run a 1-D indicator function on a large series in overlapping chunks.

    Parameters
    ----------
    fn         : callable — indicator function with signature ``fn(series, **kwargs)``
        that accepts a 1-D numpy array and returns a 1-D numpy array of the
        same length.  Examples: ``ferro_ta.SMA``, ``ferro_ta.RSI``.
    series     : array-like — the full (possibly large) input series
    chunk_size : int — output bars per chunk (default 10 000).  Tune this
        for memory/performance.
    overlap    : int — warm-up bars prepended to each chunk (default 100).
        Set to at least ``timeperiod - 1`` for the indicator to be accurate.
    **fn_kwargs : extra keyword arguments forwarded to *fn* on every chunk.

    Returns
    -------
    numpy.ndarray of float64 — full indicator output over the entire series.

    Notes
    -----
    For Dask DataFrames, call ``dask.dataframe.map_partitions`` directly::

        import dask.dataframe as dd
        from ferro_ta import RSI

        ddf = dd.from_pandas(pd.Series(close), npartitions=4)
        result = ddf.map_partitions(lambda s: pd.Series(RSI(s.values)))

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta import SMA
    >>> from ferro_ta.data.chunked import chunk_apply
    >>> rng = np.random.default_rng(0)
    >>> big_series = rng.standard_normal(50_000).cumsum() + 100
    >>> out = chunk_apply(SMA, big_series, chunk_size=5000, overlap=30,
    ...                   timeperiod=20)
    >>> out.shape
    (50000,)
    """
    s = _to_f64(series)
    n = len(s)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    fastpath = _resolve_chunk_fastpath(fn, fn_kwargs)
    if fastpath is not None:
        indicator, timeperiod = fastpath
        return np.asarray(
            _rust_chunk_apply_close_indicator(
                np.ascontiguousarray(s),
                indicator,
                int(timeperiod),
                int(chunk_size),
                int(overlap),
            ),
            dtype=np.float64,
        )

    ranges = make_chunk_ranges(n, chunk_size, overlap)
    if len(ranges) == 0:
        result = fn(s, **fn_kwargs)
        return np.asarray(result, dtype=np.float64)

    trimmed_chunks: list[NDArray[np.float64]] = []

    for i, (start, end) in enumerate(ranges):
        chunk = s[int(start) : int(end)]
        result = fn(chunk, **fn_kwargs)
        result_arr = np.asarray(result, dtype=np.float64)

        # Determine how many leading bars to discard:
        # - first chunk: keep everything (no prior overlap)
        # - subsequent chunks: discard the leading `overlap` bars
        discard = 0 if i == 0 else int(overlap)

        trimmed = trim_overlap(result_arr, discard)
        trimmed_chunks.append(trimmed)

    return stitch_chunks(trimmed_chunks)  # type: ignore[arg-type]
