"""
ferro_ta.signals — Signal composition and screening.

Provides helpers to combine multiple indicator outputs into a composite score
and to screen/rank symbols by that score.

Functions
---------
compose(signals, weights=None, method='weighted')
    Combine a DataFrame (or 2-D array) of signals into one composite score
    per bar.  Methods: ``'weighted'`` (weighted sum), ``'rank'`` (rank-based),
    ``'mean'`` (equal-weight mean).

screen(scores, top_n=None, bottom_n=None, above=None, below=None)
    Filter/rank a dict or Series of per-symbol scores.

rank_signals(x)
    Compute the fractional rank of each element in *x* (wrapper around Rust).

Rust backend
------------
    ferro_ta._ferro_ta.compose_weighted
    ferro_ta._ferro_ta.rank_series
    ferro_ta._ferro_ta.top_n_indices
    ferro_ta._ferro_ta.bottom_n_indices
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import bottom_n_indices as _rust_bottom_n
from ferro_ta._ferro_ta import compose_weighted as _rust_compose_weighted
from ferro_ta._ferro_ta import rank_series as _rust_rank_series
from ferro_ta._ferro_ta import top_n_indices as _rust_top_n
from ferro_ta._utils import _to_f64

__all__ = [
    "compose",
    "screen",
    "rank_signals",
]


# ---------------------------------------------------------------------------
# rank_signals
# ---------------------------------------------------------------------------


def rank_signals(x: ArrayLike) -> NDArray[np.float64]:
    """Compute the fractional rank of each element (1-based, ascending).

    Ties receive the average of their rank positions.

    Parameters
    ----------
    x : array-like  — 1-D

    Returns
    -------
    numpy.ndarray of ranks in [1, n]

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.signals import rank_signals
    >>> rank_signals(np.array([3.0, 1.0, 2.0]))
    array([3., 1., 2.])
    """
    return _rust_rank_series(_to_f64(x))


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


def compose(
    signals: Any,
    weights: Optional[ArrayLike] = None,
    method: str = "weighted",
) -> NDArray[np.float64]:
    """Combine multiple signal columns into one composite score per bar.

    Parameters
    ----------
    signals : pandas.DataFrame or 2-D array-like, shape (n_bars, n_signals)
        Each column is one indicator/signal.
    weights : array-like of length n_signals, optional
        Weights for each signal column.  Required for ``method='weighted'``.
        If ``None`` and method is ``'weighted'``, equal weights are used.
    method : str
        Composition method:
        - ``'weighted'`` (default) — weighted sum (Rust fast path)
        - ``'mean'``  — equal-weight mean (equivalent to weighted with 1/n)
        - ``'rank'``  — sum of per-signal ranks (rank-based scoring)

    Returns
    -------
    numpy.ndarray of length n_bars

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.signals import compose
    >>> rng = np.random.default_rng(0)
    >>> sigs = rng.standard_normal((50, 3))
    >>> score = compose(sigs, weights=[0.5, 0.3, 0.2])
    >>> score.shape
    (50,)
    """
    try:
        import pandas as pd

        if isinstance(signals, pd.DataFrame):
            arr = signals.values.astype(np.float64, copy=False)
        else:
            arr = np.asarray(signals, dtype=np.float64)
    except ImportError:
        arr = np.asarray(signals, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_bars, n_sigs = arr.shape
    arr = np.ascontiguousarray(arr)

    if method == "mean":
        w = np.full(n_sigs, 1.0 / n_sigs)
        return _rust_compose_weighted(arr, w)
    elif method == "rank":
        # Replace each column with its rank, then sum (ensure contiguous slices)
        ranked = np.column_stack(
            [_rust_rank_series(np.ascontiguousarray(arr[:, j])) for j in range(n_sigs)]
        )
        ranked = np.ascontiguousarray(ranked)
        w = np.full(n_sigs, 1.0)
        return _rust_compose_weighted(ranked, w)
    else:
        # weighted (default)
        if weights is None:
            w = np.full(n_sigs, 1.0 / n_sigs)
        else:
            w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
        return _rust_compose_weighted(arr, w)


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------


def screen(
    scores: Union[dict[str, float], Any],
    top_n: Optional[int] = None,
    bottom_n: Optional[int] = None,
    above: Optional[float] = None,
    below: Optional[float] = None,
) -> Any:
    """Filter and rank symbols by composite score.

    Parameters
    ----------
    scores : dict {symbol: score} or pandas.Series or array-like
        Per-symbol scores.
    top_n : int, optional
        Return the top-N symbols by score.
    bottom_n : int, optional
        Return the bottom-N symbols by score.
    above : float, optional
        Return all symbols with score > *above*.
    below : float, optional
        Return all symbols with score < *below*.

    Returns
    -------
    dict {symbol: score} sorted by score (descending for top_n, ascending for
    bottom_n), or a pandas.DataFrame if pandas is available and input is a
    Series/DataFrame.

    Examples
    --------
    >>> from ferro_ta.analysis.signals import screen
    >>> scores = {"AAPL": 0.8, "GOOG": 0.5, "MSFT": 0.9, "AMZN": 0.3}
    >>> result = screen(scores, top_n=2)
    >>> list(result.keys())
    ['MSFT', 'AAPL']
    """
    # Normalise to dict
    try:
        import pandas as pd

        if isinstance(scores, pd.Series):
            symbols = scores.index.tolist()  # type: ignore[union-attr]
            values = scores.values.astype(np.float64)  # type: ignore[union-attr]
        elif isinstance(scores, dict):
            symbols = list(scores.keys())
            values = np.array(list(scores.values()), dtype=np.float64)
        else:
            symbols = list(range(len(scores)))
            values = np.array(list(scores), dtype=np.float64)
    except ImportError:
        if isinstance(scores, dict):
            symbols = list(scores.keys())
            values = np.array(list(scores.values()), dtype=np.float64)
        else:
            symbols = list(range(len(scores)))
            values = np.array(list(scores), dtype=np.float64)

    if top_n is not None:
        idxs = _rust_top_n(values, int(top_n))
        # Sort by score descending
        idxs = sorted(idxs, key=lambda i: -values[i])
        return {symbols[i]: float(values[i]) for i in idxs}
    if bottom_n is not None:
        idxs = _rust_bottom_n(values, int(bottom_n))
        idxs = sorted(idxs, key=lambda i: values[i])
        return {symbols[i]: float(values[i]) for i in idxs}
    if above is not None:
        return {s: float(v) for s, v in zip(symbols, values) if v > above}
    if below is not None:
        return {s: float(v) for s, v in zip(symbols, values) if v < below}
    # Default: return all sorted descending
    order = sorted(range(len(values)), key=lambda i: -values[i])
    return {symbols[i]: float(values[i]) for i in order}
