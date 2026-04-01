"""
Corporate action price adjustment utilities.

adjust_for_splits(close, split_factors, split_indices)
    Apply split adjustments to a close price series (backward-adjusted).

adjust_for_dividends(close, dividends, ex_dates)
    Apply dividend adjustments to a close price series (backward-adjusted).

adjust_ohlcv(open_, high, low, close, volume, split_factors=None, split_indices=None,
             dividends=None, ex_date_indices=None)
    Apply both split and dividend adjustments to a full OHLCV dataset.
    Returns (adj_open, adj_high, adj_low, adj_close, adj_volume).
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["adjust_for_splits", "adjust_for_dividends", "adjust_ohlcv"]


def adjust_for_splits(
    close: ArrayLike,
    split_factors: ArrayLike,  # e.g. [2.0, 3.0] means 2-for-1 then 3-for-1
    split_indices: ArrayLike,  # bar indices of each split (must be sorted ascending)
) -> NDArray:
    """Backward-adjust close prices for stock splits.

    All prices BEFORE a split are divided by the split factor.
    e.g. a 2-for-1 split at bar 100: prices[0:100] are halved.

    Parameters
    ----------
    close : array-like
        Raw close prices.
    split_factors : array-like
        Split factor for each split event (e.g. 2.0 for a 2-for-1 split).
    split_indices : array-like
        Bar index of each split event (0-based, must be sorted ascending).

    Returns
    -------
    NDArray of adjusted close prices.
    """
    c = np.asarray(close, dtype=np.float64).copy()
    factors = np.asarray(split_factors, dtype=np.float64)
    indices = np.asarray(split_indices, dtype=np.intp)

    # Process splits in chronological order; apply backward adjustment
    # (all bars before the split are divided by the factor)
    for idx, factor in zip(indices, factors):
        if factor <= 0:
            raise ValueError(f"split_factor must be > 0, got {factor}")
        c[:idx] /= factor

    return c


def adjust_for_dividends(
    close: ArrayLike,
    dividends: ArrayLike,  # dividend amount per ex-date
    ex_date_indices: ArrayLike,  # bar indices of ex-dividend dates
) -> NDArray:
    """Backward-adjust close prices for cash dividends (proportional method).

    Adjustment factor at ex-date i = (close[i-1] - dividend) / close[i-1].
    All bars before ex-date are multiplied by the cumulative adjustment.

    Parameters
    ----------
    close : array-like
        Raw close prices.
    dividends : array-like
        Dividend amount (in currency units) at each ex-dividend date.
    ex_date_indices : array-like
        Bar index of each ex-dividend date (0-based, sorted ascending).

    Returns
    -------
    NDArray of adjusted close prices.
    """
    c = np.asarray(close, dtype=np.float64).copy()
    divs = np.asarray(dividends, dtype=np.float64)
    indices = np.asarray(ex_date_indices, dtype=np.intp)

    # Process in chronological order
    for idx, div in zip(indices, divs):
        if idx == 0:
            # No prior bar; skip adjustment (nothing to adjust)
            continue
        prev_close = c[idx - 1]
        if prev_close <= 0:
            continue
        adj_factor = (prev_close - div) / prev_close
        if adj_factor <= 0:
            continue
        # All prices before ex-date are multiplied by adj_factor
        c[:idx] *= adj_factor

    return c


def adjust_ohlcv(
    open_: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    split_factors: Optional[ArrayLike] = None,
    split_indices: Optional[ArrayLike] = None,
    dividends: Optional[ArrayLike] = None,
    ex_date_indices: Optional[ArrayLike] = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Apply split and dividend adjustments to full OHLCV data.

    Price arrays are multiplied by cumulative adjustment factor.
    Volume is divided by split factors (shares outstanding adjust inversely).
    Returns (adj_open, adj_high, adj_low, adj_close, adj_volume).

    Parameters
    ----------
    open_, high, low, close : array-like
        Raw OHLCV price arrays.
    volume : array-like
        Raw volume array.
    split_factors : array-like, optional
        Split factors for each split event.
    split_indices : array-like, optional
        Bar indices of split events (required if split_factors provided).
    dividends : array-like, optional
        Dividend amounts for each ex-date.
    ex_date_indices : array-like, optional
        Bar indices of ex-dividend dates (required if dividends provided).

    Returns
    -------
    (adj_open, adj_high, adj_low, adj_close, adj_volume)
    """
    o = np.asarray(open_, dtype=np.float64).copy()
    h = np.asarray(high, dtype=np.float64).copy()
    low_arr = np.asarray(low, dtype=np.float64).copy()
    c = np.asarray(close, dtype=np.float64).copy()
    v = np.asarray(volume, dtype=np.float64).copy()

    n = len(c)

    # Build a per-bar cumulative adjustment factor for prices (starts at 1.0)
    price_adj = np.ones(n, dtype=np.float64)
    # Separate inverse adjustment for volume (splits only)
    vol_adj = np.ones(n, dtype=np.float64)

    # -----------------------------------------------------------------------
    # Apply split adjustments
    # -----------------------------------------------------------------------
    if split_factors is not None and split_indices is not None:
        sf = np.asarray(split_factors, dtype=np.float64)
        si = np.asarray(split_indices, dtype=np.intp)
        for idx, factor in zip(si, sf):
            if factor <= 0:
                raise ValueError(f"split_factor must be > 0, got {factor}")
            # Prices before split are divided by factor
            price_adj[:idx] /= factor
            # Volume before split is multiplied by factor (more shares pre-split)
            vol_adj[:idx] *= factor

    # -----------------------------------------------------------------------
    # Apply dividend adjustments (prices only)
    # -----------------------------------------------------------------------
    if dividends is not None and ex_date_indices is not None:
        divs = np.asarray(dividends, dtype=np.float64)
        ei = np.asarray(ex_date_indices, dtype=np.intp)
        # We need the split-adjusted close at (idx-1) for each dividend event.
        # Instead of recomputing the full array each iteration, read the single
        # element we need: c[idx-1] * price_adj[idx-1].
        for idx, div in zip(ei, divs):
            if idx == 0:
                continue
            prev_close = c[idx - 1] * price_adj[idx - 1]
            if prev_close <= 0:
                continue
            adj_factor = (prev_close - div) / prev_close
            if adj_factor <= 0:
                continue
            price_adj[:idx] *= adj_factor

    adj_open = o * price_adj
    adj_high = h * price_adj
    adj_low = low_arr * price_adj
    adj_close = c * price_adj
    adj_volume = v * vol_adj

    return adj_open, adj_high, adj_low, adj_close, adj_volume
