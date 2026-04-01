"""
OHLCV bar aggregation utilities.

resample_ohlcv(open, high, low, close, volume, factor)
    Aggregate every `factor` bars into one OHLCV bar.
    open  = first bar's open
    high  = max of highs
    low   = min of lows
    close = last bar's close
    volume = sum of volumes

resample_ohlcv_labels(n_bars, factor)
    Return an integer label array of length n_bars where label[i] = i // factor.
    Useful for aligning fine-bar signals with coarse-bar indicators.

align_to_coarse(coarse_values, factor, n_fine_bars)
    Broadcast a coarse-bar array back to fine-bar length by repeating each value `factor` times.
    Handles the case where n_fine_bars % factor != 0 (last group may be partial).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["resample_ohlcv", "resample_ohlcv_labels", "align_to_coarse"]


def resample_ohlcv(
    open_: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    factor: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Aggregate fine-bar OHLCV into coarser bars.

    Parameters
    ----------
    open_ : array-like
        Fine-bar open prices.
    high : array-like
        Fine-bar high prices.
    low : array-like
        Fine-bar low prices.
    close : array-like
        Fine-bar close prices.
    volume : array-like
        Fine-bar volume.
    factor : int
        Number of fine bars per coarse bar (e.g. 5 for 1-min -> 5-min).

    Returns
    -------
    (open, high, low, close, volume) arrays of length ceil(n / factor).
    Only complete groups are returned — if n % factor != 0, trailing bars are dropped.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")

    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)

    n = len(o)
    n_complete = (n // factor) * factor  # truncate to complete bars

    o = o[:n_complete].reshape(-1, factor)
    h = h[:n_complete].reshape(-1, factor)
    low_arr = low_arr[:n_complete].reshape(-1, factor)
    c = c[:n_complete].reshape(-1, factor)
    v = v[:n_complete].reshape(-1, factor)

    return (
        o[:, 0],  # open  = first bar's open
        h.max(axis=1),  # high  = max of highs
        low_arr.min(axis=1),  # low   = min of lows
        c[:, -1],  # close = last bar's close
        v.sum(axis=1),  # volume = sum of volumes
    )


def resample_ohlcv_labels(n_bars: int, factor: int) -> NDArray:
    """Return coarse-bar index for each fine bar (i // factor).

    Parameters
    ----------
    n_bars : int
        Number of fine-resolution bars.
    factor : int
        Number of fine bars per coarse bar.

    Returns
    -------
    NDArray of int64, shape (n_bars,), where label[i] = i // factor.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    return np.arange(n_bars, dtype=np.int64) // factor


def align_to_coarse(coarse_values: ArrayLike, factor: int, n_fine_bars: int) -> NDArray:
    """Broadcast coarse-bar array back to fine-bar resolution.

    Each coarse value is repeated `factor` times. If n_fine_bars % factor != 0,
    the last coarse value covers the partial group at the end.

    Parameters
    ----------
    coarse_values : array-like
        Values at coarse resolution, shape (n_coarse,).
    factor : int
        Number of fine bars per coarse bar.
    n_fine_bars : int
        Total number of fine bars to produce.

    Returns
    -------
    NDArray of shape (n_fine_bars,).
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")

    coarse = np.asarray(coarse_values, dtype=np.float64)
    n_coarse = len(coarse)

    # Build the full repeated array (may be longer than n_fine_bars if partial group exists)
    repeated = np.repeat(coarse, factor)

    # If repeated is shorter than n_fine_bars (shouldn't happen with correct n_coarse,
    # but handle defensively), pad with last value
    if len(repeated) < n_fine_bars:
        pad = np.full(
            n_fine_bars - len(repeated), coarse[-1] if n_coarse > 0 else np.nan
        )
        repeated = np.concatenate([repeated, pad])

    return repeated[:n_fine_bars]
