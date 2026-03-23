"""
Example plugin: smoothed RSI (SMA of RSI).

Run this file to verify the plugin contract:
  python examples/custom_indicator.py

Registers "SMOOTH_RSI" and runs it on sample data.
"""

from __future__ import annotations

import numpy as np

from ferro_ta import RSI, SMA
from ferro_ta.core.registry import list_indicators, register, run


def smooth_rsi(close, timeperiod=14, smooth=3):
    """Smoothed RSI: RSI then SMA of the RSI series.

    Parameters
    ----------
    close : array-like
        Close prices.
    timeperiod : int
        RSI period (default 14).
    smooth : int
        SMA period applied to RSI (default 3).

    Returns
    -------
    numpy.ndarray
        Smoothed RSI values; same length as close.
    """
    rsi = RSI(close, timeperiod=timeperiod)
    return SMA(rsi, timeperiod=smooth)


def main():
    register("SMOOTH_RSI", smooth_rsi)
    close = np.array(
        [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15, 44.61, 44.33]
    )
    out = run("SMOOTH_RSI", close, timeperiod=5, smooth=2)
    print("SMOOTH_RSI:", out)
    assert "SMOOTH_RSI" in list_indicators(), (
        "SMOOTH_RSI should be in list_indicators()"
    )
    print("OK: plugin registered and run successfully.")


if __name__ == "__main__":
    main()
