"""
Streaming / Incremental Indicators — bar-by-bar stateful classes.

All streaming classes are implemented in Rust (PyO3) for maximum performance.
The Python module re-exports the Rust classes from the ``_ferro_ta`` extension.
The extension must be built; there is no Python fallback.

Usage
-----
>>> from ferro_ta.data.streaming import StreamingSMA, StreamingEMA, StreamingRSI
>>> import numpy as np
>>> sma = StreamingSMA(period=3)
>>> for close in [10.0, 11.0, 12.0, 13.0, 14.0]:
...     val = sma.update(close)
...     print(f"{close} → {val:.4f}" if not np.isnan(val) else f"{close} → NaN")
10.0 → NaN
11.0 → NaN
12.0 → 11.0000
13.0 → 12.0000
14.0 → 13.0000

Live bars
---------
``update(x)`` commits a closed bar; ``update(x, closed=False)`` previews the
still-forming bar without changing state. ``count`` / ``is_ready`` report
committed progress, ``copy.copy`` snapshots state, and NaN/±inf ticks are skipped
(they return NaN without poisoning state).

Available classes
-----------------
StreamingSMA        — Simple Moving Average
StreamingEMA        — Exponential Moving Average
StreamingRSI        — Relative Strength Index (Wilder seeding)
StreamingATR        — Average True Range (Wilder seeding)
StreamingBBands     — Bollinger Bands (upper, middle, lower)
StreamingMACD       — MACD line, signal, histogram
StreamingStoch      — Slow Stochastic (slowk, slowd)
StreamingVWAP       — Volume Weighted Average Price (cumulative)
StreamingSupertrend — ATR-based Supertrend
StreamingCross      — Dead-band cross of two series (1 / -1 / 0 events)
StreamingStreak     — "N bars in a row" condition
StreamingCooldown   — Mute repeat signals for N bars

Rust backend
------------
All classes are PyO3 classes compiled into the ``_ferro_ta`` extension module.
Import them directly from the extension for zero-overhead access::

    from ferro_ta._ferro_ta import StreamingSMA
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Import Rust-backed streaming classes from the compiled extension.
# ---------------------------------------------------------------------------
from ferro_ta._ferro_ta import (  # noqa: F401
    StreamingATR,
    StreamingBBands,
    StreamingCooldown,
    StreamingCross,
    StreamingEMA,
    StreamingMACD,
    StreamingRSI,
    StreamingSMA,
    StreamingStoch,
    StreamingStreak,
    StreamingSupertrend,
    StreamingVWAP,
)

__all__ = [
    "StreamingSMA",
    "StreamingEMA",
    "StreamingRSI",
    "StreamingATR",
    "StreamingBBands",
    "StreamingMACD",
    "StreamingStoch",
    "StreamingVWAP",
    "StreamingSupertrend",
    "StreamingCross",
    "StreamingStreak",
    "StreamingCooldown",
]
