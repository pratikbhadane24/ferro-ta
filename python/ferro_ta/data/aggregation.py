"""
ferro_ta.aggregation — Tick and trade aggregation pipeline.

Aggregates raw tick or trade data (streams of (timestamp, price, size)) into
OHLCV bars using three bar types:
- **time bars**   — fixed time intervals (e.g. every 1 minute)
- **volume bars** — fixed volume threshold per bar
- **tick bars**   — fixed number of ticks per bar

The compute-intensive bar accumulation is implemented in Rust; this module
provides the Python-facing API with DataFrame support.

Functions
---------
aggregate_ticks(ticks, rule)
    Aggregate a tick stream to OHLCV bars.  The *rule* string specifies the
    bar type and parameter:
    - ``'time:<seconds>'``   — e.g. ``'time:60'`` for 1-minute bars
    - ``'volume:<threshold>'`` — e.g. ``'volume:1000'`` for 1000-unit volume bars
    - ``'tick:<n>'``         — e.g. ``'tick:100'`` for 100-tick bars

Rust backend
------------
All accumulation logic delegates to::

    ferro_ta._ferro_ta.aggregate_tick_bars
    ferro_ta._ferro_ta.aggregate_volume_bars_ticks
    ferro_ta._ferro_ta.aggregate_time_bars
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ferro_ta._ferro_ta import aggregate_tick_bars as _rust_tick_bars
from ferro_ta._ferro_ta import aggregate_time_bars as _rust_time_bars
from ferro_ta._ferro_ta import aggregate_volume_bars_ticks as _rust_volume_bars_ticks
from ferro_ta._utils import _to_f64

__all__ = [
    "aggregate_ticks",
    "TickAggregator",
]

# ---------------------------------------------------------------------------
# _parse_rule
# ---------------------------------------------------------------------------


def _parse_rule(rule: str) -> tuple[str, float]:
    """Parse a rule string into (bar_type, parameter).

    Supported formats::

        'time:60'      → ('time', 60.0)
        'volume:1000'  → ('volume', 1000.0)
        'tick:100'     → ('tick', 100.0)
    """
    parts = rule.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid rule format: {rule!r}. "
            "Expected 'time:<seconds>', 'volume:<threshold>', or 'tick:<n>'."
        )
    bar_type = parts[0].lower().strip()
    if bar_type not in ("time", "volume", "tick"):
        raise ValueError(
            f"Unknown bar type {bar_type!r}. Supported types: 'time', 'volume', 'tick'."
        )
    try:
        param = float(parts[1].strip())
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse parameter {parts[1]!r} as a number in rule {rule!r}."
        ) from exc
    if param <= 0:
        raise ValueError(f"Rule parameter must be > 0, got {param} in rule {rule!r}.")
    return bar_type, param


# ---------------------------------------------------------------------------
# aggregate_ticks
# ---------------------------------------------------------------------------


def aggregate_ticks(
    ticks: Any,
    rule: str = "tick:100",
    *,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    size_col: str = "size",
) -> Any:
    """Aggregate tick/trade data into OHLCV bars.

    Parameters
    ----------
    ticks : pandas.DataFrame, list of (timestamp, price, size), or dict of arrays
        Tick data.  Accepted formats:

        1. **pandas DataFrame** with columns ``timestamp``, ``price``, ``size``
           (column names configurable via *_col* parameters).  The timestamp
           column must contain numeric Unix timestamps (seconds) for time bars.
        2. **list of tuples** ``[(ts, price, size), …]``.
        3. **dict** ``{'timestamp': array, 'price': array, 'size': array}``.

    rule : str
        Bar specification:
        - ``'tick:<n>'``           — every N ticks become one bar (default ``'tick:100'``)
        - ``'volume:<threshold>'`` — every N units of volume become one bar
        - ``'time:<seconds>'``     — every N seconds become one bar

    timestamp_col, price_col, size_col : str
        Column names when *ticks* is a DataFrame.

    Returns
    -------
    pandas.DataFrame or tuple of numpy arrays
        If a DataFrame was passed in (or pandas is available), returns a
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume``, and (for time bars) ``timestamp``.
        Otherwise returns a tuple ``(open, high, low, close, volume)``.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.data.aggregation import aggregate_ticks
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> price = 100 + np.cumsum(rng.normal(0, 0.1, n))
    >>> size  = rng.uniform(10, 100, n)
    >>> bars = aggregate_ticks({"price": price, "size": size}, rule="tick:50")
    >>> len(bars["open"]) == n // 50 + (1 if n % 50 != 0 else 0)
    True
    """
    bar_type, param = _parse_rule(rule)

    # --- Normalise input ---
    ts_arr: Optional[NDArray[np.float64]] = None
    if isinstance(ticks, list):
        # list of (ts, price, size) tuples
        arr = np.ascontiguousarray(ticks, dtype=np.float64)
        ts_arr = np.ascontiguousarray(arr[:, 0])
        price_arr = np.ascontiguousarray(arr[:, 1])
        size_arr = np.ascontiguousarray(arr[:, 2])
    elif isinstance(ticks, dict):
        price_arr = _to_f64(ticks[price_col])
        size_arr = _to_f64(ticks[size_col])
        if timestamp_col in ticks:
            ts_arr = _to_f64(ticks[timestamp_col])
    else:
        # pandas DataFrame
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for DataFrame input") from exc
        price_arr = _to_f64(ticks[price_col].values)
        size_arr = _to_f64(ticks[size_col].values)
        if timestamp_col in ticks.columns:
            ts_arr = _to_f64(ticks[timestamp_col].values)

    # --- Aggregate ---
    if bar_type == "tick":
        ro, rh, rl, rc, rv = _rust_tick_bars(price_arr, size_arr, int(param))
        extra: Optional[NDArray] = None
    elif bar_type == "volume":
        ro, rh, rl, rc, rv = _rust_volume_bars_ticks(price_arr, size_arr, param)
        extra = None
    else:  # time
        if ts_arr is None:
            raise ValueError("Time bars require a timestamp column in the tick data.")
        period_secs = int(param)
        labels = (ts_arr // period_secs).astype(np.int64)
        ro, rh, rl, rc, rv, lbl = _rust_time_bars(price_arr, size_arr, labels)
        extra = lbl

    # --- Return ---
    try:
        import pandas as pd

        df: dict[str, Any] = {
            "open": ro,
            "high": rh,
            "low": rl,
            "close": rc,
            "volume": rv,
        }
        if extra is not None:
            df["timestamp"] = (extra * int(param)).astype(np.int64)
        return pd.DataFrame(df)
    except ImportError:
        return {"open": ro, "high": rh, "low": rl, "close": rc, "volume": rv}


# ---------------------------------------------------------------------------
# TickAggregator — class-based API
# ---------------------------------------------------------------------------


class TickAggregator:
    """Class-based API for tick aggregation.

    Parameters
    ----------
    rule : str
        Bar specification (see :func:`aggregate_ticks`).

    Examples
    --------
    >>> from ferro_ta.data.aggregation import TickAggregator
    >>> agg = TickAggregator(rule="tick:50")
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ticks = {"price": rng.uniform(99, 101, 200), "size": rng.uniform(1, 10, 200)}
    >>> bars = agg.aggregate(ticks)
    >>> len(bars["open"]) >= 4
    True
    """

    def __init__(self, rule: str = "tick:100") -> None:
        self.rule = rule
        # Validate rule at construction time
        _parse_rule(rule)

    def aggregate(self, ticks: Any, **kwargs: Any) -> Any:
        """Aggregate *ticks* into bars.  See :func:`aggregate_ticks`."""
        return aggregate_ticks(ticks, rule=self.rule, **kwargs)

    def __repr__(self) -> str:
        return f"TickAggregator(rule={self.rule!r})"
