"""Class names of the streaming indicators, shared by the code generators.

The 14 hand-written streams predate the registry and keep their historical
class names (``StreamingSMA``, ``StreamingBBands`` ...). Every registry stream
is ``Streaming`` + UpperCamel(name), e.g. ``rolling_maxindex`` ->
``StreamingRollingMaxindex`` (``registry_names_follow_convention`` in
``crates/ferro_ta_core/src/streaming/registry.rs`` enforces the same rule).
"""

from __future__ import annotations

LEGACY_STREAM_CLASSES = {
    "sma": "StreamingSMA",
    "ema": "StreamingEMA",
    "rsi": "StreamingRSI",
    "atr": "StreamingATR",
    "bbands": "StreamingBBands",
    "macd": "StreamingMACD",
    "stoch": "StreamingStoch",
    "vwap": "StreamingVWAP",
    "supertrend": "StreamingSupertrend",
    "cross": "StreamingCross",
    "streak": "StreamingStreak",
    "cooldown": "StreamingCooldown",
    "option_greeks": "StreamingOptionGreeks",
    "backtest": "StreamingBacktest",
}


def upper_camel(snake: str) -> str:
    """`rolling_maxindex` -> `RollingMaxindex` (same rule as Go's `go_name`)."""
    return "".join(part[:1].upper() + part[1:] for part in snake.split("_") if part)


def stream_class(name: str) -> str:
    """Core / Python class name of the stream called ``name`` in the C ABI."""
    return LEGACY_STREAM_CLASSES.get(name) or "Streaming" + upper_camel(name)


def is_registry_stream(name: str) -> bool:
    """True for streams generated from the registry (not hand-written)."""
    return name not in LEGACY_STREAM_CLASSES
