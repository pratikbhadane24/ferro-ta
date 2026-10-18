"""Smoke test every registry-generated streaming class.

The registry (``crates/ferro_ta_core/src/streaming/registry.rs``) generates
these classes; numeric parity with the batch functions is proven in Rust. This
checks the Python surface: default construction, live previews, pickling and
``repr``.
"""

from __future__ import annotations

import csv
import inspect
import math
import pickle
from pathlib import Path

import pytest

import ferro_ta.data.streaming as streaming

# Every bar of the fixture (252), so long warm-ups (TRIX, Hilbert) still
# reach a value.
N_BARS = 1_000
CSV_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "ohlcv_daily.csv"

# Hand-written classes (and helpers) that predate the registry.
LEGACY = {
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
    "StreamingOptionGreeks",
    "Bar",
    "BarBuilder",
}
GENERATED = sorted(set(streaming.__all__) - LEGACY)

CLOSE_DERIVED = {
    "real",
    "real0",
    "value",
    "x",
    "asset",
    "a",
    "front",
    "position_size",
}
OPEN_DERIVED = {"real1", "b", "benchmark", "next"}


def _load_columns() -> dict[str, list[float]]:
    with CSV_PATH.open() as fh:
        rows = list(csv.DictReader(fh))[:N_BARS]
    return {
        key: [float(r[key]) for r in rows]
        for key in ("open", "high", "low", "close", "volume")
    }


COLUMNS = _load_columns()


def _trailing_mean(values: list[float], k: int) -> list[float]:
    return [
        sum(values[max(0, i + 1 - k) : i + 1]) / len(values[max(0, i + 1 - k) : i + 1])
        for i in range(len(values))
    ]


_CLOSE = COLUMNS["close"]
# Non-degenerate stand-ins (mirrors streaming/fixtures.rs): crossings of the
# default regime thresholds, a positive IV level, volatility bursts.
_SYNTHETIC: dict[str, list[float]] = {
    "fast": _CLOSE,
    "slow": _trailing_mean(_CLOSE, 8),
    "series": [
        c + 3.0 * math.sin(2.1 * i) if (i // 40) % 3 == 2 else c
        for i, c in enumerate(_CLOSE)
    ],
    "iv_series": [
        0.2 + 0.05 * math.sin(0.05 * i) + 0.03 * math.sin(0.31 * i)
        for i in range(len(_CLOSE))
    ],
    "adx": [32.5 + 27.5 * math.sin(0.21 * i) for i in range(len(_CLOSE))],
    "atr": [0.5 + 0.3 * math.sin(0.37 * i) for i in range(len(_CLOSE))],
    "asset_returns": [
        0.01 * math.sin(0.7 * i) + 0.002 * math.cos(1.9 * i) for i in range(len(_CLOSE))
    ],
    "benchmark_returns": [0.008 * math.cos(0.45 * i) for i in range(len(_CLOSE))],
}


def _column(name: str) -> list[float]:
    if name in COLUMNS:
        return COLUMNS[name]
    if name in CLOSE_DERIVED:
        return COLUMNS["close"]
    if name in OPEN_DERIVED:
        return COLUMNS["open"]
    if name in _SYNTHETIC:
        return _SYNTHETIC[name]
    if name == "periods":
        return [5.0 + i % 7 for i in range(N_BARS)]
    if name == "next_weights":
        return [min(1.0, i / 300) for i in range(N_BARS)]
    if name == "funding_rate":
        return [1e-4 * math.sin(i) for i in range(N_BARS)]
    raise AssertionError(f"no test column for input {name!r}")


def _inputs(cls: type) -> list[str]:
    params = inspect.signature(cls.update).parameters
    return [p for p in params if p not in ("self", "closed")]


def _bars(cls: type) -> list[tuple[float, ...]]:
    cols = [_column(name) for name in _inputs(cls)]
    return list(zip(*cols))


def _same(a: object, b: object) -> bool:
    """NaN-aware equality for scalars and tuples."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def _has_value(value: object) -> bool:
    if isinstance(value, tuple):
        return all(_has_value(v) for v in value)
    if isinstance(value, float):
        return not math.isnan(value)
    return value != -1  # i64 index sentinel; i32 codes are never -1-sentinel


def test_registry_classes_are_exported():
    assert len(GENERATED) >= 18
    assert "StreamingRollingMaxindex" in GENERATED


@pytest.mark.parametrize("name", GENERATED)
def test_every_registry_stream_smoke(name: str):
    cls = getattr(streaming, name)
    stream = cls()
    bars = _bars(cls)
    half = len(bars) // 2
    for bar in bars[:half]:
        preview = stream.update(*bar, closed=False)
        committed = stream.update(*bar)
        assert _same(preview, committed), f"{name}: preview != commit"
    assert stream.count == half

    # Our own object, round-tripped in-process: pickle is safe here.
    restored = pickle.loads(pickle.dumps(stream))
    assert restored.count == stream.count
    last: object = None
    for bar in bars[half:]:
        last = stream.update(*bar)
        assert _same(restored.update(*bar), last)
    # Readiness must agree with the last committed value (NaN / -1 index
    # sentinel = not ready), whatever the stream's warm-up length.
    assert stream.is_ready == _has_value(last), f"{name}: is_ready vs {last!r}"
    assert name in repr(stream)


def test_index_stream_reports_absolute_bar_index():
    stream = streaming.StreamingRollingMaxindex(timeperiod=3)
    out = [stream.update(v) for v in [1.0, 5.0, 2.0, 3.0, 4.0, 0.0]]
    assert out == [-1, -1, 1, 1, 4, 4]
    assert repr(stream) == "StreamingRollingMaxindex(timeperiod=3)"
    assert stream.timeperiod == 3


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        streaming.StreamingSum(timeperiod=0)
