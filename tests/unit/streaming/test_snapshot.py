"""Tests for versioned save/restore (to_json / from_json / pickle)."""

import json
import pickle

import numpy as np
import pytest

from ferro_ta.data.streaming import (
    BarBuilder,
    StreamingATR,
    StreamingBBands,
    StreamingCooldown,
    StreamingCross,
    StreamingEMA,
    StreamingMACD,
    StreamingOptionGreeks,
    StreamingRSI,
    StreamingSMA,
    StreamingStoch,
    StreamingStreak,
    StreamingSupertrend,
    StreamingVWAP,
)

RNG = np.random.default_rng(5)
CLOSE = 100 + RNG.standard_normal(300).cumsum() * 0.3


def _inputs(width: int, c: float) -> tuple[float, ...]:
    return {
        1: (c,),
        2: (c, 100.0),
        3: (c + 1, c - 1, c),
        4: (c + 1, c - 1, c, 10.0),
    }[width]


CASES = [
    (lambda: StreamingSMA(10), 1),
    (lambda: StreamingEMA(10), 1),
    (lambda: StreamingRSI(14), 1),
    (lambda: StreamingBBands(20), 1),
    (lambda: StreamingMACD(), 1),
    (lambda: StreamingATR(14), 3),
    (lambda: StreamingStoch(), 3),
    (lambda: StreamingSupertrend(), 3),
    (lambda: StreamingVWAP(), 4),
    (lambda: StreamingCross(0.2), 2),
    (lambda: StreamingStreak(3), 1),
    (lambda: StreamingCooldown(4), 1),
]


@pytest.mark.parametrize("restore", ["json", "pickle"])
@pytest.mark.parametrize(("make", "width"), CASES)
def test_restore_continues_bit_identically(make, width, restore):
    original = make()
    for c in CLOSE[:150]:
        original.update(*_inputs(width, c))
    if restore == "json":
        restored = type(original).from_json(original.to_json())
    else:
        restored = pickle.loads(pickle.dumps(original))
    assert restored.count == original.count
    for c in CLOSE[150:]:
        a = original.update(*_inputs(width, c))
        b = restored.update(*_inputs(width, c))
        assert repr(a) == repr(b)


def test_option_greeks_round_trip_keeps_seed():
    g = StreamingOptionGreeks(100.0, rate=0.05)
    g.update(8.0, 100.0, 0.5)
    restored = pickle.loads(pickle.dumps(g))
    assert restored.update(8.1, 100.2, 0.49) == g.update(8.1, 100.2, 0.49)


def test_snapshot_is_versioned_json():
    doc = json.loads(StreamingSMA(3).to_json())
    assert doc["format"] == 1
    assert doc["kind"] == "sma"


def test_wrong_kind_version_and_garbage_rejected():
    snapshot = StreamingSMA(3).to_json()
    with pytest.raises(ValueError, match="kind"):
        StreamingEMA.from_json(snapshot)
    with pytest.raises(ValueError, match="format"):
        StreamingSMA.from_json(snapshot.replace('"format":1', '"format":2'))
    with pytest.raises(ValueError):
        StreamingSMA.from_json("{}")


def test_bar_builder_round_trip():
    ist = lambda m: 20_377 * 86_400_000 + (9 * 60 + m) * 60_000 - 19_800_000  # noqa: E731
    b = BarBuilder(300)
    b.on_tick(ist(16), 100.0, 1.0)
    restored = pickle.loads(pickle.dumps(b))
    assert restored.forming == b.forming
    assert restored.on_tick(ist(20), 101.0) == b.on_tick(ist(20), 101.0)
    with pytest.raises(ValueError, match="kind"):
        StreamingSMA.from_json(b.to_json())
