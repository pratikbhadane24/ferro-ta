"""Tests for live-bar streaming: closed=False previews, introspection, copy, NaN ticks."""

import copy
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ferro_ta.data.streaming import (
    StreamingATR,
    StreamingBBands,
    StreamingEMA,
    StreamingMACD,
    StreamingRSI,
    StreamingSMA,
    StreamingStoch,
    StreamingSupertrend,
    StreamingVWAP,
)

TICKS_PER_BAR = 3

# (factory, number of float inputs per bar)
INDICATORS = [
    (lambda: StreamingSMA(5), 1),
    (lambda: StreamingEMA(5), 1),
    (lambda: StreamingRSI(5), 1),
    (lambda: StreamingBBands(5), 1),
    (lambda: StreamingMACD(3, 6, 3), 1),
    (lambda: StreamingATR(5), 3),
    (lambda: StreamingStoch(5, 3, 3), 3),
    (lambda: StreamingSupertrend(5, 3.0), 3),
    (lambda: StreamingVWAP(), 4),
]
IDS = [
    "SMA",
    "EMA",
    "RSI",
    "BBands",
    "MACD",
    "ATR",
    "Stoch",
    "Supertrend",
    "VWAP",
]


def _bars(closes: list[float], width: int) -> list[tuple[float, ...]]:
    """Build (close,) / (high, low, close) / (high, low, close, volume) bars."""
    if width == 1:
        return [(c,) for c in closes]
    hlc = [(c + 1.0, c - 1.0, c) for c in closes]
    if width == 3:
        return hlc
    return [(*b, 10.0) for b in hlc]


def _same(a, b) -> bool:
    """Bitwise-style equality that treats NaN == NaN, for scalars and tuples."""
    return repr(a) == repr(b)


@pytest.mark.parametrize(("make", "width"), INDICATORS, ids=IDS)
@settings(max_examples=50, deadline=None)
@given(
    closes=st.lists(
        st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False),
        min_size=1,
        max_size=60,
    )
)
def test_previews_never_change_committed_stream(make, width, closes):
    plain, live = make(), make()
    for bar in _bars(closes, width):
        for t in range(TICKS_PER_BAR):
            live.update(*(x + 0.1 * (t + 1) for x in bar), closed=False)
        expected = plain.update(*bar)
        assert _same(live.update(*bar, closed=False), expected)
        assert _same(live.update(*bar), expected)
    assert live.count == len(closes)


class TestIntrospection:
    def test_count_and_is_ready(self):
        sma = StreamingSMA(3)
        assert sma.count == 0
        assert not sma.is_ready
        sma.update(1.0)
        sma.update(2.0)
        sma.update(50.0, closed=False)
        assert sma.count == 2
        assert not sma.is_ready
        assert sma.update(3.0) == pytest.approx(2.0)
        assert sma.is_ready

    def test_reset_clears_introspection(self):
        ema = StreamingEMA(2)
        for v in (1.0, 2.0, 3.0):
            ema.update(v)
        ema.reset()
        assert ema.count == 0
        assert not ema.is_ready

    def test_closed_is_keyword_only(self):
        with pytest.raises(TypeError):
            StreamingSMA(3).update(1.0, False)

    def test_preview_returns_value_commit_would(self):
        rsi = StreamingRSI(3)
        for v in (10.0, 11.0, 10.5, 12.0):
            rsi.update(v)
        preview = rsi.update(13.0, closed=False)
        assert preview == rsi.update(13.0)


class TestCopy:
    @pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
    def test_copy_is_independent_snapshot(self, copier):
        a = StreamingEMA(3)
        for v in (1.0, 2.0, 3.0):
            a.update(v)
        b = copier(a)
        assert a.update(10.0) == b.update(10.0)
        a.update(0.0)
        b.update(100.0)
        assert a.update(5.0) != b.update(5.0)
        assert b.count == a.count


class TestNonFiniteTicks:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize(("make", "width"), INDICATORS, ids=IDS)
    def test_bad_tick_does_not_poison_state(self, make, width, bad):
        closes = list(np.linspace(100.0, 120.0, 40))
        bars = _bars(closes, width)
        ref = make()
        reference = [ref.update(*b) for b in bars]

        live = make()
        got = []
        for i, bar in enumerate(bars):
            if i == 20:
                nan_bar = (bad,) * width
                live.update(*nan_bar, closed=False)
                live.update(*nan_bar)
            got.append(live.update(*bar))
        assert all(_same(g, r) for g, r in zip(got, reference, strict=True))
