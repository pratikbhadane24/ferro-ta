"""Tests for the session-aligned tick -> bar builder."""

import copy

import numpy as np
import pytest

from ferro_ta import RSI
from ferro_ta.data.streaming import Bar, BarBuilder, StreamingRSI

DAY = 20_377
IST_MS = 19_800_000


def ist(h: int, m: int, s: int = 0, ms: int = 0, day_shift: int = 0) -> int:
    """UTC epoch ms for an IST wall-clock time."""
    return (DAY + day_shift) * 86_400_000 + ((h * 60 + m) * 60 + s) * 1000 + ms - IST_MS


def test_ticks_bucket_on_nse_grid():
    b = BarBuilder(300)
    for ts, price, qty in [
        (ist(9, 15, 0, 500), 100.0, 2.0),
        (ist(9, 17), 103.0, 1.0),
        (ist(9, 18), 99.0, 4.0),
        (ist(9, 19, 59), 101.0, 3.0),
    ]:
        assert b.on_tick(ts, price, qty) == ("accepted", None)
    bar = b.forming
    assert (bar.start_ms, bar.end_ms) == (ist(9, 15), ist(9, 20))
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume, bar.ticks) == (
        100.0,
        103.0,
        99.0,
        101.0,
        10.0,
        4,
    )


def test_boundary_tick_closes_and_opens_next():
    b = BarBuilder(300)
    b.on_tick(ist(9, 19), 100.0)
    status, closed = b.on_tick(ist(9, 20), 105.0)
    assert status == "accepted"
    assert isinstance(closed, Bar) and closed.end_ms == ist(9, 20)
    assert b.forming.start_ms == ist(9, 20)


def test_on_time_and_statuses():
    b = BarBuilder(300)
    b.on_tick(ist(9, 16), 100.0, 1.0)
    assert b.on_tick(ist(9, 16), 100.0, 1.0) == ("duplicate", None)
    assert b.on_tick(ist(9, 15, 30), 100.0) == ("late", None)
    assert b.on_tick(ist(9, 10), 100.0) == ("out_of_session", None)
    assert b.on_tick(ist(9, 17), float("nan")) == ("invalid", None)
    assert b.on_time(ist(9, 19, 59)) is None
    assert b.on_time(ist(9, 20)).start_ms == ist(9, 15)
    assert b.forming is None


def test_custom_session_and_offset():
    # 1-hour bars, 10:00-16:00 at UTC-5 (e.g. a US-style session).
    b = BarBuilder(
        3600, session_open="10:00", session_close="16:00", utc_offset_minutes=-300
    )
    ts = DAY * 86_400_000 + (10 * 3600 + 30 * 60) * 1000 + 5 * 3_600_000
    assert b.on_tick(ts, 50.0)[0] == "accepted"
    assert b.forming.start_ms == DAY * 86_400_000 + 10 * 3_600_000 + 5 * 3_600_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_secs": 0},
        {"interval_secs": 300, "session_open": "9.15"},
        {"interval_secs": 300, "session_open": "25:00"},
        {"interval_secs": 300, "session_open": "16:00", "session_close": "15:30"},
        {"interval_secs": 300, "utc_offset_minutes": 24 * 60},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        BarBuilder(**kwargs)


def test_copy_is_independent():
    a = BarBuilder(300)
    a.on_tick(ist(9, 16), 100.0)
    b = copy.copy(a)
    b.on_tick(ist(9, 17), 200.0)
    assert a.forming.high == 100.0 and b.forming.high == 200.0


def test_drives_streaming_rsi_like_batch_over_closed_bars():
    """Ticks -> bars -> StreamingRSI (preview on forming, commit on close)
    must equal batch RSI over the closed bars' closes."""
    rng = np.random.default_rng(11)
    builder, rsi = BarBuilder(60), StreamingRSI(14)
    closes, streamed = [], []
    ts, price = ist(9, 15), 100.0
    for _ in range(4000):
        ts += int(rng.integers(1_000, 30_000))
        price += float(rng.standard_normal()) * 0.2
        status, closed = builder.on_tick(ts, price, 1.0)
        if status != "accepted":
            continue
        if closed is not None:
            closes.append(closed.close)
            streamed.append(rsi.update(closed.close))
        rsi.update(builder.forming.close, closed=False)
    assert len(closes) > 50
    batch = RSI(np.array(closes), timeperiod=14)
    np.testing.assert_allclose(streamed, batch, equal_nan=True, rtol=1e-12)
