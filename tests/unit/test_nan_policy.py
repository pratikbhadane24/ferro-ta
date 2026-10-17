"""nan_policy for batch indicators: 'propagate' (TA-Lib parity) vs 'skip'."""

import math

import numpy as np
import pytest

import ferro_ta as ft
from ferro_ta.data.streaming import (
    StreamingATR,
    StreamingBBands,
    StreamingEMA,
    StreamingMACD,
    StreamingRSI,
    StreamingSMA,
    StreamingStoch,
)

RNG = np.random.default_rng(42)
N = 200
CLOSE = 100 + RNG.standard_normal(N).cumsum()
HIGH = CLOSE + RNG.uniform(0.1, 1.0, N)
LOW = CLOSE - RNG.uniform(0.1, 1.0, N)
BAD = [17, 60, 61, 133]


def dirty(a: np.ndarray, value: float = math.nan) -> np.ndarray:
    out = a.copy()
    out[BAD] = value
    return out


def as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


# (batch call, streaming factory, uses HLC)
CASES = [
    ("SMA", lambda c, **kw: ft.SMA(c, 10, **kw), lambda: StreamingSMA(10), False),
    ("EMA", lambda c, **kw: ft.EMA(c, 10, **kw), lambda: StreamingEMA(10), False),
    ("RSI", lambda c, **kw: ft.RSI(c, 14, **kw), lambda: StreamingRSI(14), False),
    (
        "BBANDS",
        lambda c, **kw: ft.BBANDS(c, 20, 2.0, 2.0, **kw),
        lambda: StreamingBBands(20),
        False,
    ),
    (
        "MACD",
        lambda c, **kw: ft.MACD(c, 12, 26, 9, **kw),
        lambda: StreamingMACD(12, 26, 9),
        False,
    ),
    (
        "ATR",
        lambda h, l, c, **kw: ft.ATR(h, l, c, 14, **kw),
        lambda: StreamingATR(14),
        True,
    ),
    (
        "STOCH",
        lambda h, l, c, **kw: ft.STOCH(h, l, c, 5, 3, 0, 3, 0, **kw),
        lambda: StreamingStoch(5, 3, 3),
        True,
    ),
]
IDS = [c[0] for c in CASES]


def call(batch, hlc, inputs, **kw):
    return as_tuple(batch(*inputs, **kw)) if hlc else as_tuple(batch(inputs[-1], **kw))


def inputs_for(hlc: bool, bad_close: bool = True):
    c = dirty(CLOSE) if bad_close else CLOSE
    return (HIGH, LOW, c) if hlc else (c,)


@pytest.mark.parametrize(("name", "batch", "stream", "hlc"), CASES, ids=IDS)
def test_default_is_propagate_and_unchanged(name, batch, stream, hlc):
    x = inputs_for(hlc)
    default = call(batch, hlc, x)
    explicit = call(batch, hlc, x, nan_policy="propagate")
    for a, b in zip(default, explicit, strict=True):
        np.testing.assert_array_equal(a, b)
    # TA-Lib semantics for running sums / recursions: everything after the
    # first NaN stays NaN (BBANDS recomputes each window and ATR's true range
    # drops NaN terms, so those already recover under "propagate").
    if name not in {"BBANDS", "ATR"}:
        assert np.isnan(default[0][BAD[0] :]).all()


@pytest.mark.parametrize(("name", "batch", "stream", "hlc"), CASES, ids=IDS)
def test_skip_matches_streaming_tick_by_tick(name, batch, stream, hlc):
    x = inputs_for(hlc)
    got = call(batch, hlc, x, nan_policy="skip")
    s = stream()
    streamed = [as_tuple(s.update(*(a[i] for a in x))) for i in range(N)]
    for k, out in enumerate(got):
        expected = np.array([row[k] for row in streamed], dtype=float)
        np.testing.assert_allclose(
            out, expected, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name
        )


@pytest.mark.parametrize(("name", "batch", "stream", "hlc"), CASES, ids=IDS)
def test_skip_recovers_and_marks_bad_bars(name, batch, stream, hlc):
    x = inputs_for(hlc)
    got = call(batch, hlc, x, nan_policy="skip")[0]
    assert np.isnan(got[BAD]).all(), "bad bars report NaN"
    assert np.isfinite(got[-50:]).all(), "values recover after the gaps"
    clean_idx = np.setdiff1d(np.arange(N), BAD)
    on_clean = call(batch, hlc, tuple(a[clean_idx] for a in x))[0]
    np.testing.assert_array_equal(got[clean_idx], on_clean)


@pytest.mark.parametrize(("name", "batch", "stream", "hlc"), CASES, ids=IDS)
def test_skip_without_bad_values_is_identical(name, batch, stream, hlc):
    x = inputs_for(hlc, bad_close=False)
    for a, b in zip(
        call(batch, hlc, x), call(batch, hlc, x, nan_policy="skip"), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_skip_treats_infinity_as_bad():
    x = dirty(CLOSE, math.inf)
    got = ft.SMA(x, 10, nan_policy="skip")
    assert np.isnan(got[BAD]).all() and np.isfinite(got[-50:]).all()


def test_skip_bad_bar_in_any_hlc_input():
    h = HIGH.copy()
    h[90] = math.nan
    got = ft.ATR(h, LOW, CLOSE, 14, nan_policy="skip")
    assert math.isnan(got[90]) and np.isfinite(got[-50:]).all()


def test_invalid_policy_rejected():
    with pytest.raises(ValueError, match="nan_policy"):
        ft.SMA(CLOSE, 10, nan_policy="drop")


def test_wrapped_moving_averages_accept_policy():
    x = dirty(CLOSE)
    for fn in (ft.WMA, ft.DEMA, ft.TEMA, ft.TRIMA, ft.KAMA, ft.T3):
        got = fn(x, 5, nan_policy="skip")
        assert np.isnan(got[BAD]).all() and np.isfinite(got[-30:]).all(), fn.__name__
