"""Tests for StreamingOptionGreeks: live IV + greeks per contract."""

import math

import pytest

from ferro_ta.analysis.options import greeks, implied_volatility, option_price
from ferro_ta.data.streaming import StreamingOptionGreeks

STRIKE, RATE = 100.0, 0.05


@pytest.mark.parametrize(
    ("option_type", "model"), [("call", "bsm"), ("put", "black76")]
)
def test_matches_batch_iv_and_greeks(option_type, model):
    stream = StreamingOptionGreeks(
        STRIKE, option_type=option_type, model=model, rate=RATE
    )
    for underlying, vol, t in [
        (102.0, 0.25, 0.5),
        (98.0, 0.31, 0.45),
        (101.0, 0.28, 0.4),
    ]:
        price = option_price(
            underlying, STRIKE, RATE, t, vol, option_type=option_type, model=model
        )
        iv, delta, gamma, vega, theta, rho = stream.update(price, underlying, t)
        expected_iv = implied_volatility(
            price, underlying, STRIKE, RATE, t, option_type=option_type, model=model
        )
        assert iv == pytest.approx(vol, abs=1e-8)
        # Different seeds stop at different points inside the 1e-8 tolerance.
        assert iv == pytest.approx(expected_iv, abs=1e-8)
        g = greeks(
            underlying, STRIKE, RATE, t, iv, option_type=option_type, model=model
        )
        assert (delta, gamma, vega, theta, rho) == pytest.approx(
            (g.delta, g.gamma, g.vega, g.theta, g.rho), rel=1e-12
        )


def test_unsolvable_price_returns_nan_and_recovers():
    stream = StreamingOptionGreeks(STRIKE, rate=RATE)
    stream.update(option_price(100.0, STRIKE, RATE, 0.5, 0.3), 100.0, 0.5)
    assert all(math.isnan(v) for v in stream.update(1.0, 125.0, 0.5))
    iv, *_ = stream.update(option_price(100.0, STRIKE, RATE, 0.5, 0.35), 100.0, 0.5)
    assert iv == pytest.approx(0.35, abs=1e-8)


def test_preview_does_not_move_seed_or_count():
    stream = StreamingOptionGreeks(STRIKE, rate=RATE)
    stream.update(option_price(100.0, STRIKE, RATE, 0.5, 0.3), 100.0, 0.5)
    preview = stream.update(
        option_price(101.0, STRIKE, RATE, 0.5, 0.4), 101.0, 0.5, closed=False
    )
    assert preview[0] == pytest.approx(0.4, abs=1e-8)
    assert stream.count == 1


def test_nan_tick_skipped():
    stream = StreamingOptionGreeks(STRIKE)
    assert all(math.isnan(v) for v in stream.update(math.nan, 100.0, 0.5))
    assert stream.count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strike": 0.0},
        {"strike": -1.0},
        {"strike": 100.0, "option_type": "straddle"},
        {"strike": 100.0, "model": "heston"},
        {"strike": 100.0, "rate": math.inf},
    ],
)
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        StreamingOptionGreeks(**kwargs)


def test_repr():
    assert repr(StreamingOptionGreeks(100.0)) == "StreamingOptionGreeks(strike=100)"
