"""Tests for the OptionStrategy struct API (breakevens, max P/L, PoP, scenarios)."""

import math

import numpy as np
import pytest

from ferro_ta.analysis.derivatives_payoff import (
    PayoffLeg,
    aggregate_greeks,
    option_strategy,
    strategy_payoff,
)

T, VOL = 0.25, 0.2


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def opt(side: str, right: str, strike: float, premium: float, **kw) -> PayoffLeg:
    return PayoffLeg(
        instrument="option",
        side=side,
        option_type=right,
        strike=strike,
        premium=premium,
        volatility=kw.pop("volatility", VOL),
        time_to_expiry=kw.pop("time_to_expiry", T),
        **kw,
    )


IRON_CONDOR = [
    opt("long", "put", 80, 1),
    opt("short", "put", 90, 3),
    opt("short", "call", 110, 3),
    opt("long", "call", 120, 1),
]


def test_iron_condor_breakevens_and_extremes():
    s = option_strategy(IRON_CONDOR)
    assert s.breakevens() == [86.0, 114.0]
    assert s.max_profit() == (pytest.approx(4.0), pytest.approx(90.0))
    value, spot = s.max_loss()
    assert value == pytest.approx(-6.0)
    assert spot is not None


def test_unlimited_reported_as_inf_with_no_spot():
    long_call = option_strategy([opt("long", "call", 100, 5)])
    assert long_call.max_profit() == (math.inf, None)
    naked_call = option_strategy([opt("short", "call", 100, 5)])
    assert naked_call.max_loss() == (-math.inf, None)


def test_covered_call_quantity_and_multiplier():
    s = option_strategy(
        [
            PayoffLeg(
                instrument="stock",
                side="long",
                entry_price=100,
                quantity=2,
                multiplier=50,
            ),
            opt("short", "call", 110, 3, quantity=2, multiplier=50),
        ]
    )
    assert s.max_profit()[0] == pytest.approx(13 * 100)
    assert s.max_loss() == (pytest.approx(-97 * 100), pytest.approx(0.0))
    assert s.breakevens() == [97.0]


def test_payoff_and_greeks_match_existing_api():
    grid = np.linspace(60, 140, 81)
    s = option_strategy(IRON_CONDOR)
    np.testing.assert_array_equal(
        s.payoff(grid), strategy_payoff(grid, legs=IRON_CONDOR)
    )
    g = s.greeks(100.0)
    expected = aggregate_greeks(100.0, legs=IRON_CONDOR)
    assert g == pytest.approx(
        (expected.delta, expected.gamma, expected.vega, expected.theta, expected.rho),
        rel=1e-12,
    )


def test_probability_of_profit_long_call_closed_form():
    # Core normal CDF is a ~1e-7-accurate approximation; math.erf is exact.
    s = option_strategy([opt("long", "call", 100, 5)])
    spot, r = 100.0, 0.05
    d2 = (math.log(spot / 105) + (r - 0.5 * VOL**2) * T) / (VOL * math.sqrt(T))
    assert s.probability_of_profit(spot, VOL, T, rate=r) == pytest.approx(
        norm_cdf(d2), abs=1e-7
    )


def test_scenario_grid_shape_and_vol_sensitivity():
    s = option_strategy(IRON_CONDOR)
    spots, shifts = [90.0, 100.0, 110.0], [-0.05, 0.0, 0.03]
    grid = s.scenario_grid(spots, shifts, days_forward=5)
    assert grid.shape == (3, 3)
    assert grid[2, 1] < grid[1, 1], "short premium loses when vol rises"


def test_mixed_expiries_and_bad_legs_raise():
    calendar = option_strategy(
        [
            opt("short", "call", 100, 3, time_to_expiry=0.1),
            opt("long", "call", 100, 5, time_to_expiry=0.3),
        ]
    )
    with pytest.raises(ValueError, match="mixed expiries"):
        calendar.breakevens()
    assert calendar.greeks(100.0)[2] > 0
    with pytest.raises(ValueError, match="quantity"):
        option_strategy([opt("long", "call", 100, 5, quantity=0)])
    with pytest.raises(ValueError):
        option_strategy([])


def test_accepts_dict_legs():
    s = option_strategy(
        [
            {
                "instrument": "option",
                "side": "long",
                "option_type": "put",
                "strike": 100,
                "premium": 4,
            }
        ]
    )
    assert s.breakevens() == [96.0]


def test_constructor_and_signature_introspection():
    import inspect

    from ferro_ta.analysis.derivatives_payoff import OptionStrategy

    s = OptionStrategy(IRON_CONDOR)
    assert len(s) == 4 and s.breakevens() == [86.0, 114.0]
    assert "legs" in str(inspect.signature(OptionStrategy))
