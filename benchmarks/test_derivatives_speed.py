"""
Derivatives benchmark hooks.

These are intentionally optional and skip when `py_vollib` is unavailable.
Run with:

    uv run pytest benchmarks/test_derivatives_speed.py --benchmark-only -v
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ferro_ta.analysis.options import implied_volatility, option_price


def _sample_chain(n: int = 1000) -> tuple[np.ndarray, ...]:
    spot = np.linspace(90.0, 110.0, n)
    strike = np.full(n, 100.0)
    rate = np.full(n, 0.02)
    time_to_expiry = np.full(n, 0.5)
    volatility = np.full(n, 0.2)
    return spot, strike, rate, time_to_expiry, volatility


def test_ferro_ta_option_price_speed(benchmark):
    spot, strike, rate, time_to_expiry, volatility = _sample_chain()

    benchmark.pedantic(
        lambda: option_price(
            spot,
            strike,
            rate,
            time_to_expiry,
            volatility,
            option_type="call",
            model="bsm",
        ),
        iterations=5,
        rounds=20,
        warmup_rounds=2,
    )


def test_ferro_ta_implied_vol_speed(benchmark):
    spot, strike, rate, time_to_expiry, volatility = _sample_chain()
    prices = option_price(
        spot,
        strike,
        rate,
        time_to_expiry,
        volatility,
        option_type="call",
        model="bsm",
    )

    benchmark.pedantic(
        lambda: implied_volatility(
            prices,
            spot,
            strike,
            rate,
            time_to_expiry,
            option_type="call",
            model="bsm",
        ),
        iterations=5,
        rounds=20,
        warmup_rounds=2,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("py_vollib") is None,
    reason="py_vollib is optional",
)
def test_py_vollib_scalar_loop_baseline(benchmark):
    from py_vollib.black_scholes_merton import black_scholes_merton as py_vollib_bsm
    from py_vollib.black_scholes_merton.implied_volatility import (
        implied_volatility as py_vollib_iv,
    )

    spot, strike, rate, time_to_expiry, volatility = _sample_chain(250)
    prices = [
        py_vollib_bsm("c", float(s), float(k), float(t), float(r), float(vol), 0.0)
        for s, k, r, t, vol in zip(spot, strike, rate, time_to_expiry, volatility)
    ]

    benchmark.pedantic(
        lambda: [
            py_vollib_iv(
                float(price),
                "c",
                float(s),
                float(k),
                float(t),
                float(r),
                0.0,
            )
            for price, s, k, r, t in zip(prices, spot, strike, rate, time_to_expiry)
        ],
        iterations=3,
        rounds=10,
        warmup_rounds=1,
    )
