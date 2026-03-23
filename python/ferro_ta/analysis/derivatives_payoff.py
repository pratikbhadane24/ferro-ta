"""
ferro_ta.analysis.derivatives_payoff — Multi-leg payoff and Greeks aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta.analysis.options import OptionGreeks
from ferro_ta.analysis.options import greeks as option_greeks
from ferro_ta.analysis.options_strategy import DerivativesStrategy, StrategyLeg
from ferro_ta.core.exceptions import FerroTAInputError, FerroTAValueError

__all__ = [
    "PayoffLeg",
    "option_leg_payoff",
    "futures_leg_payoff",
    "strategy_payoff",
    "aggregate_greeks",
]


@dataclass(frozen=True)
class PayoffLeg:
    instrument: str
    side: str
    quantity: float = 1.0
    option_type: str | None = None
    strike: float | None = None
    premium: float = 0.0
    entry_price: float | None = None
    volatility: float | None = None
    time_to_expiry: float | None = None
    rate: float = 0.0
    carry: float = 0.0
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.instrument not in {"option", "future"}:
            raise FerroTAValueError("instrument must be 'option' or 'future'.")
        if self.side not in {"long", "short"}:
            raise FerroTAValueError("side must be 'long' or 'short'.")
        if self.instrument == "option":
            if self.option_type not in {"call", "put"}:
                raise FerroTAValueError(
                    "option legs require option_type='call' or 'put'."
                )
            if self.strike is None:
                raise FerroTAValueError("option legs require strike.")
        if self.instrument == "future" and self.entry_price is None:
            raise FerroTAValueError("future legs require entry_price.")


def _side_sign(side: str) -> float:
    return 1.0 if side == "long" else -1.0


def _coerce_spot_grid(spot_grid: ArrayLike) -> NDArray[np.float64]:
    grid = np.asarray(spot_grid, dtype=np.float64)
    if grid.ndim != 1:
        raise FerroTAInputError("spot_grid must be a 1-D array.")
    return np.ascontiguousarray(grid)


def option_leg_payoff(
    spot_grid: ArrayLike,
    *,
    strike: float,
    premium: float = 0.0,
    option_type: str = "call",
    side: str = "long",
    quantity: float = 1.0,
    multiplier: float = 1.0,
) -> NDArray[np.float64]:
    """Expiry payoff for a single option leg."""
    grid = _coerce_spot_grid(spot_grid)
    sign = _side_sign(side) * float(quantity) * float(multiplier)
    if option_type == "call":
        intrinsic = np.maximum(grid - float(strike), 0.0)
    elif option_type == "put":
        intrinsic = np.maximum(float(strike) - grid, 0.0)
    else:
        raise FerroTAValueError("option_type must be 'call' or 'put'.")
    return sign * (intrinsic - float(premium))


def futures_leg_payoff(
    spot_grid: ArrayLike,
    *,
    entry_price: float,
    side: str = "long",
    quantity: float = 1.0,
    multiplier: float = 1.0,
) -> NDArray[np.float64]:
    """P/L profile for a futures leg."""
    grid = _coerce_spot_grid(spot_grid)
    sign = _side_sign(side) * float(quantity) * float(multiplier)
    return sign * (grid - float(entry_price))


def _mapping_to_leg(mapping: Mapping[str, Any]) -> PayoffLeg:
    return PayoffLeg(**mapping)


def _strategy_leg_to_payoff_leg(leg: StrategyLeg) -> PayoffLeg:
    return PayoffLeg(
        instrument=leg.instrument,
        side=leg.side,
        quantity=float(leg.quantity),
        option_type=leg.option_type,
        strike=leg.strike_selector.explicit_strike,
    )


def _normalize_legs(
    legs: Sequence[PayoffLeg | Mapping[str, Any]] | None = None,
    *,
    strategy: DerivativesStrategy | None = None,
) -> tuple[PayoffLeg, ...]:
    if strategy is not None:
        return tuple(_strategy_leg_to_payoff_leg(leg) for leg in strategy.legs)
    if legs is None:
        raise FerroTAInputError("Provide either legs or strategy.")
    normalized: list[PayoffLeg] = []
    for leg in legs:
        normalized.append(leg if isinstance(leg, PayoffLeg) else _mapping_to_leg(leg))
    return tuple(normalized)


def strategy_payoff(
    spot_grid: ArrayLike,
    *,
    legs: Sequence[PayoffLeg | Mapping[str, Any]] | None = None,
    strategy: DerivativesStrategy | None = None,
) -> NDArray[np.float64]:
    """Aggregate expiry payoff across option and futures legs."""
    grid = _coerce_spot_grid(spot_grid)
    normalized = _normalize_legs(legs, strategy=strategy)
    total = np.zeros_like(grid)
    for leg in normalized:
        if leg.instrument == "option":
            if leg.strike is None:
                raise FerroTAValueError("Option payoff legs require strike.")
            total += option_leg_payoff(
                grid,
                strike=float(leg.strike),
                premium=float(leg.premium),
                option_type=str(leg.option_type),
                side=str(leg.side),
                quantity=float(leg.quantity),
                multiplier=float(leg.multiplier),
            )
        else:
            if leg.entry_price is None:
                raise FerroTAValueError("Futures payoff legs require entry_price.")
            total += futures_leg_payoff(
                grid,
                entry_price=float(leg.entry_price),
                side=str(leg.side),
                quantity=float(leg.quantity),
                multiplier=float(leg.multiplier),
            )
    return total


def aggregate_greeks(
    spot: float,
    *,
    legs: Sequence[PayoffLeg | Mapping[str, Any]] | None = None,
    strategy: DerivativesStrategy | None = None,
) -> OptionGreeks:
    """Aggregate Greeks across option and futures legs."""
    normalized = _normalize_legs(legs, strategy=strategy)
    totals = {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "rho": 0.0,
    }
    for leg in normalized:
        leg_sign = _side_sign(leg.side) * float(leg.quantity) * float(leg.multiplier)
        if leg.instrument == "future":
            totals["delta"] += leg_sign
            continue
        if leg.strike is None or leg.volatility is None or leg.time_to_expiry is None:
            raise FerroTAValueError(
                "Option legs require strike, volatility, and time_to_expiry for Greeks aggregation."
            )
        leg_greeks = option_greeks(
            float(spot),
            float(leg.strike),
            float(leg.rate),
            float(leg.time_to_expiry),
            float(leg.volatility),
            option_type=str(leg.option_type),
            model="bsm",
            carry=float(leg.carry),
        )
        totals["delta"] += leg_sign * float(leg_greeks.delta)
        totals["gamma"] += leg_sign * float(leg_greeks.gamma)
        totals["vega"] += leg_sign * float(leg_greeks.vega)
        totals["theta"] += leg_sign * float(leg_greeks.theta)
        totals["rho"] += leg_sign * float(leg_greeks.rho)

    return OptionGreeks(
        totals["delta"],
        totals["gamma"],
        totals["vega"],
        totals["theta"],
        totals["rho"],
    )
