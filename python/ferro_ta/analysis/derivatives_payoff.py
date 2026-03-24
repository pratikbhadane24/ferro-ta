"""
ferro_ta.analysis.derivatives_payoff — Multi-leg payoff and Greeks aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import aggregate_greeks_legs as _rust_aggregate_greeks_legs
from ferro_ta._ferro_ta import strategy_payoff_dense as _rust_strategy_payoff_dense
from ferro_ta._ferro_ta import strategy_payoff_legs as _rust_strategy_payoff_legs
from ferro_ta.analysis.options import OptionGreeks
from ferro_ta.analysis.options_strategy import DerivativesStrategy, StrategyLeg
from ferro_ta.core.exceptions import (
    FerroTAInputError,
    FerroTAValueError,
    _normalize_rust_error,
)

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
    _side_sign(side)
    if option_type not in {"call", "put"}:
        raise FerroTAValueError("option_type must be 'call' or 'put'.")
    return np.asarray(
        _rust_strategy_payoff_dense(
            grid,
            np.array([0], dtype=np.int64),  # option
            np.array([1 if side == "long" else -1], dtype=np.int64),
            np.array([1 if option_type == "call" else -1], dtype=np.int64),
            np.array([float(strike)], dtype=np.float64),
            np.array([float(premium)], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([float(quantity)], dtype=np.float64),
            np.array([float(multiplier)], dtype=np.float64),
        ),
        dtype=np.float64,
    )


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
    _side_sign(side)
    return np.asarray(
        _rust_strategy_payoff_dense(
            grid,
            np.array([1], dtype=np.int64),  # future
            np.array([1 if side == "long" else -1], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([float(entry_price)], dtype=np.float64),
            np.array([float(quantity)], dtype=np.float64),
            np.array([float(multiplier)], dtype=np.float64),
        ),
        dtype=np.float64,
    )


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
    if len(normalized) == 0:
        return np.zeros_like(grid)

    try:
        return np.asarray(
            _rust_strategy_payoff_legs(grid, normalized), dtype=np.float64
        )
    except ValueError as err:
        _normalize_rust_error(err)


def aggregate_greeks(
    spot: float,
    *,
    legs: Sequence[PayoffLeg | Mapping[str, Any]] | None = None,
    strategy: DerivativesStrategy | None = None,
) -> OptionGreeks:
    """Aggregate Greeks across option and futures legs."""
    normalized = _normalize_legs(legs, strategy=strategy)
    if len(normalized) == 0:
        return OptionGreeks(0.0, 0.0, 0.0, 0.0, 0.0)

    try:
        delta, gamma, vega, theta, rho = _rust_aggregate_greeks_legs(
            float(spot), normalized
        )
    except ValueError as err:
        _normalize_rust_error(err)

    return OptionGreeks(
        float(delta),
        float(gamma),
        float(vega),
        float(theta),
        float(rho),
    )
