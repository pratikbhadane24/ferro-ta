"""
ferro_ta.analysis.futures — Futures and forward-curve analytics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import annualized_basis as _rust_annualized_basis
from ferro_ta._ferro_ta import (
    back_adjusted_continuous_contract as _rust_back_adjusted,
)
from ferro_ta._ferro_ta import calendar_spreads as _rust_calendar_spreads
from ferro_ta._ferro_ta import carry_spread as _rust_carry_spread
from ferro_ta._ferro_ta import curve_slope as _rust_curve_slope
from ferro_ta._ferro_ta import curve_summary as _rust_curve_summary
from ferro_ta._ferro_ta import futures_basis as _rust_basis
from ferro_ta._ferro_ta import implied_carry_rate as _rust_implied_carry_rate
from ferro_ta._ferro_ta import parity_gap as _rust_parity_gap
from ferro_ta._ferro_ta import (
    ratio_adjusted_continuous_contract as _rust_ratio_adjusted,
)
from ferro_ta._ferro_ta import roll_yield as _rust_roll_yield
from ferro_ta._ferro_ta import synthetic_forward as _rust_synthetic_forward
from ferro_ta._ferro_ta import synthetic_spot as _rust_synthetic_spot
from ferro_ta._ferro_ta import weighted_continuous_contract as _rust_weighted
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error

__all__ = [
    "CurveSummary",
    "synthetic_forward",
    "synthetic_spot",
    "parity_gap",
    "basis",
    "annualized_basis",
    "implied_carry_rate",
    "carry_spread",
    "weighted_continuous_contract",
    "back_adjusted_continuous_contract",
    "ratio_adjusted_continuous_contract",
    "roll_yield",
    "calendar_spreads",
    "curve_slope",
    "curve_summary",
]


@dataclass(frozen=True)
class CurveSummary:
    front_basis: float
    average_basis: float
    slope: float
    is_contango: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "front_basis": self.front_basis,
            "average_basis": self.average_basis,
            "slope": self.slope,
            "is_contango": self.is_contango,
        }


def synthetic_forward(
    call_price: float,
    put_price: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
) -> float:
    return float(
        _rust_synthetic_forward(
            float(call_price),
            float(put_price),
            float(strike),
            float(rate),
            float(time_to_expiry),
        )
    )


def synthetic_spot(
    call_price: float,
    put_price: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    *,
    carry: float = 0.0,
) -> float:
    return float(
        _rust_synthetic_spot(
            float(call_price),
            float(put_price),
            float(strike),
            float(rate),
            float(time_to_expiry),
            float(carry),
        )
    )


def parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    *,
    carry: float = 0.0,
) -> float:
    return float(
        _rust_parity_gap(
            float(call_price),
            float(put_price),
            float(spot),
            float(strike),
            float(rate),
            float(time_to_expiry),
            float(carry),
        )
    )


def basis(spot: float, future: float) -> float:
    return float(_rust_basis(float(spot), float(future)))


def annualized_basis(spot: float, future: float, time_to_expiry: float) -> float:
    return float(
        _rust_annualized_basis(float(spot), float(future), float(time_to_expiry))
    )


def implied_carry_rate(spot: float, future: float, time_to_expiry: float) -> float:
    return float(
        _rust_implied_carry_rate(float(spot), float(future), float(time_to_expiry))
    )


def carry_spread(
    spot: float, future: float, rate: float, time_to_expiry: float
) -> float:
    return float(
        _rust_carry_spread(
            float(spot), float(future), float(rate), float(time_to_expiry)
        )
    )


def weighted_continuous_contract(
    front: ArrayLike,
    next_contract: ArrayLike,
    next_weights: ArrayLike,
) -> NDArray[np.float64]:
    try:
        return np.asarray(
            _rust_weighted(
                _to_f64(front), _to_f64(next_contract), _to_f64(next_weights)
            ),
            dtype=np.float64,
        )
    except ValueError as err:
        _normalize_rust_error(err)


def back_adjusted_continuous_contract(
    front: ArrayLike,
    next_contract: ArrayLike,
    next_weights: ArrayLike,
) -> NDArray[np.float64]:
    try:
        return np.asarray(
            _rust_back_adjusted(
                _to_f64(front), _to_f64(next_contract), _to_f64(next_weights)
            ),
            dtype=np.float64,
        )
    except ValueError as err:
        _normalize_rust_error(err)


def ratio_adjusted_continuous_contract(
    front: ArrayLike,
    next_contract: ArrayLike,
    next_weights: ArrayLike,
) -> NDArray[np.float64]:
    try:
        return np.asarray(
            _rust_ratio_adjusted(
                _to_f64(front), _to_f64(next_contract), _to_f64(next_weights)
            ),
            dtype=np.float64,
        )
    except ValueError as err:
        _normalize_rust_error(err)


def roll_yield(front_price: float, next_price: float, time_to_expiry: float) -> float:
    return float(
        _rust_roll_yield(float(front_price), float(next_price), float(time_to_expiry))
    )


def calendar_spreads(futures_prices: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(_rust_calendar_spreads(_to_f64(futures_prices)), dtype=np.float64)


def curve_slope(tenors: ArrayLike, futures_prices: ArrayLike) -> float:
    try:
        return float(_rust_curve_slope(_to_f64(tenors), _to_f64(futures_prices)))
    except ValueError as err:
        _normalize_rust_error(err)


def curve_summary(
    spot: float, tenors: ArrayLike, futures_prices: ArrayLike
) -> CurveSummary:
    try:
        front_basis, average_basis, slope, is_contango = _rust_curve_summary(
            float(spot), _to_f64(tenors), _to_f64(futures_prices)
        )
    except ValueError as err:
        _normalize_rust_error(err)
    return CurveSummary(front_basis, average_basis, slope, is_contango)
