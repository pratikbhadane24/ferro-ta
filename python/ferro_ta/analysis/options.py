"""
ferro_ta.analysis.options — Rust-backed derivatives analytics for options.

This module preserves the legacy IV-series helpers and expands them with
pricing, Greeks, implied-volatility inversion, smile analytics, and strike
selection helpers suitable for research and simulation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import (
    black76_price as _rust_black76_price,
)
from ferro_ta._ferro_ta import (
    black76_price_batch as _rust_black76_price_batch,
)
from ferro_ta._ferro_ta import (
    bsm_price as _rust_bsm_price,
)
from ferro_ta._ferro_ta import (
    bsm_price_batch as _rust_bsm_price_batch,
)
from ferro_ta._ferro_ta import (
    implied_volatility as _rust_implied_volatility,
)
from ferro_ta._ferro_ta import (
    implied_volatility_batch as _rust_implied_volatility_batch,
)
from ferro_ta._ferro_ta import (
    iv_percentile as _rust_iv_percentile,
)
from ferro_ta._ferro_ta import (
    iv_rank as _rust_iv_rank,
)
from ferro_ta._ferro_ta import (
    iv_zscore as _rust_iv_zscore,
)
from ferro_ta._ferro_ta import (
    moneyness_labels as _rust_moneyness_labels,
)
from ferro_ta._ferro_ta import (
    option_greeks as _rust_option_greeks,
)
from ferro_ta._ferro_ta import (
    option_greeks_batch as _rust_option_greeks_batch,
)
from ferro_ta._ferro_ta import (
    select_strike_delta as _rust_select_strike_delta,
)
from ferro_ta._ferro_ta import (
    select_strike_offset as _rust_select_strike_offset,
)
from ferro_ta._ferro_ta import (
    smile_metrics as _rust_smile_metrics,
)
from ferro_ta._ferro_ta import (
    term_structure_slope as _rust_term_structure_slope,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import (
    FerroTAInputError,
    FerroTAValueError,
    _normalize_rust_error,
)

ScalarOrArray: TypeAlias = float | NDArray[np.float64]

__all__ = [
    "OptionGreeks",
    "SmileMetrics",
    "black_scholes_price",
    "black_76_price",
    "option_price",
    "greeks",
    "implied_volatility",
    "smile_metrics",
    "term_structure_slope",
    "label_moneyness",
    "select_strike",
    "iv_rank",
    "iv_percentile",
    "iv_zscore",
]


@dataclass(frozen=True)
class OptionGreeks:
    """Container for first-order Greeks."""

    delta: ScalarOrArray
    gamma: ScalarOrArray
    vega: ScalarOrArray
    theta: ScalarOrArray
    rho: ScalarOrArray

    def to_dict(self) -> dict[str, ScalarOrArray]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


@dataclass(frozen=True)
class SmileMetrics:
    """Summary metrics for a single smile slice."""

    atm_iv: float
    risk_reversal_25d: float
    butterfly_25d: float
    skew_slope: float
    convexity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "atm_iv": self.atm_iv,
            "risk_reversal_25d": self.risk_reversal_25d,
            "butterfly_25d": self.butterfly_25d,
            "skew_slope": self.skew_slope,
            "convexity": self.convexity,
        }


def _validate_option_type(option_type: str) -> str:
    value = option_type.lower()
    if value not in {"call", "put"}:
        raise FerroTAValueError("option_type must be 'call' or 'put'.")
    return value


def _validate_model(model: str) -> str:
    value = model.lower()
    aliases = {
        "bsm": "bsm",
        "black_scholes": "bsm",
        "black-scholes": "bsm",
        "blackscholes": "bsm",
        "black76": "black76",
        "black_76": "black76",
        "black-76": "black76",
    }
    if value not in aliases:
        raise FerroTAValueError(
            "model must be one of 'bsm', 'black_scholes', or 'black76'."
        )
    return aliases[value]


def _coerce_1d(data: ArrayLike | float, *, name: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim > 1:
        raise FerroTAInputError(f"{name} must be a scalar or 1-D array.")
    return np.ascontiguousarray(arr.reshape(-1)), arr.ndim == 0


def _broadcast_inputs(
    **kwargs: ArrayLike | float,
) -> tuple[dict[str, np.ndarray], bool]:
    arrays: dict[str, np.ndarray] = {}
    scalar_flags: list[bool] = []
    for name, value in kwargs.items():
        arr, is_scalar = _coerce_1d(value, name=name)
        arrays[name] = arr
        scalar_flags.append(is_scalar)
    try:
        broadcast = np.broadcast_arrays(*arrays.values())
    except ValueError as err:
        raise FerroTAInputError(
            f"Inputs could not be broadcast together: {', '.join(arrays.keys())}"
        ) from err
    out = {
        name: np.ascontiguousarray(arr, dtype=np.float64).reshape(-1)
        for name, arr in zip(arrays.keys(), broadcast)
    }
    return out, all(scalar_flags)


def _result_or_scalar(result: np.ndarray, scalar_mode: bool) -> ScalarOrArray:
    return float(result[0]) if scalar_mode else result


def iv_rank(iv_series: ArrayLike, window: int = 252) -> NDArray[np.float64]:
    """Compute rolling IV rank in Rust while preserving the legacy API."""
    try:
        arr = _to_f64(iv_series)
    except ValueError as err:
        raise FerroTAInputError(str(err)) from err
    if len(arr) == 0:
        raise FerroTAInputError("iv_series must not be empty.")
    if window < 1:
        raise FerroTAValueError(f"window must be >= 1, got {window}.")
    try:
        return np.asarray(_rust_iv_rank(arr, int(window)), dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


def iv_percentile(iv_series: ArrayLike, window: int = 252) -> NDArray[np.float64]:
    """Compute rolling IV percentile in Rust while preserving the legacy API."""
    try:
        arr = _to_f64(iv_series)
    except ValueError as err:
        raise FerroTAInputError(str(err)) from err
    if len(arr) == 0:
        raise FerroTAInputError("iv_series must not be empty.")
    if window < 1:
        raise FerroTAValueError(f"window must be >= 1, got {window}.")
    try:
        return np.asarray(_rust_iv_percentile(arr, int(window)), dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


def iv_zscore(iv_series: ArrayLike, window: int = 252) -> NDArray[np.float64]:
    """Compute rolling IV z-score in Rust while preserving the legacy API."""
    try:
        arr = _to_f64(iv_series)
    except ValueError as err:
        raise FerroTAInputError(str(err)) from err
    if len(arr) == 0:
        raise FerroTAInputError("iv_series must not be empty.")
    if window < 1:
        raise FerroTAValueError(f"window must be >= 1, got {window}.")
    try:
        return np.asarray(_rust_iv_zscore(arr, int(window)), dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


def black_scholes_price(
    spot: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
    dividend_yield: ArrayLike | float = 0.0,
) -> ScalarOrArray:
    """Price options under Black-Scholes-Merton."""
    option_type = _validate_option_type(option_type)
    arrays, scalar_mode = _broadcast_inputs(
        spot=spot,
        strike=strike,
        rate=rate,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    try:
        if scalar_mode:
            return float(
                _rust_bsm_price(
                    float(arrays["spot"][0]),
                    float(arrays["strike"][0]),
                    float(arrays["rate"][0]),
                    float(arrays["time_to_expiry"][0]),
                    float(arrays["volatility"][0]),
                    option_type,
                    float(arrays["dividend_yield"][0]),
                )
            )
        out = _rust_bsm_price_batch(
            arrays["spot"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            arrays["volatility"],
            arrays["dividend_yield"],
            option_type,
        )
        return np.asarray(out, dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


def black_76_price(
    forward: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
) -> ScalarOrArray:
    """Price options under Black-76."""
    option_type = _validate_option_type(option_type)
    arrays, scalar_mode = _broadcast_inputs(
        forward=forward,
        strike=strike,
        rate=rate,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
    )
    try:
        if scalar_mode:
            return float(
                _rust_black76_price(
                    float(arrays["forward"][0]),
                    float(arrays["strike"][0]),
                    float(arrays["rate"][0]),
                    float(arrays["time_to_expiry"][0]),
                    float(arrays["volatility"][0]),
                    option_type,
                )
            )
        out = _rust_black76_price_batch(
            arrays["forward"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            arrays["volatility"],
            option_type,
        )
        return np.asarray(out, dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


def option_price(
    underlying: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
    model: str = "bsm",
    carry: ArrayLike | float = 0.0,
) -> ScalarOrArray:
    """Model-dispatched option price helper."""
    model = _validate_model(model)
    if model == "black76":
        return black_76_price(
            underlying,
            strike,
            rate,
            time_to_expiry,
            volatility,
            option_type=option_type,
        )
    return black_scholes_price(
        underlying,
        strike,
        rate,
        time_to_expiry,
        volatility,
        option_type=option_type,
        dividend_yield=carry,
    )


def greeks(
    underlying: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
    model: str = "bsm",
    carry: ArrayLike | float = 0.0,
) -> OptionGreeks:
    """Return delta, gamma, vega, theta, and rho."""
    option_type = _validate_option_type(option_type)
    model = _validate_model(model)
    arrays, scalar_mode = _broadcast_inputs(
        underlying=underlying,
        strike=strike,
        rate=rate,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        carry=carry,
    )
    try:
        if scalar_mode:
            delta, gamma, vega, theta, rho = _rust_option_greeks(
                float(arrays["underlying"][0]),
                float(arrays["strike"][0]),
                float(arrays["rate"][0]),
                float(arrays["time_to_expiry"][0]),
                float(arrays["volatility"][0]),
                option_type,
                model,
                float(arrays["carry"][0]),
            )
            return OptionGreeks(delta, gamma, vega, theta, rho)

        delta, gamma, vega, theta, rho = _rust_option_greeks_batch(
            arrays["underlying"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            arrays["volatility"],
            option_type,
            model,
            arrays["carry"],
        )
        return OptionGreeks(
            np.asarray(delta, dtype=np.float64),
            np.asarray(gamma, dtype=np.float64),
            np.asarray(vega, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
            np.asarray(rho, dtype=np.float64),
        )
    except ValueError as err:
        _normalize_rust_error(err)


def implied_volatility(
    price: ArrayLike | float,
    underlying: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    *,
    option_type: str = "call",
    model: str = "bsm",
    carry: ArrayLike | float = 0.0,
    initial_guess: ArrayLike | float = 0.2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> ScalarOrArray:
    """Invert option prices to implied volatility."""
    option_type = _validate_option_type(option_type)
    model = _validate_model(model)
    arrays, scalar_mode = _broadcast_inputs(
        price=price,
        underlying=underlying,
        strike=strike,
        rate=rate,
        time_to_expiry=time_to_expiry,
        carry=carry,
        initial_guess=initial_guess,
    )
    try:
        if scalar_mode:
            return float(
                _rust_implied_volatility(
                    float(arrays["price"][0]),
                    float(arrays["underlying"][0]),
                    float(arrays["strike"][0]),
                    float(arrays["rate"][0]),
                    float(arrays["time_to_expiry"][0]),
                    option_type,
                    model,
                    float(arrays["carry"][0]),
                    float(arrays["initial_guess"][0]),
                    float(tolerance),
                    int(max_iterations),
                )
            )
        out = _rust_implied_volatility_batch(
            arrays["price"],
            arrays["underlying"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            option_type,
            model,
            arrays["carry"],
            arrays["initial_guess"],
            float(tolerance),
            int(max_iterations),
        )
        return np.asarray(out, dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


def smile_metrics(
    strikes: ArrayLike,
    vols: ArrayLike,
    reference_price: float,
    time_to_expiry: float,
    *,
    model: str = "bsm",
    rate: float = 0.0,
    carry: float = 0.0,
) -> SmileMetrics:
    """Compute ATM IV, 25-delta RR/BF, skew slope, and convexity."""
    model = _validate_model(model)
    strikes_arr = _to_f64(strikes)
    vols_arr = _to_f64(vols)
    order = np.argsort(strikes_arr)
    strikes_arr = strikes_arr[order]
    vols_arr = vols_arr[order]
    try:
        atm_iv, rr25, bf25, slope, convexity = _rust_smile_metrics(
            strikes_arr,
            vols_arr,
            float(reference_price),
            float(time_to_expiry),
            model,
            float(rate),
            float(carry),
        )
    except ValueError as err:
        _normalize_rust_error(err)
    return SmileMetrics(atm_iv, rr25, bf25, slope, convexity)


def term_structure_slope(tenors: ArrayLike, atm_ivs: ArrayLike) -> float:
    """Slope of ATM IV against tenor."""
    try:
        return float(_rust_term_structure_slope(_to_f64(tenors), _to_f64(atm_ivs)))
    except ValueError as err:
        _normalize_rust_error(err)


def label_moneyness(
    strikes: ArrayLike,
    reference_price: float,
    *,
    option_type: str = "call",
) -> NDArray[np.object_]:
    """Label strikes as ``ITM``, ``ATM``, or ``OTM``."""
    option_type = _validate_option_type(option_type)
    try:
        codes = np.asarray(
            _rust_moneyness_labels(
                _to_f64(strikes), float(reference_price), option_type
            ),
            dtype=np.int8,
        )
    except ValueError as err:
        _normalize_rust_error(err)
    mapping = np.array(["OTM", "ATM", "ITM"], dtype=object)
    return mapping[codes + 1]


def _parse_selector_steps(selector: str) -> int:
    suffix = selector[3:]
    if suffix == "":
        return 1
    try:
        return int(suffix)
    except ValueError as err:
        raise FerroTAValueError(
            f"Could not parse strike selector '{selector}'. Expected forms like ATM, ITM1, OTM2."
        ) from err


def select_strike(
    strikes: ArrayLike,
    reference_price: float,
    *,
    option_type: str = "call",
    selector: str = "ATM",
    delta_target: float | None = None,
    volatilities: ArrayLike | None = None,
    time_to_expiry: float | None = None,
    model: str = "bsm",
    rate: float = 0.0,
    carry: float = 0.0,
) -> float | None:
    """Select a strike by ATM/ITM/OTM offset or delta target."""
    option_type = _validate_option_type(option_type)
    model = _validate_model(model)
    strikes_arr = _to_f64(strikes)

    if len(strikes_arr) == 0:
        raise FerroTAInputError("strikes must not be empty.")

    selector_norm = selector.strip().upper()
    if delta_target is None and selector_norm.startswith("DELTA"):
        try:
            delta_target = float(selector_norm.replace("DELTA", ""))
        except ValueError as err:
            raise FerroTAValueError(
                f"Could not parse delta selector '{selector}'. Example: selector='DELTA0.25'."
            ) from err

    if delta_target is not None:
        if volatilities is None or time_to_expiry is None:
            raise FerroTAValueError(
                "Delta-based strike selection requires volatilities and time_to_expiry."
            )
        vols_arr = _to_f64(volatilities)
        if len(vols_arr) != len(strikes_arr):
            raise FerroTAInputError(
                "strikes and volatilities must have the same length."
            )
        order = np.argsort(strikes_arr)
        strikes_arr = strikes_arr[order]
        vols_arr = vols_arr[order]
        try:
            strike = _rust_select_strike_delta(
                strikes_arr,
                vols_arr,
                float(reference_price),
                float(time_to_expiry),
                float(delta_target),
                option_type,
                model,
                float(rate),
                float(carry),
            )
        except ValueError as err:
            _normalize_rust_error(err)
        return None if strike is None else float(strike)

    order = np.argsort(strikes_arr)
    sorted_strikes = strikes_arr[order]
    if selector_norm == "ATM":
        offset = 0
    elif selector_norm.startswith("ITM"):
        steps = _parse_selector_steps(selector_norm)
        offset = -steps if option_type == "call" else steps
    elif selector_norm.startswith("OTM"):
        steps = _parse_selector_steps(selector_norm)
        offset = steps if option_type == "call" else -steps
    else:
        raise FerroTAValueError(
            f"Unsupported selector '{selector}'. Use ATM, ITM<n>, OTM<n>, or DELTA<x>."
        )

    try:
        strike = _rust_select_strike_offset(
            sorted_strikes, float(reference_price), int(offset)
        )
    except ValueError as err:
        _normalize_rust_error(err)
    return None if strike is None else float(strike)
