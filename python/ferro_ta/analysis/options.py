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
    expected_move as _rust_expected_move,
)
from ferro_ta._ferro_ta import (
    extended_greeks as _rust_extended_greeks,
)
from ferro_ta._ferro_ta import (
    extended_greeks_batch as _rust_extended_greeks_batch,
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
    put_call_parity_deviation as _rust_put_call_parity_deviation,
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
    "ExtendedGreeks",
    "SmileMetrics",
    "VolCone",
    "black_scholes_price",
    "black_76_price",
    "option_price",
    "greeks",
    "extended_greeks",
    "implied_volatility",
    "smile_metrics",
    "term_structure_slope",
    "label_moneyness",
    "select_strike",
    "iv_rank",
    "iv_percentile",
    "iv_zscore",
    "put_call_parity_deviation",
    "expected_move",
    "digital_option_price",
    "digital_option_greeks",
    "american_option_price",
    "early_exercise_premium",
    "close_to_close_vol",
    "parkinson_vol",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "yang_zhang_vol",
    "vol_cone",
]


@dataclass(frozen=True)
class ExtendedGreeks:
    """Container for second-order and cross Greeks."""

    vanna: ScalarOrArray
    volga: ScalarOrArray
    charm: ScalarOrArray
    speed: ScalarOrArray
    color: ScalarOrArray

    def to_dict(self) -> dict[str, ScalarOrArray]:
        return {
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "speed": self.speed,
            "color": self.color,
        }


@dataclass(frozen=True)
class VolCone:
    """Historical realized vol distribution across window lengths."""

    windows: NDArray[np.float64]
    min: NDArray[np.float64]
    p25: NDArray[np.float64]
    median: NDArray[np.float64]
    p75: NDArray[np.float64]
    max: NDArray[np.float64]

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        return {
            "windows": self.windows,
            "min": self.min,
            "p25": self.p25,
            "median": self.median,
            "p75": self.p75,
            "max": self.max,
        }


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


def extended_greeks(
    underlying: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
    model: str = "bsm",
    carry: ArrayLike | float = 0.0,
) -> ExtendedGreeks:
    """Return vanna, volga, charm, speed, and color (second-order / cross Greeks).

    All Greeks are computed via closed-form BSM formulas.  Black-76 is not
    yet supported and returns NaN for all five values.

    Parameters
    ----------
    underlying:
        Current underlying (spot) price.
    strike:
        Option strike price.
    rate:
        Risk-free rate (annualised, decimal — e.g. ``0.05`` for 5 %).
    time_to_expiry:
        Time to expiry in years.
    volatility:
        Implied volatility (annualised, decimal).
    option_type:
        ``"call"`` (default) or ``"put"``.
    model:
        ``"bsm"`` (default).  ``"black76"`` returns NaN for all fields.
    carry:
        Continuous carry / dividend yield (annualised, decimal).  Default 0.

    Returns
    -------
    ExtendedGreeks
        Named tuple with fields:

        - **vanna** — ∂Δ/∂σ: sensitivity of delta to a change in vol.
        - **volga** — ∂²V/∂σ² (vomma): sensitivity of vega to a change in vol.
        - **charm** — ∂Δ/∂t: daily rate of change in delta (theta of delta).
        - **speed** — ∂Γ/∂S: rate of change in gamma with respect to spot.
        - **color** — ∂Γ/∂t: daily rate of change in gamma.

    Notes
    -----
    Inputs may be scalars or broadcastable arrays.  When arrays are supplied
    each field of the returned :class:`ExtendedGreeks` is an ``NDArray``.

    Closed-form expressions (BSM, zero-carry)::

        vanna = -e^{-qT} · φ(d₁) · d₂ / σ
        volga = S · e^{-qT} · φ(d₁) · √T · d₁ · d₂ / σ
        charm = -e^{-qT} · φ(d₁) · [2(r-q)T - d₂·σ·√T] / (2T·σ·√T)
        speed = -Γ/S · (d₁/(σ√T) + 1)
        color = -Γ · [r-q + d₁·σ/(2√T) + (2(r-q)T - d₂·σ√T)·d₁/(2T·σ√T)]
    """
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
            vanna, volga, charm, speed, color = _rust_extended_greeks(
                float(arrays["underlying"][0]),
                float(arrays["strike"][0]),
                float(arrays["rate"][0]),
                float(arrays["time_to_expiry"][0]),
                float(arrays["volatility"][0]),
                option_type,
                model,
                float(arrays["carry"][0]),
            )
            return ExtendedGreeks(vanna, volga, charm, speed, color)

        vanna, volga, charm, speed, color = _rust_extended_greeks_batch(
            arrays["underlying"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            arrays["volatility"],
            option_type,
            model,
            arrays["carry"],
        )
        return ExtendedGreeks(
            np.asarray(vanna, dtype=np.float64),
            np.asarray(volga, dtype=np.float64),
            np.asarray(charm, dtype=np.float64),
            np.asarray(speed, dtype=np.float64),
            np.asarray(color, dtype=np.float64),
        )
    except ValueError as err:
        _normalize_rust_error(err)


def put_call_parity_deviation(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    *,
    carry: float = 0.0,
) -> float:
    """Put-call parity deviation: ``C − P − (S·e^{−q·T} − K·e^{−r·T})``.

    At no-arbitrage the deviation is exactly 0.  A non-zero result indicates
    mispricing, a data error, or a stale quote.

    Parameters
    ----------
    call_price:
        Market or model price of the call option.
    put_price:
        Market or model price of the put option.
    spot:
        Current underlying price.
    strike:
        Common strike price of the call and put.
    rate:
        Risk-free rate (annualised, decimal).
    time_to_expiry:
        Time to expiry in years.
    carry:
        Continuous dividend yield / carry rate (annualised, decimal).

    Returns
    -------
    float
        Signed deviation.  Positive → call is overpriced relative to put;
        negative → put is overpriced relative to call.

    Examples
    --------
    >>> from ferro_ta.analysis.options import option_price, put_call_parity_deviation
    >>> call = option_price(100, 100, 0.05, 1.0, 0.2, option_type="call")
    >>> put  = option_price(100, 100, 0.05, 1.0, 0.2, option_type="put")
    >>> put_call_parity_deviation(call, put, 100, 100, 0.05, 1.0)  # ≈ 0.0
    """
    try:
        return float(
            _rust_put_call_parity_deviation(
                float(call_price),
                float(put_price),
                float(spot),
                float(strike),
                float(rate),
                float(time_to_expiry),
                float(carry),
            )
        )
    except ValueError as err:
        _normalize_rust_error(err)


def expected_move(
    spot: float,
    iv: float,
    days_to_expiry: float,
    trading_days_per_year: float = 252.0,
) -> tuple[float, float]:
    """Expected ±1σ move over *days_to_expiry* calendar days.

    Uses the log-normal approximation::

        upper_move = spot × e^{+σ√(days/trading_days)} − spot
        lower_move = spot × e^{−σ√(days/trading_days)} − spot

    Parameters
    ----------
    spot:
        Current underlying price.
    iv:
        Implied volatility (annualised, decimal — e.g. ``0.20`` for 20 %).
    days_to_expiry:
        Number of calendar days until expiry.
    trading_days_per_year:
        Annualisation factor (default 252).

    Returns
    -------
    tuple[float, float]
        ``(lower_move, upper_move)`` — signed absolute price changes from
        ``spot``.  ``lower_move < 0``, ``upper_move > 0``.

    Notes
    -----
    Because of log-normal skew, ``|upper_move| > |lower_move|``.

    Examples
    --------
    >>> from ferro_ta.analysis.options import expected_move
    >>> lower, upper = expected_move(100.0, 0.20, 30)
    >>> round(upper, 2)
    7.14
    """
    try:
        lower, upper = _rust_expected_move(
            float(spot), float(iv), float(days_to_expiry), float(trading_days_per_year)
        )
        return float(lower), float(upper)
    except ValueError as err:
        _normalize_rust_error(err)


# ---------------------------------------------------------------------------
# Digital options — populated once the Rust bridge is built
# ---------------------------------------------------------------------------


def digital_option_price(
    underlying: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
    digital_type: str = "cash_or_nothing",
    carry: ArrayLike | float = 0.0,
) -> ScalarOrArray:
    """Price a digital (binary) option under BSM.

    Parameters
    ----------
    underlying:
        Current underlying (spot) price.
    strike:
        Option strike price.
    rate:
        Risk-free rate (annualised, decimal).
    time_to_expiry:
        Time to expiry in years.
    volatility:
        Implied volatility (annualised, decimal).
    option_type:
        ``"call"`` (default) or ``"put"``.
    digital_type:
        ``"cash_or_nothing"`` (default) — pays 1 unit of cash if ITM at
        expiry; or ``"asset_or_nothing"`` — pays the underlying asset price.
    carry:
        Continuous carry / dividend yield (annualised, decimal).  Default 0.

    Returns
    -------
    float or NDArray[float64]
        Option price.  Returns a scalar when all inputs are scalars, or an
        array when any input is an array.

    Notes
    -----
    Closed-form BSM formulas::

        Cash-or-nothing call: e^{−rT} · N(d₂)
        Cash-or-nothing put:  e^{−rT} · N(−d₂)
        Asset-or-nothing call: S · e^{−qT} · N(d₁)
        Asset-or-nothing put:  S · e^{−qT} · N(−d₁)

    Put-call parity for cash-or-nothing: call + put = e^{−rT}.
    Put-call parity for asset-or-nothing: call + put = S · e^{−qT}.

    Invalid inputs (non-positive spot/strike, negative time or vol) return NaN.
    """
    from ferro_ta._ferro_ta import digital_price as _rust_digital_price
    from ferro_ta._ferro_ta import digital_price_batch as _rust_digital_price_batch

    option_type = _validate_option_type(option_type)
    digital_type = digital_type.lower().replace("-", "_")
    if digital_type not in {"cash_or_nothing", "asset_or_nothing"}:
        raise FerroTAValueError(
            "digital_type must be 'cash_or_nothing' or 'asset_or_nothing'."
        )
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
            return float(
                _rust_digital_price(
                    float(arrays["underlying"][0]),
                    float(arrays["strike"][0]),
                    float(arrays["rate"][0]),
                    float(arrays["time_to_expiry"][0]),
                    float(arrays["volatility"][0]),
                    option_type,
                    digital_type,
                    float(arrays["carry"][0]),
                )
            )
        out = _rust_digital_price_batch(
            arrays["underlying"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            arrays["volatility"],
            option_type,
            digital_type,
            arrays["carry"],
        )
        return np.asarray(out, dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


def digital_option_greeks(
    underlying: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
    digital_type: str = "cash_or_nothing",
    carry: ArrayLike | float = 0.0,
) -> OptionGreeks:
    """Delta, gamma, and vega for a digital option via numerical bumping.

    Uses central finite differences (spot bump ε = spot × 10⁻³ for delta/gamma;
    vol bump ε = 10⁻³ for vega).  Theta and rho are set to NaN.

    Parameters
    ----------
    underlying, strike, rate, time_to_expiry, volatility, option_type, carry:
        Same as :func:`digital_option_price`.
    digital_type:
        ``"cash_or_nothing"`` (default) or ``"asset_or_nothing"``.

    Returns
    -------
    OptionGreeks
        Named tuple; only ``delta``, ``gamma``, ``vega`` are finite.
        ``theta`` and ``rho`` are NaN.
    """
    from ferro_ta._ferro_ta import digital_greeks as _rust_digital_greeks
    from ferro_ta._ferro_ta import digital_greeks_batch as _rust_digital_greeks_batch

    option_type = _validate_option_type(option_type)
    digital_type = digital_type.lower().replace("-", "_")
    if digital_type not in {"cash_or_nothing", "asset_or_nothing"}:
        raise FerroTAValueError(
            "digital_type must be 'cash_or_nothing' or 'asset_or_nothing'."
        )
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
            delta, gamma, vega = _rust_digital_greeks(
                float(arrays["underlying"][0]),
                float(arrays["strike"][0]),
                float(arrays["rate"][0]),
                float(arrays["time_to_expiry"][0]),
                float(arrays["volatility"][0]),
                option_type,
                digital_type,
                float(arrays["carry"][0]),
            )
            return OptionGreeks(delta, gamma, vega, float("nan"), float("nan"))

        delta, gamma, vega = _rust_digital_greeks_batch(
            arrays["underlying"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            arrays["volatility"],
            option_type,
            digital_type,
            arrays["carry"],
        )
        nan_arr = np.full_like(delta, float("nan"))
        return OptionGreeks(
            np.asarray(delta, dtype=np.float64),
            np.asarray(gamma, dtype=np.float64),
            np.asarray(vega, dtype=np.float64),
            nan_arr,
            nan_arr,
        )
    except ValueError as err:
        _normalize_rust_error(err)


# ---------------------------------------------------------------------------
# American options — populated once the Rust bridge is built
# ---------------------------------------------------------------------------


def american_option_price(
    underlying: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
    carry: ArrayLike | float = 0.0,
) -> ScalarOrArray:
    """American option price using the Barone-Adesi-Whaley (1987) approximation.

    Accurate to within a few basis points for standard equity/index parameters.
    O(1) per evaluation — suitable for batch pricing or calibration.

    Parameters
    ----------
    underlying:
        Current underlying (spot) price.
    strike:
        Option strike price.
    rate:
        Risk-free rate (annualised, decimal).
    time_to_expiry:
        Time to expiry in years.
    volatility:
        Implied volatility (annualised, decimal).
    option_type:
        ``"call"`` (default) or ``"put"``.
    carry:
        Continuous carry / dividend yield (annualised, decimal).  Default 0.
        For calls with ``carry = 0`` (no dividends) early exercise is never
        optimal and the result equals the European BSM price.

    Returns
    -------
    float or NDArray[float64]
        American option price ≥ European BSM price.

    Notes
    -----
    The BAW approximation uses a quadratic equation to find the critical
    exercise boundary S* via Newton-Raphson iteration, then adds the early
    exercise premium on top of the European price.

    Reference: Barone-Adesi, G. & Whaley, R.E. (1987).  "Efficient Analytic
    Approximation of American Option Values."  *Journal of Finance*, 42(2),
    301–320.

    See Also
    --------
    early_exercise_premium : Difference between American and European prices.
    """
    from ferro_ta._ferro_ta import american_price as _rust_american_price
    from ferro_ta._ferro_ta import american_price_batch as _rust_american_price_batch

    option_type = _validate_option_type(option_type)
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
            return float(
                _rust_american_price(
                    float(arrays["underlying"][0]),
                    float(arrays["strike"][0]),
                    float(arrays["rate"][0]),
                    float(arrays["time_to_expiry"][0]),
                    float(arrays["volatility"][0]),
                    option_type,
                    float(arrays["carry"][0]),
                )
            )
        out = _rust_american_price_batch(
            arrays["underlying"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            arrays["volatility"],
            option_type,
            arrays["carry"],
        )
        return np.asarray(out, dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


def early_exercise_premium(
    underlying: ArrayLike | float,
    strike: ArrayLike | float,
    rate: ArrayLike | float,
    time_to_expiry: ArrayLike | float,
    volatility: ArrayLike | float,
    *,
    option_type: str = "call",
    carry: ArrayLike | float = 0.0,
) -> ScalarOrArray:
    """Early exercise premium: American price − European BSM price.

    Represents the additional value an American option holder gains from the
    right to exercise before expiry.  Always ≥ 0.

    Parameters
    ----------
    underlying, strike, rate, time_to_expiry, volatility, option_type, carry:
        Same as :func:`american_option_price`.

    Returns
    -------
    float or NDArray[float64]
        Premium ≥ 0.  Typically 0 for calls with no dividends.

    Notes
    -----
    For equity calls with zero carry (no dividends), early exercise is never
    optimal so the premium is ≈ 0.  For puts (or calls on dividend-paying
    underlyings), the premium increases with in-the-moneyness, rate, and
    time to expiry.
    """
    from ferro_ta._ferro_ta import (
        early_exercise_premium as _rust_early_exercise_premium,
    )
    from ferro_ta._ferro_ta import (
        early_exercise_premium_batch as _rust_early_exercise_premium_batch,
    )

    option_type = _validate_option_type(option_type)
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
            return float(
                _rust_early_exercise_premium(
                    float(arrays["underlying"][0]),
                    float(arrays["strike"][0]),
                    float(arrays["rate"][0]),
                    float(arrays["time_to_expiry"][0]),
                    float(arrays["volatility"][0]),
                    option_type,
                    float(arrays["carry"][0]),
                )
            )
        out = _rust_early_exercise_premium_batch(
            arrays["underlying"],
            arrays["strike"],
            arrays["rate"],
            arrays["time_to_expiry"],
            arrays["volatility"],
            option_type,
            arrays["carry"],
        )
        return np.asarray(out, dtype=np.float64)
    except ValueError as err:
        _normalize_rust_error(err)


# ---------------------------------------------------------------------------
# Historical volatility estimators — populated once the Rust bridge is built
# ---------------------------------------------------------------------------


def close_to_close_vol(
    close: ArrayLike,
    window: int = 20,
    trading_days_per_year: float = 252.0,
) -> NDArray[np.float64]:
    """Rolling close-to-close realized volatility (annualised).

    Baseline estimator — uses only closing prices.  Less efficient than OHLC
    estimators but requires only daily close data.

    Parameters
    ----------
    close:
        Array of closing prices (length ≥ window + 1).
    window:
        Rolling look-back period in bars (default 20).
    trading_days_per_year:
        Annualisation factor (default 252).

    Returns
    -------
    NDArray[float64]
        Same length as *close*.  First ``window`` values are NaN.

    Notes
    -----
    Formula::

        σ = √( Σᵢ ln²(Cᵢ/Cᵢ₋₁) / window × trading_days_per_year )

    No Bessel correction is applied (population variance, not sample variance).
    """
    from ferro_ta._ferro_ta import close_to_close_vol as _rust_ctc

    try:
        arr = _to_f64(close)
        return np.asarray(
            _rust_ctc(arr, int(window), float(trading_days_per_year)), dtype=np.float64
        )
    except ValueError as err:
        _normalize_rust_error(err)


def parkinson_vol(
    high: ArrayLike,
    low: ArrayLike,
    window: int = 20,
    trading_days_per_year: float = 252.0,
) -> NDArray[np.float64]:
    """Rolling Parkinson high-low realized volatility estimator (annualised).

    ~5× more efficient than close-to-close for diffusion processes.
    Does **not** account for drift or overnight gaps.

    Parameters
    ----------
    high, low:
        Arrays of daily high and low prices (same length, ≥ window).
    window:
        Rolling look-back period in bars (default 20).
    trading_days_per_year:
        Annualisation factor (default 252).

    Returns
    -------
    NDArray[float64]
        Same length as *high*.  First ``window - 1`` values are NaN.

    Notes
    -----
    Formula per window::

        σ² = (1 / (4·ln2·window)) · Σ ln²(Hᵢ/Lᵢ) × trading_days_per_year

    Reference: Parkinson, M. (1980). "The Extreme Value Method for
    Estimating the Variance of the Rate of Return." *Journal of Business*, 53(1).
    """
    from ferro_ta._ferro_ta import parkinson_vol as _rust_parkinson

    try:
        return np.asarray(
            _rust_parkinson(
                _to_f64(high), _to_f64(low), int(window), float(trading_days_per_year)
            ),
            dtype=np.float64,
        )
    except ValueError as err:
        _normalize_rust_error(err)


def garman_klass_vol(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    window: int = 20,
    trading_days_per_year: float = 252.0,
) -> NDArray[np.float64]:
    """Rolling Garman-Klass OHLC realized volatility estimator (annualised).

    Extends Parkinson by incorporating the open-close return.  ~7.4× more
    efficient than close-to-close.  Does **not** handle overnight gaps.

    Parameters
    ----------
    open, high, low, close:
        Arrays of daily OHLC prices (same length, ≥ window).
    window:
        Rolling look-back period in bars (default 20).
    trading_days_per_year:
        Annualisation factor (default 252).

    Returns
    -------
    NDArray[float64]
        Same length as *close*.  First ``window - 1`` values are NaN.

    Notes
    -----
    Per-bar contribution::

        GK = 0.5·ln²(H/L) − (2·ln2 − 1)·ln²(C/O)

    Reference: Garman, M.B. & Klass, M.J. (1980). "On the Estimation of
    Security Price Volatilities from Historical Data." *Journal of Business*, 53(1).
    """
    from ferro_ta._ferro_ta import garman_klass_vol as _rust_gk

    try:
        return np.asarray(
            _rust_gk(
                _to_f64(open),
                _to_f64(high),
                _to_f64(low),
                _to_f64(close),
                int(window),
                float(trading_days_per_year),
            ),
            dtype=np.float64,
        )
    except ValueError as err:
        _normalize_rust_error(err)


def rogers_satchell_vol(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    window: int = 20,
    trading_days_per_year: float = 252.0,
) -> NDArray[np.float64]:
    """Rolling Rogers-Satchell OHLC realized volatility estimator (annualised).

    Drift-invariant: unbiased for assets with non-zero expected return.
    Does **not** handle overnight gaps.

    Parameters
    ----------
    open, high, low, close:
        Arrays of daily OHLC prices (same length, ≥ window).
    window:
        Rolling look-back period in bars (default 20).
    trading_days_per_year:
        Annualisation factor (default 252).

    Returns
    -------
    NDArray[float64]
        Same length as *close*.  First ``window - 1`` values are NaN.

    Notes
    -----
    Per-bar contribution (u = ln(H/O), d = ln(L/O), c = ln(C/O))::

        RS = u·(u − c) + d·(d − c)

    Reference: Rogers, L.C.G. & Satchell, S.E. (1991). "Estimating Variance
    from High, Low and Closing Prices." *Annals of Applied Probability*, 1(4).
    """
    from ferro_ta._ferro_ta import rogers_satchell_vol as _rust_rs

    try:
        return np.asarray(
            _rust_rs(
                _to_f64(open),
                _to_f64(high),
                _to_f64(low),
                _to_f64(close),
                int(window),
                float(trading_days_per_year),
            ),
            dtype=np.float64,
        )
    except ValueError as err:
        _normalize_rust_error(err)


def yang_zhang_vol(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    window: int = 20,
    trading_days_per_year: float = 252.0,
) -> NDArray[np.float64]:
    """Rolling Yang-Zhang OHLC realized volatility estimator (annualised).

    The most efficient standard estimator (~14× vs close-to-close).  Handles
    overnight gaps by combining overnight, intraday open-close, and
    Rogers-Satchell variance components with an optimal weight *k*.

    Parameters
    ----------
    open, high, low, close:
        Arrays of daily OHLC prices (same length, ≥ window + 1).
    window:
        Rolling look-back period in bars (default 20).
    trading_days_per_year:
        Annualisation factor (default 252).

    Returns
    -------
    NDArray[float64]
        Same length as *close*.  First ``window`` values are NaN.

    Notes
    -----
    Mixed estimator::

        σ²_YZ = σ²_overnight + k·σ²_open_close + (1−k)·σ²_RS

    where k = 0.34 / (1.34 + (window+1)/(window-1)).

    Reference: Yang, D. & Zhang, Q. (2000). "Drift-Independent Volatility
    Estimation Based on High, Low, Open, and Close Prices."
    *Journal of Business*, 73(3).
    """
    from ferro_ta._ferro_ta import yang_zhang_vol as _rust_yz

    try:
        return np.asarray(
            _rust_yz(
                _to_f64(open),
                _to_f64(high),
                _to_f64(low),
                _to_f64(close),
                int(window),
                float(trading_days_per_year),
            ),
            dtype=np.float64,
        )
    except ValueError as err:
        _normalize_rust_error(err)


def vol_cone(
    close: ArrayLike,
    *,
    windows: tuple[int, ...] = (21, 42, 63, 126, 252),
    trading_days_per_year: float = 252.0,
) -> VolCone:
    """Historical realised vol distribution across window lengths (volatility cone).

    For each window, computes the full history of rolling close-to-close
    realised vol, then returns the min / p25 / median / p75 / max distribution.
    Contextualises current implied vol: "Is 30 % IV cheap or expensive?"

    Parameters
    ----------
    close:
        Array of closing prices (length ≥ max(windows) + 1).
    windows:
        Tuple of rolling window sizes in bars.  Default ``(21, 42, 63, 126, 252)``
        (approx. 1 month, 2 months, 3 months, 6 months, 1 year).
    trading_days_per_year:
        Annualisation factor (default 252).

    Returns
    -------
    VolCone
        Dataclass with arrays ``windows``, ``min``, ``p25``, ``median``,
        ``p75``, ``max`` — one value per element of *windows*.

    Notes
    -----
    Uses close-to-close vol internally.  Overlay the current IV on the cone
    to see whether it is historically cheap or expensive for each tenor.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.analysis.options import vol_cone
    >>> rng = np.random.default_rng(0)
    >>> close = 100 * np.cumprod(np.exp(rng.normal(0, 0.01, 500)))
    >>> cone = vol_cone(close, windows=(21, 63, 252))
    >>> cone.median  # annualised median realised vol per window
    """
    from ferro_ta._ferro_ta import vol_cone as _rust_vol_cone

    try:
        arr = _to_f64(close)
        slices = _rust_vol_cone(arr, list(windows), float(trading_days_per_year))
        windows_arr = np.array([s[0] for s in slices], dtype=np.float64)
        return VolCone(
            windows=windows_arr,
            min=np.array([s[1] for s in slices], dtype=np.float64),
            p25=np.array([s[2] for s in slices], dtype=np.float64),
            median=np.array([s[3] for s in slices], dtype=np.float64),
            p75=np.array([s[4] for s in slices], dtype=np.float64),
            max=np.array([s[5] for s in slices], dtype=np.float64),
        )
    except ValueError as err:
        _normalize_rust_error(err)
