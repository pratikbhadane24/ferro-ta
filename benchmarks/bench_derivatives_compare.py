"""
Compare ferro_ta derivatives analytics against analytical references and
optional third-party libraries available in the current environment.

The suite focuses on selected core workflows:

- Black-Scholes-Merton call pricing
- implied-volatility recovery
- first-order Greeks
- Black-76 call pricing

Outputs include:

- speed timings with per-run samples and variance stats
- analytical accuracy metrics
- Python-tracked peak allocation snapshots
- machine, runtime, build, and package metadata
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ferro_ta.analysis.options import (
    black_76_price as ft_black_76_price,
)
from ferro_ta.analysis.options import (
    greeks as ft_greeks,
)
from ferro_ta.analysis.options import (
    implied_volatility as ft_implied_volatility,
)
from ferro_ta.analysis.options import (
    option_price as ft_option_price,
)

try:
    from benchmarks.metadata import benchmark_metadata, package_versions
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from metadata import benchmark_metadata, package_versions


N_WARMUP = 1
N_RUNS = 7
DEFAULT_SIZES = [1_000, 10_000]
DEFAULT_ACCURACY_SIZE = 512
DEFAULT_SEED = 42
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Case:
    name: str
    label: str
    expected_key: str
    component_names: tuple[str, ...] | None = None
    accuracy_target: str = "expected_output"


@dataclass(frozen=True)
class Provider:
    name: str
    kind: str
    note: str
    functions: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]]
    max_speed_size: int | None = None

    def supports(self, case_name: str) -> bool:
        return case_name in self.functions


CASES = [
    Case("bsm_call_price", "BSM Call Price", "call_price"),
    Case(
        "bsm_call_iv",
        "BSM Call IV Recovery",
        "volatility",
        accuracy_target="reconstructed_price",
    ),
    Case(
        "bsm_call_greeks",
        "BSM Call Greeks",
        "call_greeks",
        component_names=("delta", "gamma", "vega", "theta", "rho"),
    ),
    Case("black76_call_price", "Black-76 Call Price", "black76_call_price"),
]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _summary_stats(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stddev_ms": 0.0,
            "cv_pct": 0.0,
        }

    mean_ms = sum(samples_ms) / len(samples_ms)
    variance = (
        sum((sample - mean_ms) ** 2 for sample in samples_ms) / (len(samples_ms) - 1)
        if len(samples_ms) > 1
        else 0.0
    )
    stddev_ms = math.sqrt(variance)
    cv_pct = (stddev_ms / mean_ms * 100.0) if mean_ms else 0.0
    return {
        "median_ms": round(_median(samples_ms), 4),
        "mean_ms": round(mean_ms, 4),
        "min_ms": round(min(samples_ms), 4),
        "max_ms": round(max(samples_ms), 4),
        "stddev_ms": round(stddev_ms, 4),
        "cv_pct": round(cv_pct, 3),
    }


def _timed_runs_ms(
    fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    chain: dict[str, np.ndarray],
) -> list[float]:
    for _ in range(N_WARMUP):
        fn(chain)

    samples_ms: list[float] = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(chain)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms


def _python_peak_bytes(
    fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    chain: dict[str, np.ndarray],
) -> int | None:
    try:
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn(chain)
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    except Exception:
        return None
    finally:
        tracemalloc.stop()


def _throughput_contracts_s(size: int, median_ms: float) -> float:
    if median_ms <= 0:
        return 0.0
    return size / (median_ms / 1000.0)


def _normal_pdf_numpy(x: np.ndarray) -> np.ndarray:
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _normal_cdf_numpy(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * abs_x)
    poly = (
        (((((1.330274429 * t) - 1.821255978) * t) + 1.781477937) * t - 0.356563782) * t
        + 0.319381530
    ) * t
    cdf = 1.0 - _normal_pdf_numpy(abs_x) * poly
    return np.where(x >= 0.0, cdf, 1.0 - cdf)


def _normal_cdf_scalar(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf_scalar(x: float) -> float:
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _bsm_price_numpy(
    spot: np.ndarray,
    strike: np.ndarray,
    rate: np.ndarray,
    time_to_expiry: np.ndarray,
    volatility: np.ndarray,
    *,
    option_type: str,
    carry: np.ndarray,
) -> np.ndarray:
    sqrt_t = np.sqrt(time_to_expiry)
    sigma_sqrt_t = volatility * sqrt_t
    d1 = (
        np.log(spot / strike)
        + (rate - carry + 0.5 * volatility * volatility) * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    spot_df = np.exp(-carry * time_to_expiry)
    strike_df = np.exp(-rate * time_to_expiry)
    if option_type == "call":
        out = spot * spot_df * _normal_cdf_numpy(
            d1
        ) - strike * strike_df * _normal_cdf_numpy(d2)
    else:
        out = strike * strike_df * _normal_cdf_numpy(
            -d2
        ) - spot * spot_df * _normal_cdf_numpy(-d1)
    return np.ascontiguousarray(out, dtype=np.float64)


def _black76_price_numpy(
    forward: np.ndarray,
    strike: np.ndarray,
    rate: np.ndarray,
    time_to_expiry: np.ndarray,
    volatility: np.ndarray,
    *,
    option_type: str,
) -> np.ndarray:
    sqrt_t = np.sqrt(time_to_expiry)
    sigma_sqrt_t = volatility * sqrt_t
    d1 = (
        np.log(forward / strike) + 0.5 * volatility * volatility * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    discount = np.exp(-rate * time_to_expiry)
    if option_type == "call":
        out = discount * (
            forward * _normal_cdf_numpy(d1) - strike * _normal_cdf_numpy(d2)
        )
    else:
        out = discount * (
            strike * _normal_cdf_numpy(-d2) - forward * _normal_cdf_numpy(-d1)
        )
    return np.ascontiguousarray(out, dtype=np.float64)


def _bsm_greeks_numpy(
    spot: np.ndarray,
    strike: np.ndarray,
    rate: np.ndarray,
    time_to_expiry: np.ndarray,
    volatility: np.ndarray,
    *,
    option_type: str,
    carry: np.ndarray,
) -> np.ndarray:
    sqrt_t = np.sqrt(time_to_expiry)
    sigma_sqrt_t = volatility * sqrt_t
    d1 = (
        np.log(spot / strike)
        + (rate - carry + 0.5 * volatility * volatility) * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    pdf = _normal_pdf_numpy(d1)
    carry_df = np.exp(-carry * time_to_expiry)
    strike_df = np.exp(-rate * time_to_expiry)

    if option_type == "call":
        delta = carry_df * _normal_cdf_numpy(d1)
        theta = (
            -(spot * carry_df * pdf * volatility) / (2.0 * sqrt_t)
            - rate * strike * strike_df * _normal_cdf_numpy(d2)
            + carry * spot * carry_df * _normal_cdf_numpy(d1)
        )
        rho = strike * time_to_expiry * strike_df * _normal_cdf_numpy(d2)
    else:
        delta = carry_df * (_normal_cdf_numpy(d1) - 1.0)
        theta = (
            -(spot * carry_df * pdf * volatility) / (2.0 * sqrt_t)
            + rate * strike * strike_df * _normal_cdf_numpy(-d2)
            - carry * spot * carry_df * _normal_cdf_numpy(-d1)
        )
        rho = -strike * time_to_expiry * strike_df * _normal_cdf_numpy(-d2)

    gamma = carry_df * pdf / (spot * sigma_sqrt_t)
    vega = spot * carry_df * pdf * sqrt_t
    return np.ascontiguousarray(
        np.column_stack([delta, gamma, vega, theta, rho]),
        dtype=np.float64,
    )


def _bsm_price_scalar(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    *,
    option_type: str,
    carry: float,
) -> float:
    sqrt_t = math.sqrt(time_to_expiry)
    sigma_sqrt_t = volatility * sqrt_t
    d1 = (
        math.log(spot / strike)
        + (rate - carry + 0.5 * volatility * volatility) * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    spot_df = math.exp(-carry * time_to_expiry)
    strike_df = math.exp(-rate * time_to_expiry)
    if option_type == "call":
        return spot * spot_df * _normal_cdf_scalar(
            d1
        ) - strike * strike_df * _normal_cdf_scalar(d2)
    return strike * strike_df * _normal_cdf_scalar(
        -d2
    ) - spot * spot_df * _normal_cdf_scalar(-d1)


def _black76_price_scalar(
    forward: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    *,
    option_type: str,
) -> float:
    sqrt_t = math.sqrt(time_to_expiry)
    sigma_sqrt_t = volatility * sqrt_t
    d1 = (
        math.log(forward / strike) + 0.5 * volatility * volatility * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    discount = math.exp(-rate * time_to_expiry)
    if option_type == "call":
        return discount * (
            forward * _normal_cdf_scalar(d1) - strike * _normal_cdf_scalar(d2)
        )
    return discount * (
        strike * _normal_cdf_scalar(-d2) - forward * _normal_cdf_scalar(-d1)
    )


def _bsm_greeks_scalar(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    *,
    option_type: str,
    carry: float,
) -> tuple[float, float, float, float, float]:
    sqrt_t = math.sqrt(time_to_expiry)
    sigma_sqrt_t = volatility * sqrt_t
    d1 = (
        math.log(spot / strike)
        + (rate - carry + 0.5 * volatility * volatility) * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    pdf = _normal_pdf_scalar(d1)
    carry_df = math.exp(-carry * time_to_expiry)
    strike_df = math.exp(-rate * time_to_expiry)

    if option_type == "call":
        delta = carry_df * _normal_cdf_scalar(d1)
        theta = (
            -(spot * carry_df * pdf * volatility) / (2.0 * sqrt_t)
            - rate * strike * strike_df * _normal_cdf_scalar(d2)
            + carry * spot * carry_df * _normal_cdf_scalar(d1)
        )
        rho = strike * time_to_expiry * strike_df * _normal_cdf_scalar(d2)
    else:
        delta = carry_df * (_normal_cdf_scalar(d1) - 1.0)
        theta = (
            -(spot * carry_df * pdf * volatility) / (2.0 * sqrt_t)
            + rate * strike * strike_df * _normal_cdf_scalar(-d2)
            - carry * spot * carry_df * _normal_cdf_scalar(-d1)
        )
        rho = -strike * time_to_expiry * strike_df * _normal_cdf_scalar(-d2)

    gamma = carry_df * pdf / (spot * sigma_sqrt_t)
    vega = spot * carry_df * pdf * sqrt_t
    return delta, gamma, vega, theta, rho


def _implied_vol_bisection_numpy(
    price: np.ndarray,
    spot: np.ndarray,
    strike: np.ndarray,
    rate: np.ndarray,
    time_to_expiry: np.ndarray,
    *,
    option_type: str,
    carry: np.ndarray,
    lower: float = 1e-6,
    upper: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> np.ndarray:
    lo = np.full_like(price, lower, dtype=np.float64)
    hi = np.full_like(price, upper, dtype=np.float64)
    mid = np.full_like(price, 0.2, dtype=np.float64)
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        estimate = _bsm_price_numpy(
            spot,
            strike,
            rate,
            time_to_expiry,
            mid,
            option_type=option_type,
            carry=carry,
        )
        too_low = estimate < price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
        if float(np.max(np.abs(estimate - price))) < tolerance:
            break
    return np.ascontiguousarray(mid, dtype=np.float64)


def _implied_vol_bisection_scalar(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    *,
    option_type: str,
    carry: float,
    lower: float = 1e-6,
    upper: float = 5.0,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    lo = lower
    hi = upper
    mid = 0.2
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        estimate = _bsm_price_scalar(
            spot,
            strike,
            rate,
            time_to_expiry,
            mid,
            option_type=option_type,
            carry=carry,
        )
        if abs(estimate - price) < tolerance:
            return mid
        if estimate < price:
            lo = mid
        else:
            hi = mid
    return mid


def _reference_python_loop(
    chain: dict[str, np.ndarray],
    fn: Callable[..., float | tuple[float, ...]],
    *,
    include_forward: bool = False,
    include_price: bool = False,
) -> np.ndarray:
    rows: list[Any] = []
    for idx in range(len(chain["strike"])):
        kwargs: dict[str, float | str] = {
            "strike": float(chain["strike"][idx]),
            "rate": float(chain["rate"][idx]),
            "time_to_expiry": float(chain["time_to_expiry"][idx]),
            "volatility": float(chain["volatility"][idx]),
            "carry": float(chain["carry"][idx]),
        }
        if include_forward:
            kwargs["forward"] = float(chain["forward"][idx])
        else:
            kwargs["spot"] = float(chain["spot"][idx])
        if include_price:
            kwargs["price"] = float(chain["call_price"][idx])
        rows.append(fn(**kwargs))
    return np.asarray(rows, dtype=np.float64)


def _build_chain(n: int, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spot = np.ascontiguousarray(rng.uniform(80.0, 120.0, size=n), dtype=np.float64)
    strike = np.ascontiguousarray(rng.uniform(70.0, 130.0, size=n), dtype=np.float64)
    rate = np.ascontiguousarray(rng.uniform(0.0, 0.07, size=n), dtype=np.float64)
    carry = np.ascontiguousarray(rng.uniform(0.0, 0.03, size=n), dtype=np.float64)
    time_to_expiry = np.ascontiguousarray(
        rng.uniform(7.0 / 365.0, 2.0, size=n), dtype=np.float64
    )
    volatility = np.ascontiguousarray(rng.uniform(0.08, 0.65, size=n), dtype=np.float64)
    forward = np.ascontiguousarray(
        spot * np.exp((rate - carry) * time_to_expiry),
        dtype=np.float64,
    )

    chain = {
        "spot": spot,
        "strike": strike,
        "rate": rate,
        "carry": carry,
        "time_to_expiry": time_to_expiry,
        "volatility": volatility,
        "forward": forward,
    }

    call_price = _reference_python_loop(
        chain,
        lambda *, spot, strike, rate, time_to_expiry, volatility, carry: (
            _bsm_price_scalar(
                spot,
                strike,
                rate,
                time_to_expiry,
                volatility,
                option_type="call",
                carry=carry,
            )
        ),
    )
    call_greeks = _reference_python_loop(
        chain,
        lambda *, spot, strike, rate, time_to_expiry, volatility, carry: (
            _bsm_greeks_scalar(
                spot,
                strike,
                rate,
                time_to_expiry,
                volatility,
                option_type="call",
                carry=carry,
            )
        ),
    )
    black76_call_price = _reference_python_loop(
        chain,
        lambda *, forward, strike, rate, time_to_expiry, volatility, carry: (
            _black76_price_scalar(
                forward,
                strike,
                rate,
                time_to_expiry,
                volatility,
                option_type="call",
            )
        ),
        include_forward=True,
    )

    chain["call_price"] = np.ascontiguousarray(call_price, dtype=np.float64)
    chain["call_greeks"] = np.ascontiguousarray(call_greeks, dtype=np.float64)
    chain["black76_call_price"] = np.ascontiguousarray(
        black76_call_price, dtype=np.float64
    )
    return chain


def _ferro_ta_call_price(chain: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        ft_option_price(
            chain["spot"],
            chain["strike"],
            chain["rate"],
            chain["time_to_expiry"],
            chain["volatility"],
            option_type="call",
            model="bsm",
            carry=chain["carry"],
        ),
        dtype=np.float64,
    )


def _ferro_ta_call_iv(chain: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        ft_implied_volatility(
            chain["call_price"],
            chain["spot"],
            chain["strike"],
            chain["rate"],
            chain["time_to_expiry"],
            option_type="call",
            model="bsm",
            carry=chain["carry"],
        ),
        dtype=np.float64,
    )


def _ferro_ta_call_greeks(chain: dict[str, np.ndarray]) -> np.ndarray:
    result = ft_greeks(
        chain["spot"],
        chain["strike"],
        chain["rate"],
        chain["time_to_expiry"],
        chain["volatility"],
        option_type="call",
        model="bsm",
        carry=chain["carry"],
    )
    return np.ascontiguousarray(
        np.column_stack(
            [
                np.asarray(result.delta, dtype=np.float64),
                np.asarray(result.gamma, dtype=np.float64),
                np.asarray(result.vega, dtype=np.float64),
                np.asarray(result.theta, dtype=np.float64),
                np.asarray(result.rho, dtype=np.float64),
            ]
        ),
        dtype=np.float64,
    )


def _ferro_ta_black76_call_price(chain: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        ft_black_76_price(
            chain["forward"],
            chain["strike"],
            chain["rate"],
            chain["time_to_expiry"],
            chain["volatility"],
            option_type="call",
        ),
        dtype=np.float64,
    )


def _reference_numpy_call_price(chain: dict[str, np.ndarray]) -> np.ndarray:
    return _bsm_price_numpy(
        chain["spot"],
        chain["strike"],
        chain["rate"],
        chain["time_to_expiry"],
        chain["volatility"],
        option_type="call",
        carry=chain["carry"],
    )


def _reference_numpy_call_iv(chain: dict[str, np.ndarray]) -> np.ndarray:
    return _implied_vol_bisection_numpy(
        chain["call_price"],
        chain["spot"],
        chain["strike"],
        chain["rate"],
        chain["time_to_expiry"],
        option_type="call",
        carry=chain["carry"],
    )


def _reference_numpy_call_greeks(chain: dict[str, np.ndarray]) -> np.ndarray:
    return _bsm_greeks_numpy(
        chain["spot"],
        chain["strike"],
        chain["rate"],
        chain["time_to_expiry"],
        chain["volatility"],
        option_type="call",
        carry=chain["carry"],
    )


def _reference_numpy_black76_call_price(chain: dict[str, np.ndarray]) -> np.ndarray:
    return _black76_price_numpy(
        chain["forward"],
        chain["strike"],
        chain["rate"],
        chain["time_to_expiry"],
        chain["volatility"],
        option_type="call",
    )


def _reference_python_call_price(chain: dict[str, np.ndarray]) -> np.ndarray:
    return _reference_python_loop(
        chain,
        lambda *, spot, strike, rate, time_to_expiry, volatility, carry: (
            _bsm_price_scalar(
                spot,
                strike,
                rate,
                time_to_expiry,
                volatility,
                option_type="call",
                carry=carry,
            )
        ),
    )


def _reference_python_call_iv(chain: dict[str, np.ndarray]) -> np.ndarray:
    return _reference_python_loop(
        chain,
        lambda *, price, spot, strike, rate, time_to_expiry, volatility, carry: (
            _implied_vol_bisection_scalar(
                price,
                spot,
                strike,
                rate,
                time_to_expiry,
                option_type="call",
                carry=carry,
            )
        ),
        include_price=True,
    )


def _reference_python_call_greeks(chain: dict[str, np.ndarray]) -> np.ndarray:
    return _reference_python_loop(
        chain,
        lambda *, spot, strike, rate, time_to_expiry, volatility, carry: (
            _bsm_greeks_scalar(
                spot,
                strike,
                rate,
                time_to_expiry,
                volatility,
                option_type="call",
                carry=carry,
            )
        ),
    )


def _reference_python_black76_call_price(chain: dict[str, np.ndarray]) -> np.ndarray:
    return _reference_python_loop(
        chain,
        lambda *, forward, strike, rate, time_to_expiry, volatility, carry: (
            _black76_price_scalar(
                forward,
                strike,
                rate,
                time_to_expiry,
                volatility,
                option_type="call",
            )
        ),
        include_forward=True,
    )


def _reprice_bsm_call_from_iv(
    chain: dict[str, np.ndarray],
    implied_vols: np.ndarray,
) -> np.ndarray:
    rows = [
        _bsm_price_scalar(
            float(spot),
            float(strike),
            float(rate),
            float(time_to_expiry),
            max(float(iv), 1e-12),
            option_type="call",
            carry=float(carry),
        )
        for spot, strike, rate, time_to_expiry, iv, carry in zip(
            chain["spot"],
            chain["strike"],
            chain["rate"],
            chain["time_to_expiry"],
            np.asarray(implied_vols, dtype=np.float64),
            chain["carry"],
        )
    ]
    return np.asarray(rows, dtype=np.float64)


def _py_vollib_provider() -> Provider | None:
    if importlib.util.find_spec("py_vollib") is None:
        return None

    from py_vollib.black_scholes_merton import black_scholes_merton as py_vollib_bsm
    from py_vollib.black_scholes_merton.implied_volatility import (
        implied_volatility as py_vollib_iv,
    )

    def _price(chain: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(
            [
                py_vollib_bsm(
                    "c",
                    float(s),
                    float(k),
                    float(t),
                    float(r),
                    float(vol),
                    float(q),
                )
                for s, k, r, t, vol, q in zip(
                    chain["spot"],
                    chain["strike"],
                    chain["rate"],
                    chain["time_to_expiry"],
                    chain["volatility"],
                    chain["carry"],
                )
            ],
            dtype=np.float64,
        )

    def _iv(chain: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(
            [
                py_vollib_iv(
                    float(price),
                    "c",
                    float(s),
                    float(k),
                    float(t),
                    float(r),
                    float(q),
                )
                for price, s, k, r, t, q in zip(
                    chain["call_price"],
                    chain["spot"],
                    chain["strike"],
                    chain["rate"],
                    chain["time_to_expiry"],
                    chain["carry"],
                )
            ],
            dtype=np.float64,
        )

    return Provider(
        name="py_vollib",
        kind="third_party",
        note="Scalar Black-Scholes-Merton baseline from py_vollib.",
        functions={
            "bsm_call_price": _price,
            "bsm_call_iv": _iv,
        },
        max_speed_size=1_000,
    )


def available_providers() -> list[Provider]:
    providers = [
        Provider(
            name="ferro_ta",
            kind="project",
            note="Rust-backed vectorized implementation.",
            functions={
                "bsm_call_price": _ferro_ta_call_price,
                "bsm_call_iv": _ferro_ta_call_iv,
                "bsm_call_greeks": _ferro_ta_call_greeks,
                "black76_call_price": _ferro_ta_black76_call_price,
            },
        ),
        Provider(
            name="reference_numpy",
            kind="reference",
            note="Pure NumPy analytical formulas with vectorized IV bisection.",
            functions={
                "bsm_call_price": _reference_numpy_call_price,
                "bsm_call_iv": _reference_numpy_call_iv,
                "bsm_call_greeks": _reference_numpy_call_greeks,
                "black76_call_price": _reference_numpy_black76_call_price,
            },
        ),
        Provider(
            name="reference_python_loop",
            kind="reference",
            note="Scalar math-loop analytical baseline; useful for accuracy sanity checks.",
            functions={
                "bsm_call_price": _reference_python_call_price,
                "bsm_call_iv": _reference_python_call_iv,
                "bsm_call_greeks": _reference_python_call_greeks,
                "black76_call_price": _reference_python_black76_call_price,
            },
            max_speed_size=1_000,
        ),
    ]
    optional = _py_vollib_provider()
    if optional is not None:
        providers.append(optional)
    return providers


def _accuracy_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    abs_error = np.abs(actual_arr - expected_arr)
    rel_error = abs_error / np.maximum(np.abs(expected_arr), 1e-12)
    return {
        "output_shape": list(actual_arr.shape),
        "max_abs_error": round(float(np.max(abs_error)), 12),
        "mean_abs_error": round(float(np.mean(abs_error)), 12),
        "rmse": round(
            float(np.sqrt(np.mean(np.square(actual_arr - expected_arr)))), 12
        ),
        "max_rel_error": round(float(np.max(rel_error)), 12),
    }


def _accuracy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for case in CASES:
        case_rows = [
            row for row in rows if row["case"] == case.name and "max_abs_error" in row
        ]
        if not case_rows:
            continue
        best = min(case_rows, key=lambda row: float(row["max_abs_error"]))
        worst = max(case_rows, key=lambda row: float(row["max_abs_error"]))
        summary.append(
            {
                "case": case.name,
                "best_provider": best["provider"],
                "best_max_abs_error": best["max_abs_error"],
                "worst_provider": worst["provider"],
                "worst_max_abs_error": worst["max_abs_error"],
            }
        )
    return summary


def _speed_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for case in CASES:
        for size in sorted(
            {
                int(row["size"])
                for row in rows
                if row["case"] == case.name and "median_ms" in row
            }
        ):
            case_rows = [
                row
                for row in rows
                if row["case"] == case.name
                and row.get("size") == size
                and "median_ms" in row
            ]
            if not case_rows:
                continue
            fastest = min(case_rows, key=lambda row: float(row["median_ms"]))
            ranking = [
                {
                    "provider": row["provider"],
                    "median_ms": row["median_ms"],
                    "contracts_per_s": row["contracts_per_s"],
                }
                for row in sorted(case_rows, key=lambda row: float(row["median_ms"]))
            ]
            summary.append(
                {
                    "case": case.name,
                    "size": size,
                    "fastest_provider": fastest["provider"],
                    "fastest_median_ms": fastest["median_ms"],
                    "ranking": ranking,
                }
            )
    return summary


def _print_provider_inventory(providers: list[Provider]) -> None:
    print("Providers:")
    for provider in providers:
        supported = ", ".join(
            case.name for case in CASES if provider.supports(case.name)
        )
        cap = (
            f" (speed cap {provider.max_speed_size})" if provider.max_speed_size else ""
        )
        print(f" - {provider.name} [{provider.kind}] {provider.note}{cap}")
        print(f"   supported: {supported}")
    print()


def _print_accuracy_table(rows: list[dict[str, Any]], accuracy_size: int) -> None:
    print(f"Accuracy ({accuracy_size} contracts)")
    print(
        "IV accuracy is measured as price reconstruction error from the recovered IV."
    )
    header = f"{'Case':<22} {'Provider':<24} {'Max abs err':<14} {'RMSE':<14} {'Max rel err':<14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if "max_abs_error" not in row:
            continue
        print(
            f"{row['label']:<22} {row['provider']:<24} "
            f"{row['max_abs_error']:<14.6g} {row['rmse']:<14.6g} {row['max_rel_error']:<14.6g}"
        )
    print()


def _print_speed_table(rows: list[dict[str, Any]]) -> None:
    print(f"Speed (median of {N_RUNS} measured runs after {N_WARMUP} warmup)")
    header = f"{'Case':<22} {'Size':<8} {'Provider':<24} {'Median ms':<12} {'Contracts/s':<14} {'Peak alloc':<12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if "median_ms" not in row:
            continue
        peak = row.get("python_peak_allocation_bytes")
        peak_label = "n/a" if peak is None else str(peak)
        print(
            f"{row['label']:<22} {row['size']:<8} {row['provider']:<24} "
            f"{row['median_ms']:<12.4f} {row['contracts_per_s']:<14.2f} {peak_label:<12}"
        )
    print()


def run_benchmark(
    *,
    sizes: list[int],
    accuracy_size: int,
    json_path: str | None,
) -> dict[str, Any]:
    providers = available_providers()
    speed_chains = {
        size: _build_chain(size, seed=DEFAULT_SEED + size) for size in sizes
    }
    accuracy_chain = _build_chain(accuracy_size, seed=DEFAULT_SEED)

    accuracy_rows: list[dict[str, Any]] = []
    speed_rows: list[dict[str, Any]] = []

    _print_provider_inventory(providers)

    for case in CASES:
        for provider in providers:
            if not provider.supports(case.name):
                continue
            fn = provider.functions[case.name]
            actual = fn(accuracy_chain)
            if case.name == "bsm_call_iv":
                compared_actual = _reprice_bsm_call_from_iv(accuracy_chain, actual)
                expected = np.asarray(accuracy_chain["call_price"], dtype=np.float64)
            else:
                compared_actual = actual
                expected = np.asarray(
                    accuracy_chain[case.expected_key], dtype=np.float64
                )
            row = {
                "case": case.name,
                "label": case.label,
                "provider": provider.name,
                "provider_kind": provider.kind,
                "sample_size": accuracy_size,
                "accuracy_target": case.accuracy_target,
            }
            row.update(_accuracy_metrics(compared_actual, expected))
            if case.component_names is not None:
                row["component_names"] = list(case.component_names)
            accuracy_rows.append(row)

    _print_accuracy_table(accuracy_rows, accuracy_size)

    for case in CASES:
        for size in sizes:
            chain = speed_chains[size]
            for provider in providers:
                if not provider.supports(case.name):
                    continue
                if (
                    provider.max_speed_size is not None
                    and size > provider.max_speed_size
                ):
                    continue

                fn = provider.functions[case.name]
                samples_ms = _timed_runs_ms(fn, chain)
                stats = _summary_stats(samples_ms)
                median_ms = float(stats["median_ms"])
                contracts_per_s = _throughput_contracts_s(size, median_ms)
                peak_bytes = _python_peak_bytes(fn, chain)

                speed_rows.append(
                    {
                        "case": case.name,
                        "label": case.label,
                        "size": size,
                        "provider": provider.name,
                        "provider_kind": provider.kind,
                        "median_ms": round(median_ms, 4),
                        "contracts_per_s": round(contracts_per_s, 2),
                        "runs_ms": [round(sample, 4) for sample in samples_ms],
                        "stats": stats,
                        "python_peak_allocation_bytes": peak_bytes,
                        "input_layout": {
                            "dtype": "float64",
                            "contiguous": True,
                        },
                    }
                )

    _print_speed_table(speed_rows)

    package_names = ["numpy", "ferro-ta", "py_vollib"]
    metadata = benchmark_metadata(
        "benchmark_derivatives_compare",
        extra={
            "dataset": {
                "generator": "synthetic_option_chain",
                "speed_sizes": sizes,
                "accuracy_size": accuracy_size,
                "dtype": "float64",
                "array_layout": "C-contiguous",
                "seed": DEFAULT_SEED,
                "ranges": {
                    "spot": [80.0, 120.0],
                    "strike": [70.0, 130.0],
                    "rate": [0.0, 0.07],
                    "carry": [0.0, 0.03],
                    "time_to_expiry_years": [7.0 / 365.0, 2.0],
                    "volatility": [0.08, 0.65],
                },
            },
            "methodology": {
                "warmup_runs": N_WARMUP,
                "measured_runs": N_RUNS,
                "reported_metric": "median_ms",
                "speed_metric": "contracts_per_second",
                "accuracy_reference": (
                    "Scalar analytical Black-Scholes-Merton and Black-76 formulas "
                    "using math.erf; IV accuracy is measured as repriced error "
                    "from the recovered volatility because direct volatility "
                    "differences can be unstable on low-vega contracts."
                ),
                "input_layout_notes": (
                    "Benchmarks use contiguous float64 arrays. If your workload "
                    "passes non-contiguous arrays or mixed dtypes, benchmark that "
                    "path separately."
                ),
                "allocation_notes": (
                    "python_peak_allocation_bytes is a tracemalloc snapshot of "
                    "Python-tracked allocations only; it does not measure native RSS."
                ),
                "provider_notes": (
                    "reference_python_loop and py_vollib are scalar baselines and "
                    "are size-capped in the speed table to keep runtime reasonable."
                ),
            },
            "providers": [
                {
                    "name": provider.name,
                    "kind": provider.kind,
                    "note": provider.note,
                    "max_speed_size": provider.max_speed_size,
                    "supported_cases": [
                        case.name for case in CASES if provider.supports(case.name)
                    ],
                }
                for provider in providers
            ],
            "packages": package_versions(*package_names),
        },
    )

    result = {
        "schema_version": 1,
        "command": " ".join(["python", *sys.argv]),
        "n_warmup": N_WARMUP,
        "n_runs": N_RUNS,
        "accuracy_size": accuracy_size,
        "sizes": sizes,
        "metadata": metadata,
        "accuracy": {
            "summary": _accuracy_summary(accuracy_rows),
            "results": accuracy_rows,
        },
        "speed": {
            "summary": _speed_summary(speed_rows),
            "results": speed_rows,
        },
    }

    if json_path:
        output_path = Path(json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Results written to {output_path}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare ferro_ta derivatives analytics against reference implementations"
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Write the benchmark artifact to JSON",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Contract counts to benchmark (default: 1000 10000)",
    )
    parser.add_argument(
        "--accuracy-size",
        type=int,
        default=DEFAULT_ACCURACY_SIZE,
        help="Contract count used for the accuracy pass (default: 512)",
    )
    args = parser.parse_args()
    run_benchmark(
        sizes=args.sizes,
        accuracy_size=args.accuracy_size,
        json_path=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
