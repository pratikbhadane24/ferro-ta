"""
ferro_ta.analysis.options_strategy — Typed strategy parameter schemas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any

from ferro_ta.core.exceptions import FerroTAInputError, FerroTAValueError

__all__ = [
    "ExpirySelectorKind",
    "StrikeSelectorKind",
    "LegPreset",
    "RiskMode",
    "ExpirySelector",
    "StrikeSelector",
    "RiskControl",
    "SimulationLimits",
    "StrategyLeg",
    "DerivativesStrategy",
    "build_strategy_preset",
]


class ExpirySelectorKind(str, Enum):
    CURRENT_WEEK = "current_week"
    NEXT_WEEK = "next_week"
    CURRENT_MONTH = "current_month"
    NEXT_MONTH = "next_month"
    EXPLICIT_DATE = "explicit_date"


class StrikeSelectorKind(str, Enum):
    ATM = "atm"
    ITM = "itm"
    OTM = "otm"
    DELTA = "delta"
    EXPLICIT = "explicit"


class LegPreset(str, Enum):
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    CUSTOM = "custom"


class RiskMode(str, Enum):
    PER_LEG = "per_leg"
    COMBINED_PNL = "combined_pnl"


@dataclass(frozen=True)
class ExpirySelector:
    kind: ExpirySelectorKind | str
    explicit_date: date | None = None

    def __post_init__(self) -> None:
        kind = ExpirySelectorKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if kind is ExpirySelectorKind.EXPLICIT_DATE and self.explicit_date is None:
            raise FerroTAValueError(
                "ExpirySelector(kind='explicit_date') requires explicit_date."
            )
        if (
            kind is not ExpirySelectorKind.EXPLICIT_DATE
            and self.explicit_date is not None
        ):
            raise FerroTAValueError(
                "explicit_date is only valid when kind='explicit_date'."
            )


@dataclass(frozen=True)
class StrikeSelector:
    kind: StrikeSelectorKind | str
    steps: int = 0
    delta: float | None = None
    explicit_strike: float | None = None

    def __post_init__(self) -> None:
        kind = StrikeSelectorKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.steps < 0:
            raise FerroTAValueError("steps must be >= 0.")
        if kind is StrikeSelectorKind.DELTA and self.delta is None:
            raise FerroTAValueError(
                "StrikeSelector(kind='delta') requires a delta target."
            )
        if self.delta is not None and not (0.0 < float(self.delta) < 1.0):
            raise FerroTAValueError("delta must be in the open interval (0, 1).")
        if kind is StrikeSelectorKind.EXPLICIT and self.explicit_strike is None:
            raise FerroTAValueError(
                "StrikeSelector(kind='explicit') requires explicit_strike."
            )


@dataclass(frozen=True)
class RiskControl:
    stop_loss_type: str | None = None
    stop_loss_value: float | None = None
    target_type: str | None = None
    target_value: float | None = None
    trailstop_type: str | None = None
    trailstop_value: float | None = None
    breakeven_trigger: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "stop_loss_value",
            "target_value",
            "trailstop_value",
            "breakeven_trigger",
        ):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise FerroTAValueError(f"{name} must be >= 0.")


@dataclass(frozen=True)
class SimulationLimits:
    max_premium_outlay: float | None = None
    max_loss_per_trade: float | None = None
    daily_max_drawdown: float | None = None
    cooldown_bars: int = 0
    reentry_allowed: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_premium_outlay",
            "max_loss_per_trade",
            "daily_max_drawdown",
        ):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise FerroTAValueError(f"{name} must be >= 0.")
        if self.cooldown_bars < 0:
            raise FerroTAValueError("cooldown_bars must be >= 0.")


@dataclass(frozen=True)
class StrategyLeg:
    underlying: str
    expiry_selector: ExpirySelector
    strike_selector: StrikeSelector
    option_type: str
    side: str = "long"
    quantity: int = 1
    instrument: str = "option"
    premium_limit: float | None = None

    def __post_init__(self) -> None:
        if self.underlying.strip() == "":
            raise FerroTAInputError("underlying must not be empty.")
        if self.option_type not in {"call", "put"}:
            raise FerroTAValueError("option_type must be 'call' or 'put'.")
        if self.side not in {"long", "short"}:
            raise FerroTAValueError("side must be 'long' or 'short'.")
        if self.instrument not in {"option", "future"}:
            raise FerroTAValueError("instrument must be 'option' or 'future'.")
        if self.quantity == 0:
            raise FerroTAValueError("quantity must be non-zero.")
        if self.premium_limit is not None and self.premium_limit < 0.0:
            raise FerroTAValueError("premium_limit must be >= 0.")


@dataclass(frozen=True)
class DerivativesStrategy:
    name: str
    preset: LegPreset | str = LegPreset.CUSTOM
    legs: tuple[StrategyLeg, ...] = field(default_factory=tuple)
    risk_controls: RiskControl = field(default_factory=RiskControl)
    risk_mode: RiskMode | str = RiskMode.COMBINED_PNL
    commission: float = 0.0
    slippage: float = 0.0
    spread_assumption: float = 0.0
    limits: SimulationLimits = field(default_factory=SimulationLimits)

    def __post_init__(self) -> None:
        preset = LegPreset(self.preset)
        risk_mode = RiskMode(self.risk_mode)
        object.__setattr__(self, "preset", preset)
        object.__setattr__(self, "risk_mode", risk_mode)
        if self.name.strip() == "":
            raise FerroTAInputError("name must not be empty.")
        if len(self.legs) == 0:
            raise FerroTAInputError("legs must contain at least one strategy leg.")
        for cost_name in ("commission", "slippage", "spread_assumption"):
            if float(getattr(self, cost_name)) < 0.0:
                raise FerroTAValueError(f"{cost_name} must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_strategy_preset(
    preset: LegPreset | str,
    *,
    name: str,
    underlying: str,
    expiry_selector: ExpirySelector,
    base_strike_selector: StrikeSelector | None = None,
    risk_controls: RiskControl | None = None,
    risk_mode: RiskMode | str = RiskMode.COMBINED_PNL,
    commission: float = 0.0,
    slippage: float = 0.0,
    spread_assumption: float = 0.0,
    limits: SimulationLimits | None = None,
) -> DerivativesStrategy:
    """Build a common research preset using typed leg definitions."""
    preset = LegPreset(preset)
    risk_controls = risk_controls or RiskControl()
    limits = limits or SimulationLimits()
    atm = base_strike_selector or StrikeSelector(StrikeSelectorKind.ATM)

    if preset is LegPreset.CUSTOM:
        raise FerroTAValueError(
            "build_strategy_preset does not construct CUSTOM presets."
        )

    legs: tuple[StrategyLeg, ...]

    if preset is LegPreset.STRADDLE:
        legs = (
            StrategyLeg(underlying, expiry_selector, atm, "call", "long"),
            StrategyLeg(underlying, expiry_selector, atm, "put", "long"),
        )
    elif preset is LegPreset.STRANGLE:
        legs = (
            StrategyLeg(
                underlying,
                expiry_selector,
                StrikeSelector(StrikeSelectorKind.OTM, steps=1),
                "call",
                "long",
            ),
            StrategyLeg(
                underlying,
                expiry_selector,
                StrikeSelector(StrikeSelectorKind.OTM, steps=1),
                "put",
                "long",
            ),
        )
    elif preset is LegPreset.BULL_CALL_SPREAD:
        legs = (
            StrategyLeg(underlying, expiry_selector, atm, "call", "long"),
            StrategyLeg(
                underlying,
                expiry_selector,
                StrikeSelector(StrikeSelectorKind.OTM, steps=1),
                "call",
                "short",
            ),
        )
    elif preset is LegPreset.BEAR_PUT_SPREAD:
        legs = (
            StrategyLeg(underlying, expiry_selector, atm, "put", "long"),
            StrategyLeg(
                underlying,
                expiry_selector,
                StrikeSelector(StrikeSelectorKind.OTM, steps=1),
                "put",
                "short",
            ),
        )
    elif preset is LegPreset.IRON_CONDOR:
        legs = (
            StrategyLeg(
                underlying,
                expiry_selector,
                StrikeSelector(StrikeSelectorKind.OTM, steps=1),
                "put",
                "short",
            ),
            StrategyLeg(
                underlying,
                expiry_selector,
                StrikeSelector(StrikeSelectorKind.OTM, steps=2),
                "put",
                "long",
            ),
            StrategyLeg(
                underlying,
                expiry_selector,
                StrikeSelector(StrikeSelectorKind.OTM, steps=1),
                "call",
                "short",
            ),
            StrategyLeg(
                underlying,
                expiry_selector,
                StrikeSelector(StrikeSelectorKind.OTM, steps=2),
                "call",
                "long",
            ),
        )
    else:
        raise FerroTAValueError(f"Unsupported preset '{preset.value}'.")

    return DerivativesStrategy(
        name=name,
        preset=preset,
        legs=legs,
        risk_controls=risk_controls,
        risk_mode=risk_mode,
        commission=commission,
        slippage=slippage,
        spread_assumption=spread_assumption,
        limits=limits,
    )
