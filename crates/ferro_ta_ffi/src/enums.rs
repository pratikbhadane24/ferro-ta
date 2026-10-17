//! C enums: validated `int32_t` values mapped onto core Rust enums.
//!
//! A raw integer is never transmuted into a Rust enum (an out-of-range value
//! would be undefined behaviour); each conversion goes through
//! [`Param`](crate::marshal::Param) and rejects unknown values with
//! `FT_ERR_INVALID_PARAM`.

use crate::marshal::Param;
use crate::spec::{EnumSpec, ParamKind};
use crate::status::Status;
use ferro_ta_core::options::digital::DigitalKind;
use ferro_ta_core::options::{OptionKind, PricingModel};
use ferro_ta_core::sweep::BuiltinStrategy;

pub const OPTION_KIND: EnumSpec = EnumSpec {
    name: "OptionKind",
    c_prefix: "FT_OPTION",
    values: &[("CALL", 0), ("PUT", 1)],
};

pub const DIGITAL_KIND: EnumSpec = EnumSpec {
    name: "DigitalKind",
    c_prefix: "FT_DIGITAL",
    values: &[("CASH_OR_NOTHING", 0), ("ASSET_OR_NOTHING", 1)],
};

pub const PRICING_MODEL: EnumSpec = EnumSpec {
    name: "PricingModel",
    c_prefix: "FT_MODEL",
    values: &[("BLACK_SCHOLES", 0), ("BLACK_76", 1)],
};

/// Built-in strategies for `ft_sweep_builtin`.
pub const SWEEP_STRATEGY: EnumSpec = EnumSpec {
    name: "SweepStrategy",
    c_prefix: "FT_STRATEGY",
    values: &[
        ("RSI_THRESHOLD", 0),
        ("SMA_CROSSOVER", 1),
        ("MACD_CROSSOVER", 2),
    ],
};

/// Column indices of the per-combination metric matrix written by the sweep
/// kernels (`out_metrics[combo * n_metric_cols + FT_METRIC_SHARPE]`, with
/// `n_metric_cols` from `ft_sweep_n_metrics()`).
/// Not a parameter type; a named index set.
pub const SWEEP_METRIC: EnumSpec = EnumSpec {
    name: "SweepMetric",
    c_prefix: "FT_METRIC",
    values: &[
        ("TOTAL_RETURN", 0),
        ("CAGR", 1),
        ("ANNUALIZED_VOL", 2),
        ("SHARPE", 3),
        ("SORTINO", 4),
        ("CALMAR", 5),
        ("MAX_DRAWDOWN", 6),
        ("AVG_DRAWDOWN", 7),
        ("MAX_DRAWDOWN_DURATION_BARS", 8),
        ("AVG_DRAWDOWN_DURATION_BARS", 9),
        ("ULCER_INDEX", 10),
        ("OMEGA_RATIO", 11),
        ("WIN_RATE", 12),
        ("PROFIT_FACTOR", 13),
        ("R_EXPECTANCY", 14),
        ("AVG_WIN", 15),
        ("AVG_LOSS", 16),
        ("TAIL_RATIO", 17),
        ("SKEWNESS", 18),
        ("KURTOSIS", 19),
        ("BEST_BAR", 20),
        ("WORST_BAR", 21),
        ("N_TRADES", 22),
        ("N_POSITION_CHANGES", 23),
    ],
};

/// Every C enum, for the spec dump.
pub const ALL_ENUMS: &[&EnumSpec] = &[
    &OPTION_KIND,
    &DIGITAL_KIND,
    &PRICING_MODEL,
    &SWEEP_STRATEGY,
    &SWEEP_METRIC,
];

impl Param<i32> for BuiltinStrategy {
    fn from_raw(raw: i32, _kind: ParamKind) -> Result<Self, Status> {
        match raw {
            0 => Ok(BuiltinStrategy::RsiThreshold),
            1 => Ok(BuiltinStrategy::SmaCrossover),
            2 => Ok(BuiltinStrategy::MacdCrossover),
            _ => Err(Status::InvalidParam),
        }
    }
}

impl Param<i32> for OptionKind {
    fn from_raw(raw: i32, _kind: ParamKind) -> Result<Self, Status> {
        match raw {
            0 => Ok(OptionKind::Call),
            1 => Ok(OptionKind::Put),
            _ => Err(Status::InvalidParam),
        }
    }
}

impl Param<i32> for DigitalKind {
    fn from_raw(raw: i32, _kind: ParamKind) -> Result<Self, Status> {
        match raw {
            0 => Ok(DigitalKind::CashOrNothing),
            1 => Ok(DigitalKind::AssetOrNothing),
            _ => Err(Status::InvalidParam),
        }
    }
}

impl Param<i32> for PricingModel {
    fn from_raw(raw: i32, _kind: ParamKind) -> Result<Self, Status> {
        match raw {
            0 => Ok(PricingModel::BlackScholes),
            1 => Ok(PricingModel::Black76),
            _ => Err(Status::InvalidParam),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferro_ta_core::backtest::BacktestMetrics;

    #[test]
    fn sweep_metric_enum_tracks_core_metric_order() {
        let names: Vec<String> = SWEEP_METRIC
            .values
            .iter()
            .map(|(n, _)| n.to_lowercase())
            .collect();
        assert_eq!(names, BacktestMetrics::SCALAR_FIELD_NAMES);
        assert!(SWEEP_METRIC
            .values
            .iter()
            .enumerate()
            .all(|(i, (_, v))| *v == i as i32));
    }

    #[test]
    fn sweep_strategy_enum_matches_core_names() {
        for (name, value) in SWEEP_STRATEGY.values {
            let strat = BuiltinStrategy::from_raw(*value, ParamKind::Float).unwrap();
            let canonical = BuiltinStrategy::NAMES[*value as usize];
            assert_eq!(
                BuiltinStrategy::from_name(canonical).unwrap(),
                strat,
                "{name}"
            );
        }
    }
}
