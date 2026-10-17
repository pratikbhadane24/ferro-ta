//! Built-in strategies addressable by name, so a whole sweep (signal
//! generation included) runs in Rust from any language binding.

use crate::backtest::{macd_crossover_signals, rsi_threshold_signals, sma_crossover_signals};

/// Largest look-back accepted for integer period parameters.
const MAX_PERIOD: f64 = 16_777_216.0; // 2^24, same cap as the C ABI

/// Signal generators available to [`super::sweep_builtin`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinStrategy {
    /// +1 when RSI <= oversold, -1 when RSI >= overbought.
    /// Params: `timeperiod` (14), `oversold` (30), `overbought` (70).
    RsiThreshold,
    /// +1 when fast SMA > slow SMA, -1 below. Params: `fast` (10), `slow` (30).
    SmaCrossover,
    /// +1 when MACD > signal line, -1 below.
    /// Params: `fastperiod` (12), `slowperiod` (26), `signalperiod` (9).
    MacdCrossover,
}

/// One parameter: name, default, and whether it must be a whole number >= 1.
struct ParamSpec {
    name: &'static str,
    default: f64,
    is_period: bool,
}

const RSI_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        name: "timeperiod",
        default: 14.0,
        is_period: true,
    },
    ParamSpec {
        name: "oversold",
        default: 30.0,
        is_period: false,
    },
    ParamSpec {
        name: "overbought",
        default: 70.0,
        is_period: false,
    },
];
const SMA_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        name: "fast",
        default: 10.0,
        is_period: true,
    },
    ParamSpec {
        name: "slow",
        default: 30.0,
        is_period: true,
    },
];
const MACD_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        name: "fastperiod",
        default: 12.0,
        is_period: true,
    },
    ParamSpec {
        name: "slowperiod",
        default: 26.0,
        is_period: true,
    },
    ParamSpec {
        name: "signalperiod",
        default: 9.0,
        is_period: true,
    },
];

impl BuiltinStrategy {
    /// Canonical names accepted by [`Self::from_name`] (aliases aside).
    pub const NAMES: [&'static str; 3] = ["rsi_30_70", "sma_crossover", "macd_crossover"];

    /// Look up a strategy by name. Accepts the Python registry names
    /// (`rsi_30_70`, `sma_crossover`, `macd_crossover`) and short aliases.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "rsi_30_70" | "rsi_threshold" | "rsi" => Ok(Self::RsiThreshold),
            "sma_crossover" | "sma" => Ok(Self::SmaCrossover),
            "macd_crossover" | "macd" => Ok(Self::MacdCrossover),
            other => Err(format!(
                "unknown built-in strategy '{other}'; expected one of \
                 rsi_30_70, sma_crossover, macd_crossover"
            )),
        }
    }

    fn specs(self) -> &'static [ParamSpec] {
        match self {
            Self::RsiThreshold => RSI_PARAMS,
            Self::SmaCrossover => SMA_PARAMS,
            Self::MacdCrossover => MACD_PARAMS,
        }
    }

    /// Parameter names in canonical order.
    pub fn param_names(self) -> Vec<&'static str> {
        self.specs().iter().map(|p| p.name).collect()
    }

    /// Check every name in `names` is a parameter of this strategy.
    pub fn validate_names(self, names: &[&str]) -> Result<(), String> {
        match names
            .iter()
            .find(|n| !self.specs().iter().any(|p| p.name == **n))
        {
            Some(bad) => Err(format!(
                "unknown parameter '{bad}' for this strategy; expected any of {:?}",
                self.param_names()
            )),
            None => Ok(()),
        }
    }

    /// Resolve `values` (aligned with `names`) over the defaults.
    ///
    /// Returns `None` when the combination violates the strategy's
    /// constraints (non-integer or out-of-range period, `fast >= slow`,
    /// `oversold >= overbought`): such grid points are skipped, not errors.
    /// Call [`Self::validate_names`] first.
    pub fn resolve(self, names: &[&str], values: &[f64]) -> Option<Vec<f64>> {
        let resolved: Vec<f64> = self
            .specs()
            .iter()
            .map(|p| {
                names
                    .iter()
                    .position(|n| *n == p.name)
                    .map_or(p.default, |k| values[k])
            })
            .collect();
        let periods_ok = self
            .specs()
            .iter()
            .zip(&resolved)
            .all(|(p, &v)| v.is_finite() && (!p.is_period || is_period(v)));
        let ordered = match self {
            Self::RsiThreshold => resolved[1] < resolved[2],
            Self::SmaCrossover | Self::MacdCrossover => resolved[0] < resolved[1],
        };
        (periods_ok && ordered).then_some(resolved)
    }

    /// Signals for fully resolved parameters (see [`Self::resolve`]).
    pub fn signals(self, close: &[f64], p: &[f64]) -> Result<Vec<f64>, String> {
        match self {
            Self::RsiThreshold => Ok(rsi_threshold_signals(close, p[0] as usize, p[1], p[2])),
            Self::SmaCrossover => sma_crossover_signals(close, p[0] as usize, p[1] as usize),
            Self::MacdCrossover => {
                macd_crossover_signals(close, p[0] as usize, p[1] as usize, p[2] as usize)
            }
        }
    }
}

fn is_period(v: f64) -> bool {
    (1.0..=MAX_PERIOD).contains(&v) && v.fract() == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_aliases() {
        assert_eq!(
            BuiltinStrategy::from_name("rsi_30_70"),
            Ok(BuiltinStrategy::RsiThreshold)
        );
        assert_eq!(
            BuiltinStrategy::from_name("sma"),
            Ok(BuiltinStrategy::SmaCrossover)
        );
        assert_eq!(
            BuiltinStrategy::from_name("macd_crossover"),
            Ok(BuiltinStrategy::MacdCrossover)
        );
        assert!(BuiltinStrategy::from_name("nope").is_err());
    }

    #[test]
    fn resolve_fills_defaults_and_rejects_constraints() {
        let s = BuiltinStrategy::SmaCrossover;
        assert_eq!(s.resolve(&["fast"], &[5.0]), Some(vec![5.0, 30.0]));
        assert_eq!(s.resolve(&["fast", "slow"], &[20.0, 10.0]), None);
        assert_eq!(s.resolve(&["fast"], &[2.5]), None);
        assert_eq!(s.resolve(&["fast"], &[0.0]), None);
        let r = BuiltinStrategy::RsiThreshold;
        assert_eq!(r.resolve(&["oversold", "overbought"], &[70.0, 30.0]), None);
        assert_eq!(
            r.resolve(&["oversold"], &[20.0]),
            Some(vec![14.0, 20.0, 70.0])
        );
    }

    #[test]
    fn validate_names_rejects_unknown() {
        assert!(BuiltinStrategy::MacdCrossover
            .validate_names(&["fastperiod"])
            .is_ok());
        assert!(BuiltinStrategy::MacdCrossover
            .validate_names(&["fast"])
            .is_err());
    }

    fn assert_same(a: Vec<f64>, b: Vec<f64>) {
        assert_eq!(a.len(), b.len());
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan())));
    }

    #[test]
    fn signals_match_direct_calls() {
        let close: Vec<f64> = (0..120)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let s = BuiltinStrategy::SmaCrossover;
        assert_same(
            s.signals(&close, &[5.0, 20.0]).unwrap(),
            sma_crossover_signals(&close, 5, 20).unwrap(),
        );
        let m = BuiltinStrategy::MacdCrossover;
        assert_same(
            m.signals(&close, &[6.0, 13.0, 5.0]).unwrap(),
            macd_crossover_signals(&close, 6, 13, 5).unwrap(),
        );
    }
}
