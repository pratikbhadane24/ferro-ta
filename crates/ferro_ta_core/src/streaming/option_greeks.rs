//! Streaming implied volatility and greeks for one option contract.
//!
//! Each tick (option price, underlying, time to expiry) is inverted to an
//! implied volatility, seeded from the previous tick's IV so the solver
//! starts next to the answer, and greeks are evaluated at that IV.

use super::{StreamValue, StreamingError};
use crate::options::greeks::model_greeks;
use crate::options::iv::implied_volatility;
use crate::options::{IvSolverConfig, OptionContract, OptionEvaluation};
use crate::options::{OptionKind, PricingModel};

/// Solver start when no previous IV exists (matches the batch default).
const DEFAULT_IV_SEED: f64 = 0.2;
const IV_TOLERANCE: f64 = 1e-8;
const IV_MAX_ITERATIONS: usize = 100;

/// `(iv, delta, gamma, vega, theta, rho)`.
pub type OptionGreeksOutput = (f64, f64, f64, f64, f64, f64);

/// Live IV and greeks for a fixed contract (model, kind, strike, rate, carry).
///
/// Input per tick: `(option_price, underlying, time_to_expiry_years)`.
/// When no volatility reproduces the price (e.g. below intrinsic) the tick
/// yields all-NaN and the IV seed is left unchanged.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingOptionGreeks {
    model: PricingModel,
    kind: OptionKind,
    strike: f64,
    rate: f64,
    carry: f64,
    #[cfg_attr(feature = "serde", serde(with = "super::snapshot::nan_as_null"))]
    last_iv: f64,
}

impl StreamingOptionGreeks {
    pub fn new(
        model: PricingModel,
        kind: OptionKind,
        strike: f64,
        rate: f64,
        carry: f64,
    ) -> Result<Self, StreamingError> {
        if !strike.is_finite() || strike <= 0.0 {
            return Err(StreamingError(format!(
                "strike must be finite and > 0, got {strike}"
            )));
        }
        if !rate.is_finite() || !carry.is_finite() {
            return Err(StreamingError(format!(
                "rate and carry must be finite, got rate={rate}, carry={carry}"
            )));
        }
        Ok(Self {
            model,
            kind,
            strike,
            rate,
            carry,
            last_iv: f64::NAN,
        })
    }

    /// Feed one tick; return `(iv, delta, gamma, vega, theta, rho)`.
    pub fn update(
        &mut self,
        option_price: f64,
        underlying: f64,
        time_to_expiry: f64,
    ) -> OptionGreeksOutput {
        let contract = OptionContract {
            model: self.model,
            underlying,
            strike: self.strike,
            rate: self.rate,
            carry: self.carry,
            time_to_expiry,
            kind: self.kind,
        };
        let config = IvSolverConfig {
            initial_guess: self.seed(),
            tolerance: IV_TOLERANCE,
            max_iterations: IV_MAX_ITERATIONS,
        };
        let iv = implied_volatility(contract, option_price, config);
        if !iv.is_finite() {
            return OptionGreeksOutput::nan();
        }
        self.last_iv = iv;
        let g = model_greeks(OptionEvaluation {
            contract,
            volatility: iv,
        });
        (iv, g.delta, g.gamma, g.vega, g.theta, g.rho)
    }

    /// Previous positive IV, else the batch default.
    fn seed(&self) -> f64 {
        if self.last_iv.is_finite() && self.last_iv > 0.0 {
            self.last_iv
        } else {
            DEFAULT_IV_SEED
        }
    }

    pub fn reset(&mut self) {
        self.last_iv = f64::NAN;
    }

    /// Last successfully solved IV (NaN before the first).
    pub fn last_iv(&self) -> f64 {
        self.last_iv
    }

    pub fn strike(&self) -> f64 {
        self.strike
    }

    /// Re-check invariants (used when restoring a snapshot).
    pub fn validate(&self) -> Result<(), StreamingError> {
        Self::new(self.model, self.kind, self.strike, self.rate, self.carry).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::greeks::model_greeks;
    use crate::options::pricing::model_price;
    use crate::options::{OptionContract, OptionEvaluation};
    use crate::streaming::Live;

    const STRIKE: f64 = 100.0;
    const RATE: f64 = 0.05;

    fn contract(underlying: f64, t: f64) -> OptionContract {
        OptionContract {
            model: PricingModel::BlackScholes,
            underlying,
            strike: STRIKE,
            rate: RATE,
            carry: 0.0,
            time_to_expiry: t,
            kind: OptionKind::Call,
        }
    }

    fn price_at(vol: f64, underlying: f64, t: f64) -> f64 {
        model_price(OptionEvaluation {
            contract: contract(underlying, t),
            volatility: vol,
        })
    }

    fn call() -> StreamingOptionGreeks {
        StreamingOptionGreeks::new(
            PricingModel::BlackScholes,
            OptionKind::Call,
            STRIKE,
            RATE,
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn matches_batch_iv_and_greeks() {
        let mut stream = call();
        let (iv, delta, gamma, vega, theta, rho) =
            stream.update(price_at(0.25, 102.0, 0.5), 102.0, 0.5);
        assert!((iv - 0.25).abs() < 1e-8, "iv {iv}");
        let g = model_greeks(OptionEvaluation {
            contract: contract(102.0, 0.5),
            volatility: iv,
        });
        assert_eq!(
            (delta, gamma, vega, theta, rho),
            (g.delta, g.gamma, g.vega, g.theta, g.rho)
        );
    }

    #[test]
    fn seeds_from_previous_iv() {
        let mut stream = call();
        assert!(stream.last_iv().is_nan());
        stream.update(price_at(0.25, 100.0, 0.3), 100.0, 0.3);
        assert!((stream.last_iv() - 0.25).abs() < 1e-8);
        let (iv, ..) = stream.update(price_at(0.6, 101.0, 0.3), 101.0, 0.3);
        assert!((iv - 0.6).abs() < 1e-8, "large vol jump still solves: {iv}");
        assert_eq!(stream.last_iv(), iv);
    }

    #[test]
    fn failed_solve_keeps_previous_seed() {
        let mut stream = call();
        stream.update(price_at(0.3, 100.0, 0.5), 100.0, 0.5);
        let seed = stream.last_iv();
        // Below intrinsic (S - K·e^{-rT} ≈ 22.5): no volatility prices it.
        let out = stream.update(1.0, 125.0, 0.5);
        assert!(StreamValue::has_nan(&out) && out.0.is_nan() && out.5.is_nan());
        assert_eq!(stream.last_iv(), seed);
        let (iv, ..) = stream.update(price_at(0.35, 100.0, 0.5), 100.0, 0.5);
        assert!((iv - 0.35).abs() < 1e-8);
    }

    #[test]
    fn non_finite_tick_skipped_via_live() {
        let mut live = Live::new(call());
        live.update((price_at(0.25, 100.0, 0.5), 100.0, 0.5), true);
        let before = live.inner().last_iv();
        assert!(live.update((f64::NAN, 100.0, 0.5), true).0.is_nan());
        assert!(live.update((5.0, f64::INFINITY, 0.5), true).0.is_nan());
        assert_eq!(live.inner().last_iv(), before);
        assert_eq!(live.count(), 1);
    }

    #[test]
    fn invalid_params_rejected() {
        let new = |k: f64, r: f64, c: f64| {
            StreamingOptionGreeks::new(PricingModel::BlackScholes, OptionKind::Put, k, r, c)
        };
        assert!(new(0.0, 0.0, 0.0).is_err());
        assert!(new(-5.0, 0.0, 0.0).is_err());
        assert!(new(f64::NAN, 0.0, 0.0).is_err());
        assert!(new(100.0, f64::INFINITY, 0.0).is_err());
        assert!(new(100.0, 0.0, f64::NAN).is_err());
    }
}
