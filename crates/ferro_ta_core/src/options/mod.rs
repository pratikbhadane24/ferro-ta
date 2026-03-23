//! Options analytics core.
//!
//! This module contains pricing, Greeks, implied volatility inversion,
//! IV-series helpers, and smile/chain utilities. The public API is scalar-first
//! and is used by the PyO3 bridge to build vectorized batch functions.

pub mod chain;
pub mod greeks;
pub mod iv;
pub mod normal;
pub mod pricing;
pub mod surface;

/// Option side.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptionKind {
    /// Call option.
    Call,
    /// Put option.
    Put,
}

impl OptionKind {
    /// Returns +1 for calls and -1 for puts.
    pub fn sign(self) -> f64 {
        match self {
            Self::Call => 1.0,
            Self::Put => -1.0,
        }
    }
}

/// Supported pricing models.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PricingModel {
    /// Black-Scholes-Merton with continuous carry/dividend yield.
    BlackScholes,
    /// Black-76 using the forward price as the underlying input.
    Black76,
}

/// Primary first-order Greeks returned by the pricing engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

/// Shared contract fields for model-based option analytics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptionContract {
    pub model: PricingModel,
    pub underlying: f64,
    pub strike: f64,
    pub rate: f64,
    pub carry: f64,
    pub time_to_expiry: f64,
    pub kind: OptionKind,
}

/// Contract plus volatility for pricing and Greeks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptionEvaluation {
    pub contract: OptionContract,
    pub volatility: f64,
}

/// Solver configuration for implied volatility inversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IvSolverConfig {
    pub initial_guess: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

/// Shared context for strike selection and smile analytics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainGreeksContext {
    pub model: PricingModel,
    pub reference_price: f64,
    pub rate: f64,
    pub carry: f64,
    pub time_to_expiry: f64,
    pub kind: OptionKind,
}
