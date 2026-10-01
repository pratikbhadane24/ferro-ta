//! Array indicator exports, one module per `ferro_ta_core` module.

pub mod cycle;
pub mod extended;
pub mod math;
pub mod math_ops;
pub mod momentum;
pub mod overlap;
pub mod pattern;
pub mod price_transform;
pub mod statistic;
pub mod volatility;
pub mod volume;

use crate::spec::FnSpec;

/// Every module's `SPECS` table, in a stable order.
pub const MODULE_SPECS: &[&[FnSpec]] = &[
    overlap::SPECS,
    momentum::SPECS,
    volatility::SPECS,
    volume::SPECS,
    cycle::SPECS,
    statistic::SPECS,
    math::SPECS,
    math_ops::SPECS,
    price_transform::SPECS,
    pattern::SPECS,
    extended::SPECS,
];
