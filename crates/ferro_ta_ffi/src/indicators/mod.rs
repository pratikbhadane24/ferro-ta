//! Array indicator exports, one module per `ferro_ta_core` module.

pub mod cycle;
pub mod extended;
pub mod futures;
pub mod math;
pub mod math_ops;
pub mod misc;
pub mod momentum;
pub mod options;
pub mod overlap;
pub mod pattern;
pub mod portfolio;
pub mod price_transform;
pub mod regime;
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
    options::SPECS,
    futures::SPECS,
    regime::SPECS,
    portfolio::SPECS,
    misc::SPECS,
];
