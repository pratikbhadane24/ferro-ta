//! Registry families: one file per batch-module group, each holding its
//! `stream_family_<x>!` entries and any native structs. See
//! `streaming::registry` for the entry grammar.

pub mod cycle;
pub mod directional;
pub mod extended_bands;
mod extended_common;
pub mod extended_osc;
pub mod extended_trend;
pub mod extended_volume;
pub mod ma_osc;
pub mod markets;
pub mod math;
pub mod momentum;
pub mod overlap;
pub mod pattern;
pub mod sar;
pub mod stats;
pub mod volume;

pub use cycle::*;
pub use directional::*;
pub use extended_bands::*;
pub use extended_osc::*;
pub use extended_trend::*;
pub use extended_volume::*;
pub use ma_osc::*;
pub use markets::*;
pub use math::*;
pub use momentum::*;
pub use overlap::*;
pub use pattern::*;
pub use sar::*;
pub use stats::*;
pub use volume::*;

#[cfg(test)]
mod batch_exact_tests;
#[cfg(test)]
mod drift_tests;
#[cfg(test)]
mod extended_drift_tests;
#[cfg(test)]
mod extended_tests;
#[cfg(test)]
mod hard_tests;
#[cfg(test)]
mod markets_tests;
#[cfg(test)]
mod momentum_tests;
#[cfg(test)]
mod tests;
