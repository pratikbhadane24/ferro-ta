//! Registry families: one file per batch-module group, each holding its
//! `stream_family_<x>!` entries and any native structs. See
//! `streaming::registry` for the entry grammar.

pub mod directional;
pub mod ma_osc;
pub mod markets;
pub mod math;
pub mod momentum;
pub mod overlap;
pub mod stats;
pub mod volume;

pub use directional::*;
pub use ma_osc::*;
pub use markets::*;
pub use math::*;
pub use momentum::*;
pub use overlap::*;
pub use stats::*;
pub use volume::*;

#[cfg(test)]
mod drift_tests;
#[cfg(test)]
mod markets_tests;
#[cfg(test)]
mod momentum_tests;
#[cfg(test)]
mod tests;
