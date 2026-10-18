//! Registry families: one file per batch-module group, each holding its
//! `stream_family_<x>!` entries and any native structs. See
//! `streaming::registry` for the entry grammar.

pub mod ma_osc;
pub mod math;
pub mod overlap;

pub use ma_osc::*;
pub use math::*;
pub use overlap::*;

#[cfg(test)]
mod tests;
