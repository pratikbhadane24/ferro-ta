//! Registry families: one file per batch-module group, each holding its
//! `stream_family_<x>!` entries and any native structs. See
//! `streaming::registry` for the entry grammar.

pub mod math;

pub use math::*;
