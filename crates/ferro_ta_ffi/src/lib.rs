//! ferro_ta_ffi — stable C ABI over `ferro_ta_core`.
//!
//! This crate is the shared foundation for every C-ABI language binding (Go via
//! cgo, C/C++, C# P/Invoke, JVM Panama, Ruby FFI, ...). `ferro_ta_core` stays
//! `#![forbid(unsafe_code)]`; all pointer handling lives here.
//!
//! # ABI conventions
//!
//! * Every symbol is prefixed `ft_` and returns an `int32_t` status
//!   (`FT_OK == 0`); see [`status`].
//! * Inputs are `const double *` arrays sharing one `size_t len`.
//! * Outputs are caller-allocated arrays of length `len`, written only on
//!   success. NULL is allowed for every array when `len == 0`.
//! * Periods are `int64_t` (so negative values from signed host languages are
//!   rejected, not wrapped), MA types `int32_t`, other scalars `double`.
//! * Panics in the core are contained and reported as `FT_ERR_PANIC`.
//!
//! Every export is described in [`all_specs`]; `examples/dump_spec.rs` writes
//! that table to `ffi_spec.json`, the single source the header and language
//! wrappers are generated from.

#![deny(unsafe_op_in_unsafe_fn)]

mod macros;
pub mod marshal;
pub mod spec;
pub mod status;
pub mod streaming;
mod validation;

pub mod indicators;

pub use indicators::cycle::*;
pub use indicators::extended::*;
pub use indicators::futures::*;
pub use indicators::math::*;
pub use indicators::math_ops::*;
pub use indicators::misc::*;
pub use indicators::momentum::*;
pub use indicators::options::*;
pub use indicators::overlap::*;
pub use indicators::pattern::*;
pub use indicators::portfolio::*;
pub use indicators::price_transform::*;
pub use indicators::regime::*;
pub use indicators::statistic::*;
pub use indicators::volatility::*;
pub use indicators::volume::*;
pub use status::*;
pub use streaming::*;

/// Signature metadata for every array export.
pub fn all_specs() -> impl Iterator<Item = &'static spec::FnSpec> {
    indicators::MODULE_SPECS
        .iter()
        .flat_map(|specs| specs.iter())
}

/// Signature metadata for every streaming (opaque handle) export.
pub fn all_stream_specs() -> impl Iterator<Item = &'static spec::StreamSpec> {
    streaming::STREAM_SPECS.iter()
}
