//! Extended indicators — pure Rust implementations (no PyO3, no numpy).
//!
//! These indicators are not part of TA-Lib and provide additional technical
//! analysis capabilities. All functions operate on `&[f64]` slices and return
//! `Vec<f64>` (or tuples thereof).
//!
//! New kernels land in the category modules (`trend`, `momentum`, …). The
//! pre-existing catalog lives in [`existing`].

#![allow(clippy::too_many_arguments)]

mod existing;
mod hybrid;
mod momentum;
mod oscillators;
mod stat;
mod trend;
mod volatility;
mod volume;

pub use existing::*;
pub use hybrid::*;
pub use momentum::*;
pub use oscillators::*;
pub use stat::*;
pub use trend::*;
pub use volatility::*;
pub use volume::*;
