//! Extended indicators — pure Rust implementations (no PyO3, no numpy).
//!
//! These indicators are not part of TA-Lib and provide additional technical
//! analysis capabilities. All functions operate on `&[f64]` slices and return
//! `Vec<f64>` (or tuples thereof).
//!
//! New kernels land in the category modules (`trend`, `momentum`, …). The
//! pre-existing catalog now lives in its own kernel modules (`vwap`,
//! `channels`, `choppiness`, `hull`, `ichimoku`, `pivots`).
//!
//! Two conventions hold throughout that catalog. Outputs are built with
//! `Vec::with_capacity` + `push` wherever the writes are sequential, because
//! the `vec![f64::NAN; n]` prologue is a genuine store pass (`NaN` is not a
//! zero page) that the kernel then overwrites. And the channel kernels
//! (Donchian, Chandelier Exit, Choppiness, Ichimoku) drive
//! `crate::rolling`'s monotonic deques directly — one traversal, no
//! `sliding_max` / `sliding_min` `Vec` per extreme and no fix-up loop.

#![allow(clippy::too_many_arguments)]

mod channels;
mod choppiness;
mod hull;
mod hybrid;
mod ichimoku;
mod momentum;
mod oscillators;
mod pivots;
mod stat;
mod trend;
mod volatility;
mod volume;
mod vwap;

#[cfg(test)]
mod test_support;

pub use channels::*;
pub use choppiness::*;
pub use hull::*;
pub use hybrid::*;
pub use ichimoku::*;
pub use momentum::*;
pub use oscillators::*;
pub use pivots::*;
pub use stat::*;
pub use trend::*;
pub use volatility::*;
pub use volume::*;
pub use vwap::*;
