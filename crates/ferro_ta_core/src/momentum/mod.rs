//! Momentum indicators.
//!
//! # Wilder smoothing and the hoisted reciprocal
//!
//! `rsi` and `cmo` advance Wilder's average as
//! `avg = (avg * (p - 1) + x) * inv_p` with `inv_p = 1 / p` precomputed,
//! rather than the textbook `(avg * (p - 1) + x) / p`. `p` is loop-invariant,
//! but replacing a divide with a reciprocal multiply is not a transform LLVM
//! may perform without fast-math, and the divide sits *in* the serial
//! dependency chain: mul(4) + add(4) + div(~14) is ~22 cycles per bar against
//! ~12 for the multiply form.
//!
//! `1 / p` is inexact unless `p` is a power of two, so each step can differ
//! from the division form by at most one ulp. This is safe because the
//! recurrence is a **contraction** — the previous average is scaled by
//! `(p - 1) / p < 1` every step — so per-step rounding decays geometrically
//! instead of accumulating. Steady-state relative error is `O(p * eps)`,
//! around `3e-15` at `p = 14`, seven orders below the `atol = 1e-8` the
//! TA-Lib conformance suite gates RSI at.
//!
//! `adx_inner` and `dm_only_inner` already carry the equivalent hoist as
//! `decay = (period - 1) / period`, so they need no change.

mod adx;
mod aroon;
mod price_osc;
mod range;
mod roc;
mod rsi;
mod stoch;
mod ultosc;

#[cfg(test)]
mod test_support;

pub use adx::*;
pub use aroon::*;
pub use price_osc::*;
pub use range::*;
pub use roc::*;
pub use rsi::*;
pub use stoch::*;
pub use ultosc::*;
