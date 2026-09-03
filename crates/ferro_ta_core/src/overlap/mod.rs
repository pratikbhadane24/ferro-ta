//! Overlap studies — moving averages and trend indicators.
//!
//! All functions return a `Vec<f64>` of the same length as the input.
//! Leading values are `f64::NAN` for the warm-up period.
//!
//! # Output construction
//!
//! Kernels that fill their output strictly left-to-right build it with
//! `Vec::with_capacity` + `resize(warmup, NAN)` + `push`, rather than
//! `vec![f64::NAN; n]` followed by overwriting nearly every slot. `NaN` is not
//! the all-zero bit pattern, so the `vec!` form is a real store pass over the
//! whole array — 800 KB at 100k bars — that the kernel then repeats. The values
//! written are the same ones in the same order, so this is bit-identical.

mod bands;
mod dispatch;
mod ema;
mod kama;
mod macd;
mod mama;
mod midpoint;
mod sar;
mod sma;
mod trima;
mod wma;

pub use bands::*;
pub use dispatch::*;
pub use ema::*;
pub use kama::*;
pub use macd::*;
pub use mama::*;
pub use midpoint::*;
pub use sar::*;
pub use sma::*;
pub use trima::*;
pub use wma::*;

#[cfg(test)]
pub(super) mod test_support {
    // -----------------------------------------------------------------------
    // Equivalence oracles: verbatim copies of the pre-optimization kernels.
    //
    // `sar`/`ema`/`ema_from_first_finite`/`midpoint`/`midprice` changed only in
    // how the output vector is built, so those are asserted with `to_bits()`.
    // `wma`/`trima`/`kama` changed arithmetic (reciprocal multiply, running
    // numerator, rolling volatility sum) and carry a stated tolerance.
    // -----------------------------------------------------------------------

    // -- Test fixtures -----------------------------------------------------

    /// Deterministic random walk around 100.0 (SplitMix-style LCG, no deps).
    pub(super) fn synthetic_series(n: usize) -> Vec<f64> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut price = 100.0f64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                price += u - 0.5;
                price
            })
            .collect()
    }

    /// `high`/`low` bracketing a synthetic close, with enough movement to make
    /// the SAR flip direction many times.
    pub(super) fn synthetic_hl(n: usize) -> (Vec<f64>, Vec<f64>) {
        let close = synthetic_series(n);
        let high = close.iter().map(|c| c + 0.75).collect();
        let low = close.iter().map(|c| c - 0.75).collect();
        (high, low)
    }

    /// Assert bit-for-bit equality (so `NaN` slots must match too).
    pub(super) fn assert_bits_eq(got: &[f64], want: &[f64], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}: bit mismatch at {i}: got {g} want {w}"
            );
        }
    }

    /// Assert `NaN` slots line up and finite slots agree within `atol`.
    pub(super) fn assert_close(got: &[f64], want: &[f64], atol: f64, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.is_nan(),
                w.is_nan(),
                "{label}: NaN placement differs at {i}: got {g} want {w}"
            );
            if !w.is_nan() {
                assert!(
                    (g - w).abs() <= atol,
                    "{label}: at {i} got {g} want {w} (delta {})",
                    (g - w).abs()
                );
            }
        }
    }
}
