//! Shared rolling-window machinery (crate-internal).
//!
//! Indicator kernels repeatedly need the same four primitives over a sliding
//! window: the extreme value, the *index* of the extreme, a running sum, and a
//! running variance. This module holds one vetted copy of each, so kernels can
//! compose them in a **single traversal** instead of materializing an
//! intermediate `Vec` per pass.
//!
//! # Why not `VecDeque`
//!
//! The monotonic-deque sliding extremum is O(n), but a `VecDeque<usize>`
//! implementation pays for it twice: every front/back access does ring-buffer
//! mask arithmetic, and the pop predicate has to *gather* `real[j]` to recover
//! the value behind a stored index. [`MaxDeque`] / [`MinDeque`] instead keep
//! **parallel `Vec<f64>` values and `Vec<u32>` indices** with a `head` cursor,
//! so the hot pop predicate reads `vals[len - 1]` — sequential, in L1, no
//! gather. `pop_front` is `head += 1`; the buffers are compacted only when the
//! deque drains completely (the common case, free) or when `head` has walked a
//! full window (amortized one element move per bar), which bounds both buffers
//! at `2 * timeperiod + 2` entries with no reallocation.
//!
//! # Bit-identity contract
//!
//! The comparison predicates here are byte-for-byte the ones the previous
//! `math::sliding_max` / `math::sliding_min` used, and that is **load-bearing**:
//!
//! * **NaN behaviour.** A `NaN` satisfies neither `<=` nor `>=`, so it is never
//!   popped and can legitimately surface as the window extreme. Switching to
//!   `f64::max` (which ignores NaN) would silently change `DONCHIAN`, `AROON`
//!   and `WILLR` output.
//! * **Tie-breaking.** `<=` (not `<`) means an incoming equal value pops the
//!   entire monotonic run *including the front*, so among equal extremes the
//!   **most recent** index wins. `AROON` depends on exactly this; inverting it
//!   shifts `AROON` output by `100 / period`.
//!
//! The `tests` module keeps a private verbatim copy of each pre-existing
//! implementation and asserts `to_bits()` equality against it over adversarial
//! inputs (plateaus, monotone runs, NaN/±inf, degenerate periods).

// This module is the shared machinery; several primitives land here one commit
// ahead of the kernels that consume them. The alternative is a scattering of
// per-item `#[allow]`s that would all have to be removed again.
#![allow(dead_code)]

mod deque;
mod reseed;
mod sum;
mod variance;

pub(crate) use deque::*;
pub(crate) use reseed::*;
pub(crate) use sum::*;
pub(crate) use variance::*;

/// Drivers shared by the reseed-policy tests of more than one accumulator.
#[cfg(test)]
mod test_support {
    use super::{RollingSum, RollingVariance};

    /// Drive a `RollingVariance` over `data`, returning every full window's
    /// reported variance and the number of exact recomputes it took.
    ///
    /// Deliberately open-codes `advance` so the reseed count is observable:
    /// a policy that is correct but fires constantly turns an O(n) kernel back
    /// into O(n * p), and no correctness assertion would notice.
    pub(super) fn drive_variance(data: &[f64], p: usize) -> (Vec<f64>, usize) {
        let mut acc = RollingVariance::new(&data[..p]);
        let mut out = vec![acc.population_var()];
        let mut reseeds = 0usize;
        for i in p..data.len() {
            acc.push_pop(data[i], data[i - p]);
            if acc.needs_reseed() {
                acc.reseed(&data[i + 1 - p..=i]);
                reseeds += 1;
            }
            out.push(acc.population_var());
        }
        (out, reseeds)
    }

    /// Same for `RollingSum`.
    pub(super) fn drive_sum(data: &[f64], p: usize) -> (Vec<f64>, usize) {
        let mut acc = RollingSum::new(&data[..p]);
        let mut out = vec![acc.value()];
        let mut reseeds = 0usize;
        for i in p..data.len() {
            acc.push_pop(data[i], data[i - p]);
            if acc.needs_reseed() {
                acc.reseed(&data[i + 1 - p..=i]);
                reseeds += 1;
            }
            out.push(acc.value());
        }
        (out, reseeds)
    }
}
