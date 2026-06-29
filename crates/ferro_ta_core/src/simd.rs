//! Runtime-dispatched SIMD primitives.
//!
//! Each public reduction here is compiled into several CPU-feature-specific
//! variants (baseline, SSE, AVX2/FMA, AVX-512 on x86_64; NEON on aarch64; …)
//! by [`multiversion`]. The fastest variant the *current* CPU supports is
//! chosen at runtime via CPUID. This gives one binary that:
//!
//! * runs on **any** CPU of the target architecture — no illegal-instruction
//!   (SIGILL) crashes on pre-AVX2 chips, unlike a static `-C target-cpu=…`;
//! * still uses wide vector units where the hardware has them.
//!
//! The hot loops accumulate into **independent lanes** before a final
//! horizontal combine. That is what lets the optimizer auto-vectorize them:
//! a plain sequential `iter().sum()` is a dependency chain LLVM may not
//! reorder (doing so would change floating-point rounding). As a consequence
//! these results differ from a strict left-to-right sum by a few ULPs — well
//! inside every indicator's documented tolerance.

/// Number of independent accumulator lanes. Eight `f64` lanes cover the
/// widest target we dispatch to (AVX-512 = 8×f64); narrower targets (AVX2,
/// NEON) simply use a subset.
#[cfg(feature = "simd")]
const LANES: usize = 8;

/// Sum of a slice of `f64`, runtime-dispatched.
#[cfg(feature = "simd")]
#[multiversion::multiversion(targets = "simd")]
pub(crate) fn sum(data: &[f64]) -> f64 {
    let mut acc = [0.0f64; LANES];
    let mut chunks = data.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (a, &v) in acc.iter_mut().zip(chunk) {
            *a += v;
        }
    }
    let remainder: f64 = chunks.remainder().iter().sum();
    remainder + acc.iter().sum::<f64>()
}

/// Pure-scalar fallback when the `simd` feature is disabled.
#[cfg(not(feature = "simd"))]
pub(crate) fn sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Weighted-moving-average seed for the first window.
///
/// Returns `(t, s)` where `t = Σ data[k] * (k + 1)` (1-based linear weights)
/// and `s = Σ data[k]`. Used to seed the O(n) WMA recurrence.
#[cfg(feature = "simd")]
#[multiversion::multiversion(targets = "simd")]
pub(crate) fn wma_seed(data: &[f64]) -> (f64, f64) {
    wma_seed_impl(data)
}

/// Pure-scalar fallback when the `simd` feature is disabled.
#[cfg(not(feature = "simd"))]
pub(crate) fn wma_seed(data: &[f64]) -> (f64, f64) {
    wma_seed_impl(data)
}

/// Shared body for [`wma_seed`]. `#[inline(always)]` so that, inside a
/// multiversioned clone, it is inlined and recompiled with that clone's
/// target features.
#[inline(always)]
fn wma_seed_impl(data: &[f64]) -> (f64, f64) {
    let mut t = 0.0;
    let mut s = 0.0;
    for (k, &v) in data.iter().enumerate() {
        t += v * (k + 1) as f64;
        s += v;
    }
    (t, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strict sequential reference — the ground truth we compare against.
    fn naive_sum(data: &[f64]) -> f64 {
        data.iter().sum()
    }

    fn naive_wma_seed(data: &[f64]) -> (f64, f64) {
        let t = data
            .iter()
            .enumerate()
            .map(|(k, &v)| v * (k + 1) as f64)
            .sum();
        let s = data.iter().sum();
        (t, s)
    }

    /// Deterministic test vectors spanning the lane boundaries: empty, a
    /// partial chunk (< LANES), an exact multiple, and an exact-multiple +
    /// remainder. This exercises every branch of the chunked reduction.
    fn cases() -> Vec<Vec<f64>> {
        let big: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.5 - 123.0).collect();
        vec![
            vec![],
            vec![42.0],
            vec![1.0, 2.0, 3.0],                  // < LANES
            (1..=8).map(|i| i as f64).collect(),  // exactly LANES
            (1..=17).map(|i| i as f64).collect(), // LANES*2 + 1
            big,
        ]
    }

    #[test]
    fn sum_matches_sequential_within_tolerance() {
        for data in cases() {
            let got = sum(&data);
            let want = naive_sum(&data);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "sum mismatch: got {got}, want {want}, len {}",
                data.len()
            );
        }
    }

    #[test]
    fn wma_seed_matches_sequential_within_tolerance() {
        for data in cases() {
            let (t, s) = wma_seed(&data);
            let (wt, ws) = naive_wma_seed(&data);
            assert!(
                (t - wt).abs() <= 1e-9 * wt.abs().max(1.0),
                "wma t mismatch: got {t}, want {wt}, len {}",
                data.len()
            );
            assert!(
                (s - ws).abs() <= 1e-9 * ws.abs().max(1.0),
                "wma s mismatch: got {s}, want {ws}, len {}",
                data.len()
            );
        }
    }

    #[test]
    fn sum_empty_is_zero() {
        assert_eq!(sum(&[]), 0.0);
    }
}
