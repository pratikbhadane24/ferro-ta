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
    let (chunks, remainder) = data.as_chunks::<LANES>();
    for chunk in chunks {
        for (a, &v) in acc.iter_mut().zip(chunk) {
            *a += v;
        }
    }
    remainder.iter().sum::<f64>() + acc.iter().sum::<f64>()
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
    // Lane-local accumulation (same idea as `sum`) so each CPU-feature clone
    // can vectorize: `t` weights each value by its 1-based global index.
    let mut t_acc = [0.0f64; LANES];
    let mut s_acc = [0.0f64; LANES];
    let (chunks, rem) = data.as_chunks::<LANES>();
    let mut base = 0.0f64; // global index of this chunk's first element
    for chunk in chunks {
        for (lane, ((t, s), &v)) in t_acc
            .iter_mut()
            .zip(s_acc.iter_mut())
            .zip(chunk)
            .enumerate()
        {
            *t += v * (base + lane as f64 + 1.0);
            *s += v;
        }
        base += LANES as f64;
    }
    let mut t = 0.0;
    let mut s = 0.0;
    for (i, &v) in rem.iter().enumerate() {
        t += v * (base + i as f64 + 1.0);
        s += v;
    }
    (t + t_acc.iter().sum::<f64>(), s + s_acc.iter().sum::<f64>())
}

/// Pure-scalar fallback when the `simd` feature is disabled.
#[cfg(not(feature = "simd"))]
pub(crate) fn wma_seed(data: &[f64]) -> (f64, f64) {
    let mut t = 0.0;
    let mut s = 0.0;
    for (k, &v) in data.iter().enumerate() {
        t += v * (k + 1) as f64;
        s += v;
    }
    (t, s)
}

/// Sum of absolute deviations from `mean`: `Σ |x - mean|`.
///
/// This is the mean-absolute-deviation numerator CCI needs. Lane-local so the
/// per-CPU clones can vectorize the `abs` and the accumulate together.
#[cfg(feature = "simd")]
#[allow(dead_code)] // consumed by the CCI kernel rewrite that follows this commit
#[multiversion::multiversion(targets = "simd")]
pub(crate) fn abs_dev_sum(window: &[f64], mean: f64) -> f64 {
    let mut acc = [0.0f64; LANES];
    let (chunks, remainder) = window.as_chunks::<LANES>();
    for chunk in chunks {
        for (a, &v) in acc.iter_mut().zip(chunk) {
            *a += (v - mean).abs();
        }
    }
    remainder.iter().map(|&v| (v - mean).abs()).sum::<f64>() + acc.iter().sum::<f64>()
}

/// Pure-scalar fallback when the `simd` feature is disabled.
#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
pub(crate) fn abs_dev_sum(window: &[f64], mean: f64) -> f64 {
    window.iter().map(|&v| (v - mean).abs()).sum()
}

/// Sum of absolute first differences: `Σ |x[k + 1] - x[k]|`.
///
/// This is KAMA's volatility term, used both to seed the efficiency ratio and
/// to recompute it exactly on reseed. Returns `0.0` for fewer than two values.
#[cfg(feature = "simd")]
#[allow(dead_code)] // consumed by the KAMA kernel rewrite that follows this commit
#[multiversion::multiversion(targets = "simd")]
pub(crate) fn abs_diff_sum(window: &[f64]) -> f64 {
    if window.len() < 2 {
        return 0.0;
    }
    // Two views offset by one bar. They have equal length, so `as_chunks`
    // splits them identically and lane `l` always pairs `x[k]` with `x[k + 1]`.
    let lhs = &window[..window.len() - 1];
    let rhs = &window[1..];
    let mut acc = [0.0f64; LANES];
    let (lhs_chunks, lhs_rem) = lhs.as_chunks::<LANES>();
    let (rhs_chunks, rhs_rem) = rhs.as_chunks::<LANES>();
    for (lc, rc) in lhs_chunks.iter().zip(rhs_chunks) {
        for (a, (&x, &y)) in acc.iter_mut().zip(lc.iter().zip(rc)) {
            *a += (y - x).abs();
        }
    }
    let tail: f64 = lhs_rem
        .iter()
        .zip(rhs_rem)
        .map(|(&x, &y)| (y - x).abs())
        .sum();
    tail + acc.iter().sum::<f64>()
}

/// Pure-scalar fallback when the `simd` feature is disabled.
#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
pub(crate) fn abs_diff_sum(window: &[f64]) -> f64 {
    window.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}

/// Count of values in `window` that are `<= pivot`.
///
/// This is the percent-rank inner loop. The comparison is a plain IEEE-754
/// `<=`, so it is **`false` whenever either operand is `NaN`** — a `NaN`
/// pivot yields `0`, and a `NaN` in the window never counts. `-0.0` and
/// `+0.0` compare equal, and `±inf` compare normally.
///
/// The lane accumulators are `u64` rather than `f64`: a compare-and-add on
/// integer lanes is a mask-and-subtract on every target we dispatch to, and
/// integer counts are exact, so unlike the floating-point reductions above
/// this one is bit-identical to a sequential count regardless of lane width.
#[cfg(feature = "simd")]
#[multiversion::multiversion(targets = "simd")]
pub(crate) fn count_le(window: &[f64], pivot: f64) -> usize {
    let mut acc = [0u64; LANES];
    let (chunks, remainder) = window.as_chunks::<LANES>();
    for chunk in chunks {
        for (a, &v) in acc.iter_mut().zip(chunk) {
            *a += u64::from(v <= pivot);
        }
    }
    let tail = remainder.iter().filter(|&&v| v <= pivot).count() as u64;
    (acc.iter().sum::<u64>() + tail) as usize
}

/// Pure-scalar fallback when the `simd` feature is disabled.
#[cfg(not(feature = "simd"))]
pub(crate) fn count_le(window: &[f64], pivot: f64) -> usize {
    window.iter().filter(|&&v| v <= pivot).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strict sequential reference — the ground truth we compare against.
    fn naive_sum(data: &[f64]) -> f64 {
        data.iter().sum()
    }

    fn naive_abs_dev_sum(data: &[f64], mean: f64) -> f64 {
        data.iter().map(|&v| (v - mean).abs()).sum()
    }

    fn naive_abs_diff_sum(data: &[f64]) -> f64 {
        data.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
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
    fn abs_dev_sum_matches_sequential_within_tolerance() {
        for data in cases() {
            let mean = if data.is_empty() {
                0.0
            } else {
                naive_sum(&data) / data.len() as f64
            };
            let got = abs_dev_sum(&data, mean);
            let want = naive_abs_dev_sum(&data, mean);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "abs_dev_sum mismatch: got {got}, want {want}, len {}",
                data.len()
            );
        }
    }

    #[test]
    fn abs_diff_sum_matches_sequential_within_tolerance() {
        for data in cases() {
            let got = abs_diff_sum(&data);
            let want = naive_abs_diff_sum(&data);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "abs_diff_sum mismatch: got {got}, want {want}, len {}",
                data.len()
            );
        }
    }

    /// Strict sequential count — the ground truth for `count_le`.
    fn naive_count_le(window: &[f64], pivot: f64) -> usize {
        let mut count = 0;
        for &v in window {
            if v <= pivot {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn count_le_matches_sequential_exactly() {
        let pivots = [
            -1000.0,
            -0.0,
            0.0,
            1.0,
            123.5,
            1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for data in cases() {
            for &pivot in &pivots {
                assert_eq!(
                    count_le(&data, pivot),
                    naive_count_le(&data, pivot),
                    "count_le mismatch: len {}, pivot {pivot}",
                    data.len()
                );
            }
        }
    }

    #[test]
    fn count_le_counts_every_tie() {
        // All-equal window: `<=` must count all of them, at every length
        // around the lane boundary.
        for len in 0..20 {
            let flat = vec![4.0; len];
            assert_eq!(count_le(&flat, 4.0), len);
            assert_eq!(count_le(&flat, 3.999), 0);
        }
    }

    #[test]
    fn count_le_nan_and_infinity_semantics() {
        // A NaN pivot compares false against everything.
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert_eq!(count_le(&data, f64::NAN), 0);

        // A NaN in the window never counts, whatever the pivot.
        let mut holed = data.clone();
        holed[3] = f64::NAN;
        holed[17] = f64::NAN;
        assert_eq!(count_le(&holed, f64::INFINITY), 18);
        assert_eq!(naive_count_le(&holed, f64::INFINITY), 18);

        // ±inf compare normally.
        let edges = [
            f64::NEG_INFINITY,
            -1.0,
            0.0,
            1.0,
            f64::INFINITY,
            f64::NAN,
            f64::NEG_INFINITY,
            2.0,
            3.0,
        ];
        assert_eq!(count_le(&edges, f64::INFINITY), 8);
        assert_eq!(count_le(&edges, f64::NEG_INFINITY), 2);
        assert_eq!(count_le(&edges, 0.0), 4);

        // -0.0 and +0.0 compare equal in both directions.
        let zeros = [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0];
        assert_eq!(count_le(&zeros, 0.0), 9);
        assert_eq!(count_le(&zeros, -0.0), 9);
    }

    #[test]
    fn sum_empty_is_zero() {
        assert_eq!(sum(&[]), 0.0);
        assert_eq!(abs_dev_sum(&[], 0.0), 0.0);
        assert_eq!(abs_diff_sum(&[]), 0.0);
        assert_eq!(abs_diff_sum(&[1.0]), 0.0);
    }
}
