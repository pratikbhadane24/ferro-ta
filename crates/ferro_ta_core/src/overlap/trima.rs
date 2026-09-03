//! Triangular Moving Average.

use crate::rolling::RESEED_INTERVAL;

// ---------------------------------------------------------------------------
// TRIMA — Triangular Moving Average
// ---------------------------------------------------------------------------

/// Triangular Moving Average (triangle-weighted).
///
/// # Algorithm
///
/// The triangular weight vector is the convolution of two box filters, so the
/// weighted numerator is a *double* rolling sum and advances in O(1):
///
/// ```text
/// a = timeperiod / 2 + 1        b = timeperiod + 1 - a      (a + b = p + 1)
/// B(t) = Σ close[t - b + 1 ..= t]                (b-bar rolling sum)
/// N(i) = Σ_{m=0}^{a-1} B(i - m)  ==  Σ_j weight[j] * close[i - p + 1 + j]
/// N(i) = N(i - 1) + B(i) - B(i - a)
/// ```
///
/// `a * b` is exactly the triangular weight sum (`p = 5` → `a = b = 3`,
/// weights `1,2,3,2,1`, sum 9; `p = 4` → `a = 3, b = 2`, weights `1,2,2,1`,
/// sum 6) and `(b - 1) + (a - 1) = p - 1` reproduces the warmup, so the output
/// alignment is unchanged. `B(i)` is one running `b`-bar sum; the delayed
/// `B(i - a)` is read from an `a`-slot ring buffer whose slot for bar `i` is
/// the one bar `i` overwrites, so it costs a load and a store rather than a
/// second accumulator.
///
/// The numerator is recomputed exactly from the triangular weights every
/// 8192 bars, which bounds drift independently of `n`, and again
/// as soon as a non-finite input leaves the window. The latter preserves the
/// previous per-bar dot product's *localized* NaN behaviour: one `NaN` corrupts
/// exactly `timeperiod` outputs and results then resume, instead of a running
/// accumulator being poisoned forever.
pub fn trima(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 || n < timeperiod {
        return vec![f64::NAN; n];
    }
    let p = timeperiod;
    let half = p.div_ceil(2);
    let mut weights = Vec::with_capacity(p);
    for i in 1..=p {
        let w = if i <= half { i } else { p + 1 - i };
        weights.push(w as f64);
    }
    let weight_sum: f64 = weights.iter().sum();
    // Reciprocal-multiply rather than divide: a division is ~14 cycles of
    // latency against ~4 for a multiply, and at p = 20 that division was a
    // third of the whole per-bar cost. Costs at most one extra rounding;
    // TRIMA is gated at 1e-4.
    let inv_weight_sum = 1.0 / weight_sum;
    // Exact triangular dot product over one full window, oldest weight first —
    // the same operand order the previous per-bar loop used, so a seed or
    // reseed reproduces the old value for that bar bit-for-bit.
    let dot = |window: &[f64]| -> f64 {
        window
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| x * w)
            .sum()
    };

    let a = p / 2 + 1;
    let b = p + 1 - a;

    // `ring[t % a] == B(t)`. The `a` live entries are exactly the `b`-bar sums
    // the numerator still spans, and the slot read at bar `i` (`i % a`, which
    // holds `B(i - a)`) is the one bar `i` then overwrites — so the delayed
    // sum costs one load and one store, not a second accumulator.
    let mut ring = vec![0.0f64; a];
    let mut bsum = crate::simd::sum(&close[..b]);
    ring[(b - 1) % a] = bsum;
    for t in b..p {
        bsum += close[t] - close[t - b];
        ring[t % a] = bsum;
    }
    let mut num = dot(&close[..p]);

    let mut non_finite_inside = close[..p].iter().filter(|x| !x.is_finite()).count();
    let mut contaminated = non_finite_inside > 0;
    let mut since_reseed = 0usize;

    // Indexed stores over a pre-filled `NaN` buffer, not `push` (see `ema`).
    // The reseed below reads only from `close`, never from `result`, so
    // pre-sizing the output cannot disturb where it reads from.
    let mut result = vec![f64::NAN; n];
    result[p - 1] = num * inv_weight_sum;

    for i in p..n {
        bsum += close[i] - close[i - b];
        let slot = i % a;
        num += bsum - ring[slot];
        ring[slot] = bsum;
        since_reseed += 1;

        if !close[i].is_finite() {
            non_finite_inside += 1;
            contaminated = true;
        }
        if !close[i - p].is_finite() {
            non_finite_inside -= 1;
        }
        if since_reseed >= RESEED_INTERVAL || (contaminated && non_finite_inside == 0) {
            // Rebuild every accumulator exactly from the current window: the
            // `a` ring entries `B(i + 1 - a) ..= B(i)`, the running `b`-bar
            // sum, and the numerator.
            let base = i + 1 - a;
            let mut t_sum = crate::simd::sum(&close[i + 1 - p..=base]);
            ring[base % a] = t_sum;
            for t in base + 1..=i {
                t_sum += close[t] - close[t - b];
                ring[t % a] = t_sum;
            }
            bsum = t_sum;
            num = dot(&close[i + 1 - p..=i]);
            since_reseed = 0;
            contaminated = non_finite_inside > 0;
        }
        result[i] = num * inv_weight_sum;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap::test_support::*;

    fn reference_trima(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod == 0 || n < timeperiod {
            return result;
        }
        let half = timeperiod.div_ceil(2);
        let mut weights = Vec::with_capacity(timeperiod);
        for i in 1..=timeperiod {
            let w = if i <= half { i } else { timeperiod + 1 - i };
            weights.push(w as f64);
        }
        let weight_sum: f64 = weights.iter().sum();
        for i in (timeperiod - 1)..n {
            let mut val = 0.0_f64;
            for (j, &w) in weights.iter().enumerate() {
                val += close[i - (timeperiod - 1 - j)] * w;
            }
            result[i] = val / weight_sum;
        }
        result
    }

    // -- TRIMA -------------------------------------------------------------

    #[test]
    fn trima_matches_reference_dot_product() {
        let close = synthetic_series(4096);
        for &p in &[1usize, 2, 3, 4, 5, 6, 14, 30, 31, 200] {
            let got = trima(&close, p);
            let want = reference_trima(&close, p);
            assert_close(&got, &want, 1e-8, &format!("trima p={p}"));
        }
    }

    #[test]
    fn trima_degenerate_inputs() {
        assert!(trima(&[], 5).is_empty());
        assert!(trima(&[1.0, 2.0], 5).iter().all(|v| v.is_nan()));
        assert!(trima(&[1.0, 2.0], 0).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn trima_mid_series_nan_corrupts_exactly_timeperiod() {
        // The rewrite must keep the old *localized* NaN behaviour: a running
        // numerator without recovery would be poisoned for the rest of the
        // series.
        let p = 12usize;
        let mut close = synthetic_series(400);
        let bad = 200usize;
        close[bad] = f64::NAN;

        let got = trima(&close, p);
        let want = reference_trima(&close, p);
        assert_close(&got, &want, 1e-8, "trima nan");

        let corrupted: Vec<usize> = got
            .iter()
            .enumerate()
            .skip(p - 1)
            .filter(|(_, v)| v.is_nan())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            corrupted,
            (bad..bad + p).collect::<Vec<_>>(),
            "expected exactly {p} corrupted outputs starting at {bad}"
        );
        assert!(got[bad + p].is_finite(), "expected recovery at {}", bad + p);
    }

    #[test]
    fn trima_matches_reference_across_reseed_intervals() {
        // 20_000 bars crosses RESEED_INTERVAL (8192) twice, so both the
        // periodic exact recompute and the bar after it are exercised. The
        // reseed must not introduce a discontinuity.
        let p = 30usize;
        let close = synthetic_series(20_000);
        let got = trima(&close, p);
        let want = reference_trima(&close, p);
        assert!(20_000 - p > 2 * RESEED_INTERVAL, "series too short");
        assert_close(&got, &want, 1e-8, "trima long");

        // No step change at the reseed boundaries themselves.
        for boundary in [p - 1 + RESEED_INTERVAL, p - 1 + 2 * RESEED_INTERVAL] {
            let jump = (got[boundary] - got[boundary - 1]).abs();
            let reference_jump = (want[boundary] - want[boundary - 1]).abs();
            assert!(
                (jump - reference_jump).abs() < 1e-8,
                "reseed discontinuity at {boundary}: {jump} vs {reference_jump}"
            );
        }
    }
}
