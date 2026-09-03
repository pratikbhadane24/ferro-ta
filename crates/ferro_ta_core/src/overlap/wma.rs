//! Weighted Moving Average.

/// Compute the Weighted Moving Average (WMA).
///
/// Assigns linearly increasing weights (1, 2, ..., timeperiod) to the window.
/// Uses an O(n) incremental recurrence to avoid recomputing weights each bar.
/// Returns a `Vec<f64>` of length `n`; the first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Rolling window size (must be >= 1).
pub fn wma(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 || n < timeperiod {
        return vec![f64::NAN; n];
    }
    let denom: f64 = (timeperiod * (timeperiod + 1) / 2) as f64;
    // `denom` is loop-invariant, but LLVM cannot turn `t / denom` into
    // `t * (1 / denom)` without fast-math: a division is ~14 cycles of latency
    // against ~4 for the multiply. Reciprocal-then-multiply costs at most one
    // extra rounding (<= 1 ulp of the quotient); WMA is gated at 1e-4.
    let inv_denom = 1.0 / denom;
    let p = timeperiod as f64;

    // Seed: compute T and S for the first window via a runtime-dispatched
    // reduction (the streaming recurrence below is sequential).
    let (mut t, mut s) = crate::simd::wma_seed(&close[..timeperiod]);

    // `vec![NaN; n]` + indexed stores rather than `with_capacity` + `push`:
    // `push` adds a per-bar capacity check and vec-header reload, and the two
    // forms measured within noise here (<= 7%, inside the layout variance
    // this machine shows for *identical* code), so this keeps the crate-wide
    // indexed shape. The warm-up prefix keeps its initialized `NaN`s;
    // `n >= timeperiod` is guaranteed above.
    let mut result = vec![f64::NAN; n];
    result[timeperiod - 1] = t * inv_denom;

    let mut i = timeperiod;
    while i + 1 < n {
        t += p * close[i] - s;
        s += close[i] - close[i - timeperiod];
        result[i] = t * inv_denom;

        t += p * close[i + 1] - s;
        s += close[i + 1] - close[i + 1 - timeperiod];
        result[i + 1] = t * inv_denom;

        i += 2;
    }
    if i < n {
        t += p * close[i] - s;
        result[i] = t * inv_denom;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap::test_support::*;

    fn reference_wma(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return result;
        }
        let denom: f64 = (timeperiod * (timeperiod + 1) / 2) as f64;
        // Fully naive weighted dot product — independent of the recurrence.
        for i in (timeperiod - 1)..n {
            let mut acc = 0.0f64;
            for j in 0..timeperiod {
                acc += close[i + 1 - timeperiod + j] * (j + 1) as f64;
            }
            result[i] = acc / denom;
        }
        result
    }

    // -- WMA ---------------------------------------------------------------

    #[test]
    fn wma_matches_naive_weighted_sum() {
        // `t * (1 / denom)` instead of `t / denom` costs at most one extra
        // rounding; the recurrence itself is unchanged. Gate is 1e-4.
        let close = synthetic_series(4096);
        for &p in &[1usize, 2, 5, 14, 30, 200] {
            let got = wma(&close, p);
            let want = reference_wma(&close, p);
            assert_close(&got, &want, 1e-9, &format!("wma p={p}"));
        }
    }

    #[test]
    fn wma_basic() {
        let prices = vec![1.0, 2.0, 3.0];
        let result = wma(&prices, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // weights: 1, 2, 3; denom 6 => (1*1 + 2*2 + 3*3)/6 = 14/6
        assert!((result[2] - 14.0 / 6.0).abs() < 1e-10);
    }
}
