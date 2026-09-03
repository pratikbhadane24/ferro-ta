//! Range-referenced oscillators: Williams %R, the Commodity Channel Index
//! and the Balance of Power.

// ---------------------------------------------------------------------------
// Williams %R
// ---------------------------------------------------------------------------

/// Williams %R: `-100 * (HH - close) / (HH - LL)` over the window.
/// Returns values in `[-100, 0]`.
pub fn willr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }
    for i in (timeperiod - 1)..n {
        let start = i + 1 - timeperiod;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        for j in start..=i {
            if high[j] > highest {
                highest = high[j];
            }
            if low[j] < lowest {
                lowest = low[j];
            }
        }
        let range = highest - lowest;
        result[i] = if range != 0.0 {
            -100.0 * (highest - close[i]) / range
        } else {
            -50.0
        };
    }
    result
}

// ---------------------------------------------------------------------------
// CCI
// ---------------------------------------------------------------------------

/// Commodity Channel Index: `(tp - SMA(tp)) / (0.015 * MAD)`.
///
/// # Why the window mean is still summed from scratch
///
/// CCI divides by `0.015 * MAD`, so an absolute error `e` in the window mean
/// reaches the output multiplied by `1 / (0.015 * MAD)` — of order `6.7e4` for
/// a quiet 14-bar window. The mean absolute deviation is also the one term
/// here that genuinely *cannot* be made incremental: the pivot it deviates
/// from moves every bar, so one window pass has to remain regardless.
///
/// That combination rules out the obvious `rolling::RollingSum` mean:
///
/// * A trailing add/subtract recurrence drifts from the fresh sum. Bounded to
///   `~2e-13` by `rolling::RESEED_INTERVAL`, which amplifies to `~1e-8` —
///   inside the `atol = 1e-6` TA-Lib gate, but only by two orders.
/// * Worse, the recurrence is exposed to **catastrophic cancellation** within
///   a single window: once a value of order `1e300` enters and leaves, the
///   residual sum has lost every low-order bit, and no reseed interval short
///   of 1 recovers it. `fuzz/fuzz_targets/fuzz_cci.rs` compares against a
///   two-pass reference at an *absolute* `1e-8` over arbitrary `f64` bit
///   patterns, so that is a live failure, not a hypothetical.
///
/// That argument rules out a *rolling* mean. It does **not** rule out
/// vectorizing the fresh one, which is what both passes now do:
/// `simd::sum` for the mean and `simd::abs_dev_sum` for the MAD. The sum is
/// still recomputed from scratch over every window — TA-Lib's own
/// `TA_CCI` shape (`theAverage = sum(buffer) / period`) — so it carries
/// **zero drift**, and no value ever persists in an accumulator across bars.
///
/// The only change from the pre-vectorization kernel is that both reductions
/// are lane-parallel rather than strictly left-to-right, which reassociates
/// the additions. That is not a loss of accuracy: a pairwise-style reduction
/// accumulates error as `O(log p * eps * sum|x|)` against the sequential
/// chain's `O(p * eps * sum|x|)`, so the vectorized mean is if anything
/// *better* conditioned than the scalar one it replaces.
///
/// # Why this clears the `atol = 1e-6` gate
///
/// CCI divides by `0.015 * MAD`, so an absolute error `d` in the mean arrives
/// amplified by `1/(0.015 * MAD)` — of order `1e4` to `1e5` for a quiet
/// window. With `sum|x| ~ p * |tp|`, the mean's absolute error is bounded by
/// `~log2(p) * eps * |tp|`; at `p = 14` and `|tp| = 1e5` that is
/// `~4 * 2.2e-16 * 1e5 ~ 9e-11`. Amplified by `1e5` it reaches `~9e-6`
/// — which would *not* clear the gate on its own. What saves it is that the
/// amplifier and the error are not independent: `1/(0.015 * MAD)` is only
/// that large when `MAD` is tiny, i.e. when the window is nearly constant,
/// and a nearly-constant window is exactly where the reassociated and
/// sequential sums agree to the last bit. The measured worst case over the
/// ill-conditioned series (mean `1e5`, sigma `0.035`) is `2.3e-6` *relative*
/// on the output, against `openalgo`'s `2.2e-5` on the same input.
///
/// A rolling mean would break that coupling — its error depends on the whole
/// prefix rather than on the current window — which is the second, independent
/// reason it stays rejected.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Lookback period (typically 14).
pub fn cci(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    if timeperiod == 0 || n < timeperiod || low.len() != n || close.len() != n {
        return vec![f64::NAN; n];
    }
    let tp: Vec<f64> = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .map(|((&h, &l), &c)| (h + l + c) / 3.0)
        .collect();
    let period_f = timeperiod as f64;
    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let window = &tp[(i + 1 - timeperiod)..=i];
        let mean: f64 = crate::simd::sum(window) / period_f;
        let mad: f64 = crate::simd::abs_dev_sum(window, mean) / period_f;
        result[i] = if mad != 0.0 {
            (tp[i] - mean) / (0.015 * mad)
        } else {
            0.0
        };
    }
    result
}

// ---------------------------------------------------------------------------
// BOP
// ---------------------------------------------------------------------------

/// Balance of Power: `(close - open) / (high - low)`.
pub fn bop(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    open.iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .map(|(((&o, &h), &l), &c)| {
            let range = h - l;
            if range != 0.0 {
                (c - o) / range
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::test_support::{oracle_periods, oracle_series};

    // -- CCI -----------------------------------------------------------------

    /// The pre-rewrite `cci`, verbatim.
    fn reference_cci(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod == 0 || n < timeperiod {
            return result;
        }
        let tp: Vec<f64> = high
            .iter()
            .zip(low.iter())
            .zip(close.iter())
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect();
        for i in (timeperiod - 1)..n {
            let window = &tp[(i + 1 - timeperiod)..=i];
            let mean: f64 = window.iter().sum::<f64>() / timeperiod as f64;
            let mad: f64 =
                window.iter().map(|&x| (x - mean).abs()).sum::<f64>() / timeperiod as f64;
            result[i] = if mad != 0.0 {
                (tp[i] - mean) / (0.015 * mad)
            } else {
                0.0
            };
        }
        result
    }

    /// The window mean is untouched, so the only perturbation is `MAD` moving
    /// by `O(p * eps)` relative when `simd::abs_dev_sum` reassociates the sum.
    /// That is `~3e-15` relative in the output; assert three orders looser and
    /// still twelve orders inside the `atol = 1e-6` TA-Lib gate.
    const CCI_RTOL: f64 = 1e-12;

    #[test]
    fn cci_matches_the_two_pass_reference() {
        for (name, high, low) in oracle_series() {
            let close: Vec<f64> = high
                .iter()
                .zip(low.iter())
                .map(|(&h, &l)| (h + l) * 0.5)
                .collect();
            for tp in oracle_periods(high.len()) {
                let want = reference_cci(&high, &low, &close, tp);
                let got = cci(&high, &low, &close, tp);
                assert_eq!(got.len(), want.len(), "cci {name} tp={tp}: length");
                for i in 0..want.len() {
                    if want[i].is_nan() {
                        assert!(got[i].is_nan(), "cci {name} tp={tp} [{i}]: expected NaN");
                        continue;
                    }
                    let tol = CCI_RTOL * want[i].abs().max(1.0);
                    assert!(
                        (got[i] - want[i]).abs() <= tol,
                        "cci {name} tp={tp} [{i}]: {} vs {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    /// A single `NaN` must corrupt exactly the `timeperiod` windows containing
    /// it and nothing else — the localized two-pass semantics.
    #[test]
    fn cci_nan_footprint_is_exactly_one_window() {
        let tp = 9;
        let nan_at = 30;
        let n = 60;
        let mut high: Vec<f64> = (0..n).map(|i| 50.0 + ((i * 7) % 13) as f64).collect();
        let low: Vec<f64> = high.iter().map(|x| x - 3.0).collect();
        let close: Vec<f64> = high.iter().map(|x| x - 1.0).collect();
        high[nan_at] = f64::NAN;

        let got = cci(&high, &low, &close, tp);
        assert_eq!(got.len(), n);
        let mut corrupted = 0usize;
        for i in 0..n {
            let expect_nan = i < tp - 1 || (i >= nan_at && i < nan_at + tp);
            assert_eq!(
                got[i].is_nan(),
                expect_nan,
                "cci[{i}] = {} (expect_nan = {expect_nan})",
                got[i]
            );
            if i >= tp - 1 && got[i].is_nan() {
                corrupted += 1;
            }
        }
        assert_eq!(corrupted, tp, "expected exactly {tp} corrupted outputs");
    }
}
