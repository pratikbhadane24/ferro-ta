//! Aroon and the Aroon Oscillator.

// ---------------------------------------------------------------------------
// Aroon
// ---------------------------------------------------------------------------

// Both Aroon kernels scan the whole `timeperiod + 1` window twice per bar
// (once for the argmax of `high`, once for the argmin of `low`), which is
// O(n * p).
//
// # Why not a monotonic deque
//
// This was tried, and it made `AROON` and `AROONOSC` 23% *slower* on the
// benchmark suite. `rolling::sliding_arg_extrema` replaces the two scans with
// one O(n) monotonic-deque traversal whose cost is flat in `p`, but its
// constant is high: every pop predicate is an unpredictable branch, and both
// deques do work on each bar whether or not the window extreme actually
// moved. The scan is instead a dense, branch-predictable sweep over `p + 1`
// contiguous `f64`s that stays in L1. Interleaved measurement, `n = 100_000`,
// LTO release builds, min of 20 reps per variant, `AROON` in microseconds:
//
// | period | scan | deque | scan speedup |
// |--------|------|-------|--------------|
// |      5 |  363 |  1264 |        3.48x |
// |     14 |  896 |  1391 |        1.55x |
// |     20 | 1346 |  1379 |        1.02x |
// |     30 | 1948 |  1366 |        0.70x |
// |    100 | 8500 |  1343 |        0.16x |
//
// So the deque only starts paying around `p = 20` — above the default of 14
// and above every period the conformance and benchmark suites exercise. A
// `median`-style period threshold would recover the large-`p` end, but it was
// not taken: it would leave the `NaN` handling below depending on `timeperiod`
// (the deque can surface a `NaN` as the extreme, the scan never does), which
// is a worse defect than a slow `p = 100`. `willr` in `momentum::range` scans
// for the same reason.
//
// # Tie-breaking
//
// The outputs are `100 * k / period` for an integer `k`, so there is no
// floating-point error to reason about — only the choice of `k`. The `>=` for
// the max and `<=` for the min are load-bearing: among equal extremes the
// *most recent* bar wins, and inverting that would shift the output by
// `100 / period` (about 7.1 at `p = 14`) against an `atol = 1e-3` conformance
// gate. Because those comparisons are both false against `NaN`, a `NaN` can
// never become the running extreme and is silently skipped, which is what
// TA-Lib's own `TA_AROON` scan does. The `#[cfg(test)]` `reference_aroon` copy
// of this loop asserts bit-identity on `to_bits()`.

/// Aroon indicator. Returns `(aroon_down, aroon_up)`.
///
/// # Arguments
/// * `high` / `low` - High and low price series (same length).
/// * `timeperiod` - Lookback period (typically 14).
pub fn aroon(high: &[f64], low: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut aroon_down = vec![f64::NAN; n];
    let mut aroon_up = vec![f64::NAN; n];
    if timeperiod == 0 || n <= timeperiod || low.len() != n {
        return (aroon_down, aroon_up);
    }
    let period_f = timeperiod as f64;
    let window_size = timeperiod + 1;
    for i in timeperiod..n {
        let start = i + 1 - window_size;
        let (max_idx, min_idx) = window_arg_extrema(&high[start..=i], &low[start..=i]);
        aroon_up[i] = 100.0 * (max_idx as f64) / period_f;
        aroon_down[i] = 100.0 * (min_idx as f64) / period_f;
    }
    (aroon_down, aroon_up)
}

/// Aroon Oscillator: `aroon_up - aroon_down`.
///
/// Computed in its own single pass rather than by subtracting two `aroon`
/// outputs, which needed a third full-length `Vec`.
///
/// # Arguments
/// * `high` / `low` - High and low price series (same length).
/// * `timeperiod` - Lookback period (typically 14).
pub fn aroonosc(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n <= timeperiod || low.len() != n {
        return result;
    }
    let period_f = timeperiod as f64;
    let window_size = timeperiod + 1;
    for i in timeperiod..n {
        let start = i + 1 - window_size;
        let (max_idx, min_idx) = window_arg_extrema(&high[start..=i], &low[start..=i]);
        // Kept as the difference of the two rounded quotients, not
        // `100 * (max_idx - min_idx) / p`: the two forms are not the same
        // `f64`, and the difference of quotients is what `aroon` produced.
        let up = 100.0 * (max_idx as f64) / period_f;
        let down = 100.0 * (min_idx as f64) / period_f;
        result[i] = up - down;
    }
    result
}

/// Offsets of the argmax of `high` and the argmin of `low` within one window.
///
/// `high` and `low` must be the same non-empty length. Ties resolve to the
/// most recent bar — see the note above; `NaN` fails both comparisons and is
/// therefore skipped.
#[inline]
fn window_arg_extrema(high: &[f64], low: &[f64]) -> (usize, usize) {
    let mut max_val = high[0];
    let mut min_val = low[0];
    let mut max_idx = 0usize;
    let mut min_idx = 0usize;
    for (j, (&h, &l)) in high.iter().zip(low.iter()).enumerate() {
        if h >= max_val {
            max_val = h;
            max_idx = j;
        }
        if l <= min_val {
            min_val = l;
            min_idx = j;
        }
    }
    (max_idx, min_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::test_support::{assert_bits, oracle_periods, oracle_series};

    // -- Aroon ---------------------------------------------------------------

    /// The pre-rewrite `aroon`, verbatim.
    fn reference_aroon(high: &[f64], low: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut aroon_down = vec![f64::NAN; n];
        let mut aroon_up = vec![f64::NAN; n];
        if timeperiod == 0 || n <= timeperiod {
            return (aroon_down, aroon_up);
        }
        let period_f = timeperiod as f64;
        let window_size = timeperiod + 1;
        for i in timeperiod..n {
            let start = i + 1 - window_size;
            let mut max_val = high[start];
            let mut min_val = low[start];
            let mut max_idx = 0usize;
            let mut min_idx = 0usize;
            for j in 0..window_size {
                if high[start + j] >= max_val {
                    max_val = high[start + j];
                    max_idx = j;
                }
                if low[start + j] <= min_val {
                    min_val = low[start + j];
                    min_idx = j;
                }
            }
            aroon_up[i] = 100.0 * (max_idx as f64) / period_f;
            aroon_down[i] = 100.0 * (min_idx as f64) / period_f;
        }
        (aroon_down, aroon_up)
    }

    #[test]
    fn aroon_is_bit_identical_to_the_naive_scan() {
        for (name, high, low) in oracle_series() {
            for tp in oracle_periods(high.len()) {
                let (want_down, want_up) = reference_aroon(&high, &low, tp);
                let (got_down, got_up) = aroon(&high, &low, tp);
                assert_bits(&got_up, &want_up, &format!("aroon_up {name} tp={tp}"));
                assert_bits(&got_down, &want_down, &format!("aroon_down {name} tp={tp}"));
            }
        }
    }

    /// Constant ties are the whole risk in the Aroon rewrite: the outputs are
    /// `100 * k / p` for integer `k`, so there is no float error to hide
    /// behind — an inverted tie-break shows up as a clean `100 / p` shift
    /// (7.14 at `p = 14`) against an `atol = 1e-3` gate.
    #[test]
    fn aroon_tie_break_keeps_the_most_recent_extreme() {
        let series: Vec<f64> = (0..120).map(|i| (i % 5) as f64).collect();
        for tp in [1, 2, 4, 5, 9, 14] {
            let (want_down, want_up) = reference_aroon(&series, &series, tp);
            let (got_down, got_up) = aroon(&series, &series, tp);
            assert_bits(&got_up, &want_up, &format!("tie aroon_up tp={tp}"));
            assert_bits(&got_down, &want_down, &format!("tie aroon_down tp={tp}"));
        }
        // A plateau of five equal highs at p = 4 must report the newest bar,
        // i.e. aroon_up = 100, not the oldest (0).
        let plateau = vec![3.0; 10];
        let (down, up) = aroon(&plateau, &plateau, 4);
        assert_eq!(up[9].to_bits(), 100.0_f64.to_bits());
        assert_eq!(down[9].to_bits(), 100.0_f64.to_bits());
    }

    #[test]
    fn aroonosc_is_bit_identical_to_the_naive_difference() {
        for (name, high, low) in oracle_series() {
            for tp in oracle_periods(high.len()) {
                let (down, up) = reference_aroon(&high, &low, tp);
                let want: Vec<f64> = up
                    .iter()
                    .zip(down.iter())
                    .map(|(&u, &d)| {
                        if u.is_nan() || d.is_nan() {
                            f64::NAN
                        } else {
                            u - d
                        }
                    })
                    .collect();
                let got = aroonosc(&high, &low, tp);
                assert_bits(&got, &want, &format!("aroonosc {name} tp={tp}"));
            }
        }
    }

    /// Aroon over a window containing `NaN` has no defined value. The scan's
    /// `>=` / `<=` are both false against `NaN`, so the `NaN` is skipped and
    /// the last finite extreme is kept — see the note above `aroon`. Because
    /// the kernel *is* that scan, it now agrees with `reference_aroon` bit for
    /// bit even inside the contaminated window; the assertion below is left
    /// scoped to the clean bars anyway, so this test keeps holding whatever a
    /// future rewrite chooses to do with `NaN`. What it pins down is that the
    /// kernel does not panic, keeps its warmup, stays inside `[0, 100]`, and
    /// is exact again once the `NaN` has left the window.
    #[test]
    fn aroon_recovers_after_a_mid_series_nan() {
        let tp = 6;
        let window = tp + 1;
        let nan_at = 40;
        let mut high: Vec<f64> = (0..80).map(|i| 100.0 + (i % 7) as f64).collect();
        let mut low: Vec<f64> = high.iter().map(|x| x - 2.0).collect();
        high[nan_at] = f64::NAN;
        low[nan_at] = f64::NAN;

        let (down, up) = aroon(&high, &low, tp);
        let (want_down, want_up) = reference_aroon(&high, &low, tp);
        assert_eq!(up.len(), high.len());
        for i in 0..tp {
            assert!(up[i].is_nan() && down[i].is_nan(), "warmup {i}");
        }
        for i in tp..high.len() {
            assert!((0.0..=100.0).contains(&up[i]), "up[{i}] = {}", up[i]);
            assert!((0.0..=100.0).contains(&down[i]), "down[{i}] = {}", down[i]);
            // Outside the `window_size` bars whose window holds the NaN the
            // two implementations must agree bit for bit again.
            let touches_nan = i >= nan_at && i < nan_at + window;
            if !touches_nan {
                assert_eq!(up[i].to_bits(), want_up[i].to_bits(), "up[{i}]");
                assert_eq!(down[i].to_bits(), want_down[i].to_bits(), "down[{i}]");
            }
        }
    }
}
