//! The Ultimate Oscillator.

// ---------------------------------------------------------------------------
// Ultimate Oscillator
// ---------------------------------------------------------------------------

/// The three Ultimate Oscillator lookbacks, in weight order.
const ULTOSC_PERIODS: usize = 3;

/// TA-Lib's `TA_ULTOSC` weights: 4 for the shortest period, 2 for the middle,
/// 1 for the longest.
const ULTOSC_WEIGHTS: [f64; ULTOSC_PERIODS] = [4.0, 2.0, 1.0];

/// Combine the three buying-pressure / true-range window sums into one bar.
///
/// Factored out so the seeded first bar and the advanced bars share one copy
/// of the arithmetic, in the same association order the previous closure used:
/// `100 * (4 * a1 + 2 * a2 + a3) / 7`.
#[inline]
fn ultosc_bar(
    bp_acc: &[crate::rolling::RollingSum; ULTOSC_PERIODS],
    tr_acc: &[crate::rolling::RollingSum; ULTOSC_PERIODS],
) -> f64 {
    let mut acc = 0.0_f64;
    for k in 0..ULTOSC_PERIODS {
        let sum_tr = tr_acc[k].value();
        let ratio = if sum_tr != 0.0 {
            bp_acc[k].value() / sum_tr
        } else {
            0.0
        };
        acc += ULTOSC_WEIGHTS[k] * ratio;
    }
    100.0 * acc / 7.0
}

/// Ultimate Oscillator: weighted average of buying pressure over three periods.
///
/// # Complexity
///
/// The O(n) `bp` / `tr` prologue is unchanged. The main loop used to re-sum
/// both series over all three windows on every bar — six slice sums, i.e.
/// `O(n * (p1 + p2 + p3))`. It now carries **six [`crate::rolling::RollingSum`]
/// accumulators**, seeded from the first full window at `max_period` and
/// advanced once per bar, making the whole kernel O(n).
///
/// # Numerics
///
/// Not bit-identical: the trailing add/subtract recurrence, and the
/// lane-parallel seed, both round differently from a fresh left-to-right sum.
/// The margin is large:
///
/// * `bp` and `tr` are **non-negative** by construction (`close - true_low`
///   and `true_high - true_low`, where `true_low <= close <= true_high`), so
///   the window sums accumulate monotonically with no cancellation. Relative
///   error stays `O(p * eps)`.
/// * `RollingSum::advance` recomputes each sum exactly every
///   `rolling::RESEED_INTERVAL` bars, so drift does not grow with `n`, and it
///   also restores the two-pass `NaN` semantics: a single non-finite input
///   corrupts exactly the windows containing it, then results resume.
/// * This is the same trailing add/subtract TA-Lib's own `TA_ULTOSC` uses, and
///   ULTOSC is gated at the loose default `(rtol 1e-4, atol 1e-5)`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod1` / `timeperiod2` / `timeperiod3` - The three lookbacks
///   (TA-Lib defaults 7 / 14 / 28), weighted 4 / 2 / 1.
pub fn ultosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> Vec<f64> {
    let n = high.len();
    if timeperiod1 == 0 || timeperiod2 == 0 || timeperiod3 == 0 || n < 2 {
        return vec![f64::NAN; n];
    }
    let periods = [timeperiod1, timeperiod2, timeperiod3];
    let max_period = timeperiod1.max(timeperiod2).max(timeperiod3);
    if n <= max_period {
        return vec![f64::NAN; n];
    }

    let mut bp = vec![0.0_f64; n];
    let mut tr = vec![0.0_f64; n];
    for i in 1..n {
        let true_low = low[i].min(close[i - 1]);
        let true_high = high[i].max(close[i - 1]);
        bp[i] = close[i] - true_low;
        tr[i] = true_high - true_low;
    }

    // Seed every accumulator from the window ending at `max_period`, the first
    // bar the old loop emitted. Every window start is `>= 1` there, so the
    // synthetic `bp[0] = tr[0] = 0.0` entries never take part.
    let mut bp_acc: [crate::rolling::RollingSum; ULTOSC_PERIODS] = std::array::from_fn(|k| {
        crate::rolling::RollingSum::new(&bp[max_period + 1 - periods[k]..=max_period])
    });
    let mut tr_acc: [crate::rolling::RollingSum; ULTOSC_PERIODS] = std::array::from_fn(|k| {
        crate::rolling::RollingSum::new(&tr[max_period + 1 - periods[k]..=max_period])
    });

    let mut result = vec![f64::NAN; n];
    result[max_period] = ultosc_bar(&bp_acc, &tr_acc);
    for i in (max_period + 1)..n {
        for k in 0..ULTOSC_PERIODS {
            let p = periods[k];
            let lo = i + 1 - p;
            bp_acc[k].advance(bp[i], bp[i - p], &bp[lo..=i]);
            tr_acc[k].advance(tr[i], tr[i - p], &tr[lo..=i]);
        }
        result[i] = ultosc_bar(&bp_acc, &tr_acc);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::test_support::ultosc_ohlc;

    // -- Ultimate Oscillator -------------------------------------------------

    /// The pre-rewrite `ultosc`, verbatim (six slice sums per bar).
    fn reference_ultosc(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        timeperiod1: usize,
        timeperiod2: usize,
        timeperiod3: usize,
    ) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod1 == 0 || timeperiod2 == 0 || timeperiod3 == 0 || n < 2 {
            return result;
        }
        let max_period = timeperiod1.max(timeperiod2).max(timeperiod3);
        if n <= max_period {
            return result;
        }
        let mut bp = vec![0.0_f64; n];
        let mut tr = vec![0.0_f64; n];
        for i in 1..n {
            let true_low = low[i].min(close[i - 1]);
            let true_high = high[i].max(close[i - 1]);
            bp[i] = close[i] - true_low;
            tr[i] = true_high - true_low;
        }
        for i in max_period..n {
            let avg = |period: usize| -> f64 {
                let sum_bp: f64 = bp[(i + 1 - period)..=i].iter().sum();
                let sum_tr: f64 = tr[(i + 1 - period)..=i].iter().sum();
                if sum_tr != 0.0 {
                    sum_bp / sum_tr
                } else {
                    0.0
                }
            };
            result[i] =
                100.0 * (4.0 * avg(timeperiod1) + 2.0 * avg(timeperiod2) + avg(timeperiod3)) / 7.0;
        }
        result
    }

    /// `bp` and `tr` are non-negative, so the trailing add/subtract sums have
    /// no cancellation and the deviation from a fresh sum stays `O(p * eps)`.
    /// ULTOSC is gated at the loose default `(rtol 1e-4, atol 1e-5)`; this is
    /// six orders tighter.
    const ULTOSC_RTOL: f64 = 1e-10;

    #[test]
    fn ultosc_matches_the_slice_sum_reference() {
        let cases = [(7, 14, 28), (1, 1, 1), (2, 3, 5), (28, 14, 7), (5, 5, 5)];
        for n in [0, 1, 2, 3, 30, 40, 137] {
            let (high, low, close) = ultosc_ohlc(n);
            for (p1, p2, p3) in cases {
                let want = reference_ultosc(&high, &low, &close, p1, p2, p3);
                let got = ultosc(&high, &low, &close, p1, p2, p3);
                assert_eq!(got.len(), want.len(), "n={n} {p1}/{p2}/{p3}: length");
                for i in 0..want.len() {
                    if want[i].is_nan() {
                        assert!(got[i].is_nan(), "n={n} {p1}/{p2}/{p3} [{i}]: expected NaN");
                        continue;
                    }
                    let tol = ULTOSC_RTOL * want[i].abs().max(1.0);
                    assert!(
                        (got[i] - want[i]).abs() <= tol,
                        "ultosc n={n} {p1}/{p2}/{p3} [{i}]: {} vs {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
        // Degenerate periods must still short-circuit to all-NaN.
        let (high, low, close) = ultosc_ohlc(50);
        for (p1, p2, p3) in [(0, 14, 28), (7, 0, 28), (7, 14, 0)] {
            assert!(ultosc(&high, &low, &close, p1, p2, p3)
                .iter()
                .all(|v| v.is_nan()));
        }
    }

    /// Long enough to cross `rolling::RESEED_INTERVAL` (8192) twice, which is
    /// the only way the periodic exact recompute is exercised at all. Drift
    /// must stay bounded rather than growing with `n`.
    #[test]
    fn ultosc_stays_accurate_across_two_reseed_intervals() {
        let n = 3 * 8192 + 500;
        let (high, low, close) = ultosc_ohlc(n);
        let want = reference_ultosc(&high, &low, &close, 7, 14, 28);
        let got = ultosc(&high, &low, &close, 7, 14, 28);
        let mut worst = 0.0_f64;
        for i in 0..n {
            if want[i].is_nan() {
                assert!(got[i].is_nan(), "[{i}] expected NaN");
                continue;
            }
            worst = worst.max((got[i] - want[i]).abs() / want[i].abs().max(1.0));
        }
        assert!(worst <= ULTOSC_RTOL, "worst relative deviation {worst}");
    }

    /// A single `NaN` must corrupt exactly the windows containing it, then
    /// results must resume — the localized two-pass semantics that
    /// `RollingSum`'s non-finite guard exists to preserve. Asserted against
    /// the reference's own NaN pattern, which is the definition.
    #[test]
    fn ultosc_nan_footprint_matches_the_reference() {
        let n = 200;
        let (mut high, low, mut close) = ultosc_ohlc(n);
        high[90] = f64::NAN;
        close[120] = f64::NAN;
        let want = reference_ultosc(&high, &low, &close, 7, 14, 28);
        let got = ultosc(&high, &low, &close, 7, 14, 28);
        for i in 0..n {
            assert_eq!(
                got[i].is_nan(),
                want[i].is_nan(),
                "[{i}]: got {} want {}",
                got[i],
                want[i]
            );
        }
        // And the tail, well clear of both NaNs, is back to full accuracy.
        // The later NaN is at bar 120 and feeds `bp`/`tr` at 120 and 121, so
        // the longest (28-bar) window is clean again from bar 149 on.
        for i in 149..n {
            assert!(got[i].is_finite(), "[{i}] not finite");
            assert!((got[i] - want[i]).abs() <= ULTOSC_RTOL * want[i].abs().max(1.0));
        }
    }
}
