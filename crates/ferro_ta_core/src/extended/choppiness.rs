//! The Choppiness Index, plus the true-range helper it shares with the
//! channel kernels' reference forms.

use crate::rolling::{MaxDeque, MinDeque, RollingSum};

// ---------------------------------------------------------------------------
// CHOPPINESS_INDEX
// ---------------------------------------------------------------------------

/// True Range at bar `i` (`ATR(1)`), with the bar-0 seed `high - low`.
#[inline]
fn true_range_at(high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
    if i == 0 {
        return high[0] - low[0];
    }
    let hl = high[i] - low[i];
    let hc = (high[i] - close[i - 1]).abs();
    let lc = (low[i] - close[i - 1]).abs();
    hl.max(hc).max(lc)
}

/// One choppiness reading from a windowed TR sum and the window's high-low
/// range. A `NaN` `hl_range` fails the `> 0.0` test, so a `NaN` input yields
/// `NaN` exactly as the array-based form did.
#[inline]
fn choppiness_value(sum_tr: f64, hl_range: f64, log_n: f64) -> f64 {
    if hl_range > 0.0 && log_n > 0.0 {
        100.0 * (sum_tr / hl_range).log10() / log_n
    } else {
        f64::NAN
    }
}

/// Choppiness Index — measures market choppiness vs trending.
///
/// Values near 100 indicate a choppy market; near 0 indicates trending.
/// The first `timeperiod` values are `NaN`. Mismatched input lengths yield
/// all `NaN`.
///
/// # Summation order
///
/// The windowed TR sum is a running accumulator ([`RollingSum`]), not the
/// difference of a whole-series prefix-sum array, so output is **not**
/// bit-identical to the previous implementation. It is the better-conditioned
/// of the two: TR is non-negative, so a windowed sum has no cancellation and
/// the accumulator's periodic exact reseed bounds its relative error at
/// `O(sqrt(RESEED_INTERVAL) * eps)` ~ 1e-14, whereas the prefix-sum difference
/// accumulated over the whole series *and* let one `NaN` poison every later
/// window. The gate is the default `(rtol 1e-4, atol 1e-5)`.
pub fn choppiness_index(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    if timeperiod < 1 || n <= timeperiod || low.len() != n || close.len() != n {
        return vec![f64::NAN; n];
    }

    let log_n = (timeperiod as f64).log10();
    let mut result = vec![f64::NAN; n];
    let mut hi = MaxDeque::with_window(timeperiod, n);
    let mut lo = MinDeque::with_window(timeperiod, n);

    // Ring holding the live TR window. `slot(i)` is where `tr[i]` lives and is
    // exactly the slot `tr[i - timeperiod]` vacates, so one store per bar
    // suffices and no full-length `tr` / `cum_tr` array is needed.
    let slot = |i: usize| (i - 1) % timeperiod;
    let mut tr_win = vec![0.0_f64; timeperiod];

    for i in 0..timeperiod {
        hi.advance(i, high[i], timeperiod);
        lo.advance(i, low[i], timeperiod);
        if i >= 1 {
            tr_win[slot(i)] = true_range_at(high, low, close, i);
        }
    }

    // Bar `timeperiod` closes the first full TR window (`tr[1..=timeperiod]`)
    // and is the first bar that emits a value.
    hi.advance(timeperiod, high[timeperiod], timeperiod);
    lo.advance(timeperiod, low[timeperiod], timeperiod);
    tr_win[slot(timeperiod)] = true_range_at(high, low, close, timeperiod);
    let mut sum_tr = RollingSum::new(&tr_win);
    result[timeperiod] = choppiness_value(sum_tr.value(), hi.front() - lo.front(), log_n);

    for i in (timeperiod + 1)..n {
        hi.advance(i, high[i], timeperiod);
        lo.advance(i, low[i], timeperiod);
        let tr = true_range_at(high, low, close, i);
        let s = slot(i);
        let leaving = tr_win[s];
        tr_win[s] = tr;
        // `tr_win` is the live window in ring order — all a reseed needs.
        sum_tr.advance(tr, leaving, &tr_win);
        result[i] = choppiness_value(sum_tr.value(), hi.front() - lo.front(), log_n);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extended::test_support::{assert_close, periods_for, sample_ohlcv, stress_cases};

    // -----------------------------------------------------------------------
    // CHOPPINESS_INDEX tests
    // -----------------------------------------------------------------------

    #[test]
    fn choppiness_index_basic() {
        let (h, l, c, _) = sample_ohlcv();
        let result = choppiness_index(&h, &l, &c, 3);
        assert_eq!(result.len(), h.len());
        // First 3 values should be NaN (timeperiod=3, i+1 > 3 starts at i=3)
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        // Index 3 should have a valid value (i+1=4 > 3)
        assert!(!result[3].is_nan());
        // CI should be between 0 and 100
        for val in result.iter().filter(|v| !v.is_nan()) {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn choppiness_index_empty_input() {
        let result = choppiness_index(&[], &[], &[], 3);
        assert!(result.is_empty());
    }

    fn reference_choppiness_index(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        timeperiod: usize,
    ) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n <= timeperiod {
            return result;
        }
        let mut tr = vec![0.0_f64; n];
        tr[0] = high[0] - low[0];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }
        let mut cum_tr = vec![0.0_f64; n];
        cum_tr[0] = tr[0];
        for i in 1..n {
            cum_tr[i] = cum_tr[i - 1] + tr[i];
        }
        let log_n = (timeperiod as f64).log10();
        let hh = crate::math::sliding_max(high, timeperiod);
        let ll = crate::math::sliding_min(low, timeperiod);
        for i in timeperiod..n {
            let sum_tr = cum_tr[i] - cum_tr[i - timeperiod];
            let hl_range = hh[i] - ll[i];
            if hl_range > 0.0 && log_n > 0.0 {
                result[i] = 100.0 * (sum_tr / hl_range).log10() / log_n;
            }
        }
        result
    }

    /// Not bit-identical: a windowed running sum replaces the whole-series
    /// prefix-sum difference, so the summation order changes. TR is
    /// non-negative (no cancellation) and the accumulator reseeds exactly
    /// every 8192 advances, so the deviation is ~1e-13 relative — the gate
    /// here is 1e-8, and the cross-library gate is `(1e-4, 1e-5)`.
    #[test]
    fn choppiness_index_matches_reference_within_tolerance() {
        for (name, h, l, c) in stress_cases() {
            for p in periods_for(h.len()) {
                let got = choppiness_index(&h, &l, &c, p);
                let want = reference_choppiness_index(&h, &l, &c, p);
                assert_close(&got, &want, 1e-8, &format!("choppiness({name}, {p})"));
            }
        }
    }

    /// Where the prefix-sum form let one `NaN` poison every later window, the
    /// running accumulator recovers as soon as the `NaN` leaves the window.
    #[test]
    fn choppiness_index_recovers_after_a_mid_series_nan() {
        let p = 5usize;
        let n = 40usize;
        let clean_high: Vec<f64> = (0..n).map(|i| 20.0 + (i % 6) as f64 + 1.0).collect();
        let clean_low: Vec<f64> = (0..n).map(|i| 20.0 + (i % 6) as f64 - 1.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 20.0 + (i % 6) as f64).collect();
        let mut high = clean_high.clone();
        let mut low = clean_low.clone();
        high[15] = f64::NAN;
        low[15] = f64::NAN;

        let out = choppiness_index(&high, &low, &close, p);
        // `tr[15]` and the high/low window both carry the NaN for exactly the
        // `p` bars 15..=19.
        for i in 15..(15 + p) {
            assert!(out[i].is_nan(), "out[{i}] should be NaN, got {}", out[i]);
        }
        for i in (15 + p)..n {
            assert!(
                out[i].is_finite(),
                "out[{i}] should have recovered, got {}",
                out[i]
            );
        }
        // Recovery is exact, not merely finite: from bar 15 + p on, neither
        // the TR window nor the high/low window contains bar 15, so the values
        // must equal those from the NaN-free series. (The reference cannot be
        // fed the NaN series here — its `cum_tr` prefix array stays `NaN` for
        // the whole remainder, which is the defect this rewrite removes.)
        let clean = reference_choppiness_index(&clean_high, &clean_low, &close, p);
        assert_close(
            &out[(15 + p)..],
            &clean[(15 + p)..],
            1e-8,
            "choppiness_tail",
        );
    }
}
