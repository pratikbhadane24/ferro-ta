//! The Hull Moving Average and the full-length WMA recurrence it is built
//! on.

use crate::overlap;
use crate::rolling::RESEED_INTERVAL;

// ---------------------------------------------------------------------------
// HULL_MA
// ---------------------------------------------------------------------------

/// Hull Moving Average (HMA).
///
/// `HMA(n) = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))`
pub fn hull_ma(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 || n < timeperiod {
        return vec![f64::NAN; n];
    }

    let half = (timeperiod / 2).max(1);
    let sqrt_p = ((timeperiod as f64).sqrt() as usize).max(1);

    let wma_full = overlap::wma(close, timeperiod);

    // raw = 2 * wma_half - wma_full, folded **in place** into the half-period
    // buffer rather than allocating a third full-length vector.
    let mut raw = overlap::wma(close, half);
    for i in 0..n {
        raw[i] = if wma_full[i].is_nan() || raw[i].is_nan() {
            f64::NAN
        } else {
            2.0 * raw[i] - wma_full[i]
        };
    }

    // Final WMA on the full-length raw series (windows aligned to original
    // indices). `overlap::wma` is not NaN-aware, so windows that still contain
    // leading raw NaNs stay NaN.
    wma_full_length(&raw, sqrt_p)
}

/// WMA with windows aligned to the original series. A window that contains
/// any NaN yields NaN (so leading raw NaNs do not get compacted away).
///
/// O(n): the weighted sum `W` and the plain sum `S` advance by
/// `W += period * s[i] - S; S += s[i] - s[i - period]`, replacing the old
/// per-window dot product *and* its per-window `iter().any(is_nan)` scan —
/// `last_non_finite` makes "this window is contaminated" a single comparison.
/// A contaminated window is evaluated fresh (`inf - inf` would poison the
/// recurrence permanently, and a window holding a `NaN` must still yield
/// `NaN`); the recurrence re-seeds on the next clean window and every
/// [`RESEED_INTERVAL`] bars to bound drift.
///
/// Not bit-identical to the per-window dot product: expect ~1e-14 relative.
fn wma_full_length(series: &[f64], period: usize) -> Vec<f64> {
    let n = series.len();
    if period < 1 || n < period {
        return vec![f64::NAN; n];
    }
    let denom = (period * (period + 1) / 2) as f64;
    let p = period as f64;

    let mut out = vec![f64::NAN; n];

    let mut last_non_finite = series[..period - 1].iter().rposition(|v| !v.is_finite());
    let mut weighted = 0.0_f64;
    let mut plain = 0.0_f64;
    let mut seeded = false;
    let mut since_reseed = 0_usize;

    for i in (period - 1)..n {
        let start = i + 1 - period;
        if !series[i].is_finite() {
            last_non_finite = Some(i);
        }
        if last_non_finite.is_some_and(|j| j >= start) {
            seeded = false;
            out[i] = crate::simd::wma_seed(&series[start..=i]).0 / denom;
            continue;
        }
        if seeded && since_reseed < RESEED_INTERVAL {
            // `start >= 1` here: `seeded` can only be true from bar `period`
            // onwards, so `start - 1` never underflows.
            weighted += p * series[i] - plain;
            plain += series[i] - series[start - 1];
            since_reseed += 1;
        } else {
            (weighted, plain) = crate::simd::wma_seed(&series[start..=i]);
            seeded = true;
            since_reseed = 0;
        }
        out[i] = weighted / denom;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extended::test_support::{assert_close, periods_for, stress_cases};

    // -----------------------------------------------------------------------
    // HULL_MA tests
    // -----------------------------------------------------------------------

    #[test]
    fn hull_ma_basic() {
        let prices: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let result = hull_ma(&prices, 4);
        assert_eq!(result.len(), prices.len());
        // Should have some NaN warmup, then valid values
        let valid_count = result.iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn hull_ma_uses_floor_sqrt_and_full_length_wma() {
        // period=8 → floor(sqrt(8))=2, round(sqrt(8))=3. Distinguishes the old
        // `.round()` period and a WMA run on a truncated raw sub-slice.
        let close: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let result = hull_ma(&close, 8);

        // raw first finite at index 7; floor-sqrt WMA period 2 → first HMA at 8.
        // (round(sqrt(8))=3 would leave HMA[8] NaN.)
        for v in result.iter().take(8) {
            assert!(v.is_nan(), "expected NaN warmup, got {v}");
        }
        // WMA8[7]=204/36, WMA4[7]=7 → raw[7]=25/3
        // WMA8[8]=240/36, WMA4[8]=8 → raw[8]=28/3
        // WMA2[8]=(25/3 + 2*28/3)/3 = 9
        assert!(
            (result[8] - 9.0).abs() < 1e-12,
            "HMA[8] should be 9.0 with floor(sqrt(8))=2, got {}",
            result[8]
        );
        // raw[9]=31/3 → WMA2(28/3, 31/3) = 10
        assert!((result[9] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn hull_ma_empty_input() {
        let result = hull_ma(&[], 4);
        assert!(result.is_empty());
    }

    #[test]
    fn hull_ma_period_larger_than_data() {
        let result = hull_ma(&[1.0, 2.0], 10);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    fn reference_wma_full_length(series: &[f64], period: usize) -> Vec<f64> {
        let n = series.len();
        let mut out = vec![f64::NAN; n];
        if period < 1 || n < period {
            return out;
        }
        let denom = (period * (period + 1) / 2) as f64;
        for i in (period - 1)..n {
            let start = i + 1 - period;
            let window = &series[start..=i];
            if window.iter().any(|v| v.is_nan()) {
                continue;
            }
            let mut weighted = 0.0;
            for (k, &v) in window.iter().enumerate() {
                weighted += (k + 1) as f64 * v;
            }
            out[i] = weighted / denom;
        }
        out
    }

    fn reference_hull_ma(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        if timeperiod < 1 || n < timeperiod {
            return vec![f64::NAN; n];
        }
        let half = (timeperiod / 2).max(1);
        let sqrt_p = ((timeperiod as f64).sqrt() as usize).max(1);
        let wma_full = overlap::wma(close, timeperiod);
        let wma_half = overlap::wma(close, half);
        let mut raw = vec![f64::NAN; n];
        for i in 0..n {
            if !wma_full[i].is_nan() && !wma_half[i].is_nan() {
                raw[i] = 2.0 * wma_half[i] - wma_full[i];
            }
        }
        reference_wma_full_length(&raw, sqrt_p)
    }

    /// Not bit-identical: `wma_full_length` is now the O(n) WMA recurrence
    /// instead of a fresh per-window dot product, and the seed comes from the
    /// SIMD reduction. Expect ~1e-14 relative; the gate here is 1e-9.
    #[test]
    fn hull_ma_matches_reference_within_tolerance() {
        for (name, _, _, c) in stress_cases() {
            for p in periods_for(c.len()) {
                let got = hull_ma(&c, p);
                let want = reference_hull_ma(&c, p);
                assert_close(&got, &want, 1e-9, &format!("hull_ma({name}, {p})"));
            }
        }
    }

    /// `wma_full_length` must not let an infinity poison the recurrence: a
    /// window holding `inf` is evaluated fresh, and the next clean window
    /// re-seeds.
    #[test]
    fn wma_full_length_survives_infinities() {
        let mut series: Vec<f64> = (0..30).map(|i| 1.0 + i as f64).collect();
        series[12] = f64::INFINITY;
        let got = wma_full_length(&series, 4);
        let want = reference_wma_full_length(&series, 4);
        assert_close(&got, &want, 1e-12, "wma_full_length_inf");
        for i in 16..30 {
            assert!(got[i].is_finite(), "got[{i}] = {} was poisoned", got[i]);
        }
    }
}
