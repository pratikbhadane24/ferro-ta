//! ATR- and extreme-based channel kernels: Supertrend, Donchian channels,
//! Keltner channels and the Chandelier Exit.
//!
//! All four drive [`crate::rolling`]'s monotonic deques or TA-Lib's ATR
//! directly, writing the rolling extremes straight into their outputs.

use crate::overlap;
use crate::rolling;
use crate::volatility;

// ---------------------------------------------------------------------------
// SUPERTREND
// ---------------------------------------------------------------------------

/// ATR-based Supertrend indicator.
///
/// # Returns
/// `(supertrend_line, direction)` where direction values are:
/// * `1` = uptrend
/// * `-1` = downtrend
/// * `0` = warmup (first `timeperiod` bars)
pub fn supertrend(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<i8>) {
    let n = high.len();
    let mut supertrend_out = vec![f64::NAN; n];
    let mut direction = vec![0_i8; n];

    if timeperiod < 1 || n <= timeperiod || low.len() != n || close.len() != n {
        return (supertrend_out, direction);
    }

    let atr = volatility::atr(high, low, close, timeperiod);

    let mut upper_band = vec![f64::NAN; n];
    let mut lower_band = vec![f64::NAN; n];

    // TA-Lib ATR first value is at `timeperiod`. Initialize and emit there.
    let first_valid = timeperiod;
    if first_valid >= n || atr[first_valid].is_nan() {
        return (supertrend_out, direction);
    }

    {
        let hl2 = (high[first_valid] + low[first_valid]) / 2.0;
        upper_band[first_valid] = hl2 + multiplier * atr[first_valid];
        lower_band[first_valid] = hl2 - multiplier * atr[first_valid];
        direction[first_valid] = if close[first_valid] > upper_band[first_valid] {
            1
        } else {
            -1
        };
        supertrend_out[first_valid] = if direction[first_valid] == 1 {
            lower_band[first_valid]
        } else {
            upper_band[first_valid]
        };
    }

    for i in (first_valid + 1)..n {
        if atr[i].is_nan() {
            continue;
        }

        let hl2 = (high[i] + low[i]) / 2.0;
        let upper_basic = hl2 + multiplier * atr[i];
        let lower_basic = hl2 - multiplier * atr[i];

        lower_band[i] = if lower_basic > lower_band[i - 1] || close[i - 1] < lower_band[i - 1] {
            lower_basic
        } else {
            lower_band[i - 1]
        };

        upper_band[i] = if upper_basic < upper_band[i - 1] || close[i - 1] > upper_band[i - 1] {
            upper_basic
        } else {
            upper_band[i - 1]
        };

        let prev_dir = direction[i - 1];
        direction[i] = if prev_dir == 0 || prev_dir == -1 {
            if close[i] > upper_band[i] {
                1
            } else {
                -1
            }
        } else if close[i] < lower_band[i] {
            -1
        } else {
            1
        };
        supertrend_out[i] = if direction[i] == 1 {
            lower_band[i]
        } else {
            upper_band[i]
        };
    }

    (supertrend_out, direction)
}

// ---------------------------------------------------------------------------
// DONCHIAN
// ---------------------------------------------------------------------------

/// Donchian Channels — rolling highest high / lowest low.
///
/// # Returns
/// `(upper, middle, lower)` arrays. Mismatched input lengths yield all `NaN`.
pub fn donchian(high: &[f64], low: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];

    if timeperiod < 1 || n < timeperiod || low.len() != n {
        return (upper, middle, lower);
    }

    // One traversal writes both rolling extremes straight into the outputs.
    rolling::sliding_min_max_into(high, low, timeperiod, &mut upper, &mut lower);

    for i in (timeperiod - 1)..n {
        if upper[i].is_nan() {
            // A `NaN` surfacing as the window high suppressed the low too in
            // the two-pass form; keep that exactly.
            lower[i] = f64::NAN;
        } else {
            middle[i] = (upper[i] + lower[i]) / 2.0;
        }
    }

    (upper, middle, lower)
}

// ---------------------------------------------------------------------------
// KELTNER_CHANNELS
// ---------------------------------------------------------------------------

/// Keltner Channels — EMA +/- (multiplier x ATR).
///
/// # Returns
/// `(upper, middle, lower)` arrays.
pub fn keltner_channels(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
    atr_period: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    if timeperiod < 1
        || atr_period < 1
        || n < timeperiod
        || n < atr_period
        || low.len() != n
        || close.len() != n
    {
        let nan = vec![f64::NAN; n];
        return (nan.clone(), nan.clone(), nan);
    }

    let middle = overlap::ema(close, timeperiod);
    let atr = volatility::atr(high, low, close, atr_period);

    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in 0..n {
        if !middle[i].is_nan() && !atr[i].is_nan() {
            let band = multiplier * atr[i];
            upper[i] = middle[i] + band;
            lower[i] = middle[i] - band;
        }
    }

    (upper, middle, lower)
}

// ---------------------------------------------------------------------------
// CHANDELIER_EXIT
// ---------------------------------------------------------------------------

/// Chandelier Exit — ATR-based trailing stop levels.
///
/// # Returns
/// `(long_exit, short_exit)` arrays. Mismatched input lengths yield all `NaN`.
pub fn chandelier_exit(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    if timeperiod < 1 || n < timeperiod || low.len() != n || close.len() != n {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }

    let atr = volatility::atr(high, low, close, timeperiod);

    // The fused pass writes the rolling extremes into the outputs themselves,
    // which the arithmetic loop then converts in place.
    let mut long_exit = vec![f64::NAN; n];
    let mut short_exit = vec![f64::NAN; n];
    rolling::sliding_min_max_into(high, low, timeperiod, &mut long_exit, &mut short_exit);

    for i in (timeperiod - 1)..n {
        let highest_high = long_exit[i];
        let lowest_low = short_exit[i];
        if highest_high.is_nan() || atr[i].is_nan() {
            long_exit[i] = f64::NAN;
            short_exit[i] = f64::NAN;
        } else {
            long_exit[i] = highest_high - multiplier * atr[i];
            short_exit[i] = lowest_low + multiplier * atr[i];
        }
    }

    (long_exit, short_exit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extended::test_support::{assert_bit_eq, periods_for, sample_ohlcv, stress_cases};

    // -----------------------------------------------------------------------
    // SUPERTREND tests
    // -----------------------------------------------------------------------

    #[test]
    fn supertrend_basic() {
        let (h, l, c, _) = sample_ohlcv();
        let (st, dir) = supertrend(&h, &l, &c, 3, 2.0);
        assert_eq!(st.len(), h.len());
        assert_eq!(dir.len(), h.len());
        // First 3 bars should be warmup (direction = 0, st = NaN)
        for i in 0..3 {
            assert_eq!(dir[i], 0);
            assert!(st[i].is_nan());
        }
        // From bar 3 onward, direction should be 1 or -1
        for i in 3..h.len() {
            assert!(dir[i] == 1 || dir[i] == -1);
            assert!(!st[i].is_nan());
        }
    }

    #[test]
    fn supertrend_empty_input() {
        let (st, dir) = supertrend(&[], &[], &[], 3, 2.0);
        assert!(st.is_empty());
        assert!(dir.is_empty());
    }

    #[test]
    fn supertrend_insufficient_data() {
        let (st, dir) = supertrend(&[1.0, 2.0], &[0.5, 1.5], &[1.5, 1.8], 5, 2.0);
        assert!(st.iter().all(|v| v.is_nan()));
        assert!(dir.iter().all(|&d| d == 0));
    }

    // -----------------------------------------------------------------------
    // DONCHIAN tests
    // -----------------------------------------------------------------------

    #[test]
    fn donchian_basic() {
        let (h, l, _, _) = sample_ohlcv();
        let (upper, middle, lower) = donchian(&h, &l, 3);
        assert_eq!(upper.len(), h.len());
        // First 2 are NaN
        assert!(upper[0].is_nan());
        assert!(upper[1].is_nan());
        // Index 2: max(11,12,13)=13, min(9,10,11)=9
        assert!((upper[2] - 13.0).abs() < 1e-10);
        assert!((lower[2] - 9.0).abs() < 1e-10);
        assert!((middle[2] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn donchian_empty_input() {
        let (u, m, l) = donchian(&[], &[], 3);
        assert!(u.is_empty());
        assert!(m.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn donchian_period_1() {
        let h = vec![5.0, 3.0, 7.0];
        let l = vec![2.0, 1.0, 4.0];
        let (upper, middle, lower) = donchian(&h, &l, 1);
        // Every bar is its own window
        assert!((upper[0] - 5.0).abs() < 1e-10);
        assert!((lower[0] - 2.0).abs() < 1e-10);
        assert!((middle[0] - 3.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // KELTNER_CHANNELS tests
    // -----------------------------------------------------------------------

    #[test]
    fn keltner_channels_basic() {
        let (h, l, c, _) = sample_ohlcv();
        let (upper, middle, lower) = keltner_channels(&h, &l, &c, 3, 3, 1.5);
        assert_eq!(upper.len(), h.len());
        // Where both EMA and ATR are valid, upper > middle > lower
        for i in 0..h.len() {
            if !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(upper[i] > middle[i]);
                assert!(lower[i] < middle[i]);
            }
        }
    }

    #[test]
    fn keltner_channels_empty_input() {
        let (u, m, l) = keltner_channels(&[], &[], &[], 3, 3, 1.5);
        assert!(u.is_empty());
        assert!(m.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn keltner_uses_talib_atr() {
        let (h, l, c, _) = sample_ohlcv();
        let atr_period = 3;
        let (upper, middle, lower) = keltner_channels(&h, &l, &c, 3, atr_period, 1.5);
        let atr = crate::volatility::atr(&h, &l, &c, atr_period);
        // TA-Lib ATR first value is at `atr_period`, not `atr_period - 1`.
        assert!(atr[atr_period - 1].is_nan());
        assert!(upper[atr_period - 1].is_nan());
        assert!(!atr[atr_period].is_nan());
        assert!(!upper[atr_period].is_nan());
        for i in 0..h.len() {
            if !atr[i].is_nan() && !middle[i].is_nan() {
                assert!((upper[i] - (middle[i] + 1.5 * atr[i])).abs() < 1e-12);
                assert!((lower[i] - (middle[i] - 1.5 * atr[i])).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn chandelier_uses_talib_atr() {
        let (h, l, c, _) = sample_ohlcv();
        let period = 3;
        let (long_exit, short_exit) = chandelier_exit(&h, &l, &c, period, 2.0);
        let atr = crate::volatility::atr(&h, &l, &c, period);
        assert!(atr[period - 1].is_nan());
        assert!(long_exit[period - 1].is_nan());
        assert!(!long_exit[period].is_nan());
        assert!(!short_exit[period].is_nan());
        let hh = crate::math::sliding_max(&h, period);
        let ll = crate::math::sliding_min(&l, period);
        assert!((long_exit[period] - (hh[period] - 2.0 * atr[period])).abs() < 1e-12);
        assert!((short_exit[period] - (ll[period] + 2.0 * atr[period])).abs() < 1e-12);
    }

    #[test]
    fn supertrend_uses_talib_atr_and_starts_at_period() {
        // Wide first bar so bar-0 ATR seeding would shift the first Supertrend value.
        let h = vec![20.0, 12.0, 13.0, 14.0, 15.0, 14.5, 15.5, 16.0];
        let l = vec![5.0, 10.0, 11.0, 12.0, 13.0, 12.5, 13.5, 14.0];
        let c = vec![10.0, 11.0, 12.0, 13.0, 14.0, 13.5, 14.5, 15.0];
        let period = 3;
        let (st, dir) = supertrend(&h, &l, &c, period, 2.0);
        let atr = crate::volatility::atr(&h, &l, &c, period);
        for i in 0..period {
            assert!(st[i].is_nan());
            assert_eq!(dir[i], 0);
        }
        assert!(!atr[period].is_nan());
        assert!(!st[period].is_nan());
        assert!(dir[period] == 1 || dir[period] == -1);
        // First emitted line uses unadjusted bands from TA-Lib ATR at `period`.
        let hl2 = (h[period] + l[period]) / 2.0;
        let upper = hl2 + 2.0 * atr[period];
        let lower = hl2 - 2.0 * atr[period];
        let expected = if c[period] > upper { lower } else { upper };
        assert!(
            (st[period] - expected).abs() < 1e-12,
            "supertrend[{period}]={} expected {}",
            st[period],
            expected
        );
    }

    // -----------------------------------------------------------------------
    // CHANDELIER_EXIT tests
    // -----------------------------------------------------------------------

    #[test]
    fn chandelier_exit_basic() {
        let (h, l, c, _) = sample_ohlcv();
        let (long_exit, short_exit) = chandelier_exit(&h, &l, &c, 3, 2.0);
        assert_eq!(long_exit.len(), h.len());
        assert_eq!(short_exit.len(), h.len());
        // Where valid, long_exit should be below highest high
        for i in 0..h.len() {
            if !long_exit[i].is_nan() {
                // long_exit = highest_high - multiplier * atr, should be < max high
                assert!(long_exit[i] < 20.0); // sanity
            }
        }
    }

    #[test]
    fn chandelier_exit_empty_input() {
        let (le, se) = chandelier_exit(&[], &[], &[], 3, 2.0);
        assert!(le.is_empty());
        assert!(se.is_empty());
    }

    fn reference_donchian(
        high: &[f64],
        low: &[f64],
        timeperiod: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        let mut middle = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return (upper, middle, lower);
        }
        let hh = crate::math::sliding_max(high, timeperiod);
        let ll = crate::math::sliding_min(low, timeperiod);
        for i in 0..n {
            if !hh[i].is_nan() {
                upper[i] = hh[i];
                lower[i] = ll[i];
                middle[i] = (upper[i] + lower[i]) / 2.0;
            }
        }
        (upper, middle, lower)
    }

    fn reference_chandelier_exit(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        timeperiod: usize,
        multiplier: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        if timeperiod < 1 || n < timeperiod {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }
        let atr = crate::volatility::atr(high, low, close, timeperiod);
        let highest_high = crate::math::sliding_max(high, timeperiod);
        let lowest_low = crate::math::sliding_min(low, timeperiod);
        let mut long_exit = vec![f64::NAN; n];
        let mut short_exit = vec![f64::NAN; n];
        for i in 0..n {
            if !highest_high[i].is_nan() && !atr[i].is_nan() {
                long_exit[i] = highest_high[i] - multiplier * atr[i];
                short_exit[i] = lowest_low[i] + multiplier * atr[i];
            }
        }
        (long_exit, short_exit)
    }

    /// Bit-identical: the fused pass only reorders *where* the deque output is
    /// stored, never how it is computed.
    #[test]
    fn donchian_matches_reference_bitwise() {
        for (name, h, l, _) in stress_cases() {
            for p in periods_for(h.len()) {
                let (u, m, lo) = donchian(&h, &l, p);
                let (ru, rm, rlo) = reference_donchian(&h, &l, p);
                assert_bit_eq(&u, &ru, &format!("donchian({name}, {p}).upper"));
                assert_bit_eq(&m, &rm, &format!("donchian({name}, {p}).middle"));
                assert_bit_eq(&lo, &rlo, &format!("donchian({name}, {p}).lower"));
            }
        }
    }

    /// Bit-identical: ATR is untouched and the arithmetic is unchanged.
    #[test]
    fn chandelier_exit_matches_reference_bitwise() {
        for (name, h, l, c) in stress_cases() {
            for p in periods_for(h.len()) {
                let (le, se) = chandelier_exit(&h, &l, &c, p, 2.5);
                let (rle, rse) = reference_chandelier_exit(&h, &l, &c, p, 2.5);
                assert_bit_eq(&le, &rle, &format!("chandelier({name}, {p}).long"));
                assert_bit_eq(&se, &rse, &format!("chandelier({name}, {p}).short"));
            }
        }
    }
}
