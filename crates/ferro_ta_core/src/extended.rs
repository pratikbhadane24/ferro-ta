//! Extended indicators — pure Rust implementations (no PyO3, no numpy).
//!
//! These indicators are not part of TA-Lib and provide additional technical
//! analysis capabilities. All functions operate on `&[f64]` slices and return
//! `Vec<f64>` (or tuples thereof).

#![allow(clippy::too_many_arguments)]

use crate::math;
use crate::overlap;
// Note: we use a local compute_atr helper (seeds from bar 0) rather than
// crate::volatility::atr (which seeds from bar 1, TA-Lib style).

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute ATR array using Wilder smoothing (same algorithm as in the PyO3
/// extended module — seeds from bar 0, not bar 1 like TA-Lib's `volatility::atr`).
fn compute_atr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if n <= timeperiod {
        return result;
    }
    // Seed: SMA of first `timeperiod` true range values
    let mut seed_sum = high[0] - low[0]; // first TR has no prev_close
    for i in 1..timeperiod {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        seed_sum += hl.max(hc).max(lc);
    }
    let mut atr = seed_sum / timeperiod as f64;
    result[timeperiod - 1] = atr;
    let pf = (timeperiod - 1) as f64;
    for i in timeperiod..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        let tr = hl.max(hc).max(lc);
        atr = (atr * pf + tr) / timeperiod as f64;
        result[i] = atr;
    }
    result
}

// ---------------------------------------------------------------------------
// VWAP
// ---------------------------------------------------------------------------

/// Volume Weighted Average Price (cumulative or rolling).
///
/// # Arguments
/// * `high`, `low`, `close`, `volume` — equal-length price/volume slices.
/// * `timeperiod` — 0 for cumulative VWAP from bar 0; >= 1 for a rolling window.
///
/// # Returns
/// A `Vec<f64>` of VWAP values. For rolling mode the first `timeperiod - 1`
/// entries are `NaN`.
pub fn vwap(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timeperiod: usize,
) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];

    if timeperiod == 0 {
        let mut cum_tpv = 0.0_f64;
        let mut cum_vol = 0.0_f64;
        for i in 0..n {
            let tp = (high[i] + low[i] + close[i]) / 3.0;
            cum_tpv += tp * volume[i];
            cum_vol += volume[i];
            result[i] = if cum_vol != 0.0 {
                cum_tpv / cum_vol
            } else {
                f64::NAN
            };
        }
    } else {
        // Pre-compute cumulative sums for O(n) rolling window
        let mut cum_tpv_arr = vec![0.0_f64; n];
        let mut cum_vol_arr = vec![0.0_f64; n];
        for i in 0..n {
            let tp = (high[i] + low[i] + close[i]) / 3.0;
            let tpv = tp * volume[i];
            cum_tpv_arr[i] = tpv + if i > 0 { cum_tpv_arr[i - 1] } else { 0.0 };
            cum_vol_arr[i] = volume[i] + if i > 0 { cum_vol_arr[i - 1] } else { 0.0 };
        }
        for i in (timeperiod - 1)..n {
            let prev_tpv = if i >= timeperiod {
                cum_tpv_arr[i - timeperiod]
            } else {
                0.0
            };
            let prev_vol = if i >= timeperiod {
                cum_vol_arr[i - timeperiod]
            } else {
                0.0
            };
            let w_tpv = cum_tpv_arr[i] - prev_tpv;
            let w_vol = cum_vol_arr[i] - prev_vol;
            result[i] = if w_vol != 0.0 {
                w_tpv / w_vol
            } else {
                f64::NAN
            };
        }
    }
    result
}

// ---------------------------------------------------------------------------
// VWMA
// ---------------------------------------------------------------------------

/// Volume Weighted Moving Average.
///
/// `VWMA = sum(close * volume, n) / sum(volume, n)`
///
/// # Arguments
/// * `close` — price series.
/// * `volume` — volume series (same length as `close`).
/// * `timeperiod` — rolling window size (>= 1).
///
/// # Returns
/// A `Vec<f64>` with `NaN` for the first `timeperiod - 1` entries.
pub fn vwma(close: &[f64], volume: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }

    let mut cum_cv = vec![0.0_f64; n];
    let mut cum_v = vec![0.0_f64; n];
    for i in 0..n {
        cum_cv[i] = close[i] * volume[i] + if i > 0 { cum_cv[i - 1] } else { 0.0 };
        cum_v[i] = volume[i] + if i > 0 { cum_v[i - 1] } else { 0.0 };
    }

    for i in (timeperiod - 1)..n {
        let prev_cv = if i >= timeperiod {
            cum_cv[i - timeperiod]
        } else {
            0.0
        };
        let prev_v = if i >= timeperiod {
            cum_v[i - timeperiod]
        } else {
            0.0
        };
        let w_cv = cum_cv[i] - prev_cv;
        let w_v = cum_v[i] - prev_v;
        result[i] = if w_v != 0.0 { w_cv / w_v } else { f64::NAN };
    }
    result
}

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

    if timeperiod < 1 || n <= timeperiod {
        return (supertrend_out, direction);
    }

    let atr = compute_atr(high, low, close, timeperiod);

    let mut upper_band = vec![f64::NAN; n];
    let mut lower_band = vec![f64::NAN; n];

    let first_valid = timeperiod - 1;
    if first_valid >= n || atr[first_valid].is_nan() {
        return (supertrend_out, direction);
    }

    // Initialize band state at first valid ATR bar (compute basic bands inline)
    {
        let hl2 = (high[first_valid] + low[first_valid]) / 2.0;
        upper_band[first_valid] = hl2 + multiplier * atr[first_valid];
        lower_band[first_valid] = hl2 - multiplier * atr[first_valid];
    }

    for i in (first_valid + 1)..n {
        if atr[i].is_nan() {
            continue;
        }

        // Compute basic bands as scalars — no Vec allocation needed
        let hl2 = (high[i] + low[i]) / 2.0;
        let upper_basic = hl2 + multiplier * atr[i];
        let lower_basic = hl2 - multiplier * atr[i];

        // Adjust lower band
        lower_band[i] = if lower_basic > lower_band[i - 1] || close[i - 1] < lower_band[i - 1] {
            lower_basic
        } else {
            lower_band[i - 1]
        };

        // Adjust upper band
        upper_band[i] = if upper_basic < upper_band[i - 1] || close[i - 1] > upper_band[i - 1] {
            upper_basic
        } else {
            upper_band[i - 1]
        };

        // Direction and output only from index timeperiod (warmup = 0, NaN)
        if i >= timeperiod {
            let prev_dir = direction[i - 1];
            direction[i] = if prev_dir == 0 {
                if close[i] > upper_band[i] {
                    1
                } else {
                    -1
                }
            } else if prev_dir == -1 {
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
    }

    (supertrend_out, direction)
}

// ---------------------------------------------------------------------------
// DONCHIAN
// ---------------------------------------------------------------------------

/// Donchian Channels — rolling highest high / lowest low.
///
/// # Returns
/// `(upper, middle, lower)` arrays.
pub fn donchian(high: &[f64], low: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];

    if timeperiod < 1 || n < timeperiod {
        return (upper, middle, lower);
    }

    let hh = math::sliding_max(high, timeperiod);
    let ll = math::sliding_min(low, timeperiod);

    for i in 0..n {
        if !hh[i].is_nan() {
            upper[i] = hh[i];
            lower[i] = ll[i];
            middle[i] = (upper[i] + lower[i]) / 2.0;
        }
    }

    (upper, middle, lower)
}

// ---------------------------------------------------------------------------
// CHOPPINESS_INDEX
// ---------------------------------------------------------------------------

/// Choppiness Index — measures market choppiness vs trending.
///
/// Values near 100 indicate a choppy market; near 0 indicates trending.
/// The first `timeperiod` values are `NaN`.
pub fn choppiness_index(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n <= timeperiod {
        return result;
    }

    // ATR(1) = True Range per bar
    let mut tr = vec![0.0_f64; n];
    tr[0] = high[0] - low[0];
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Cumulative TR for rolling sum
    let mut cum_tr = vec![0.0_f64; n];
    cum_tr[0] = tr[0];
    for i in 1..n {
        cum_tr[i] = cum_tr[i - 1] + tr[i];
    }

    let log_n = (timeperiod as f64).log10();

    let hh = math::sliding_max(high, timeperiod);
    let ll = math::sliding_min(low, timeperiod);

    for i in (timeperiod)..n {
        let prev_cum = cum_tr[i - timeperiod];
        let sum_tr = cum_tr[i] - prev_cum;
        let hl_range = hh[i] - ll[i];
        if hl_range > 0.0 && log_n > 0.0 {
            result[i] = 100.0 * (sum_tr / hl_range).log10() / log_n;
        }
    }

    result
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
    if timeperiod < 1 || atr_period < 1 || n < timeperiod || n < atr_period {
        let nan = vec![f64::NAN; n];
        return (nan.clone(), nan.clone(), nan);
    }

    let middle = overlap::ema(close, timeperiod);
    let atr = compute_atr(high, low, close, atr_period);

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
    let sqrt_p = ((timeperiod as f64).sqrt().round() as usize).max(1);

    let wma_full = overlap::wma(close, timeperiod);
    let wma_half = overlap::wma(close, half);

    // raw = 2 * wma_half - wma_full
    let mut raw = vec![f64::NAN; n];
    for i in 0..n {
        if !wma_full[i].is_nan() && !wma_half[i].is_nan() {
            raw[i] = 2.0 * wma_half[i] - wma_full[i];
        }
    }

    // Find first valid index in raw
    let first_valid = raw.iter().position(|x| !x.is_nan()).unwrap_or(n);
    let mut hull = vec![f64::NAN; n];
    if first_valid < n {
        let raw_valid = &raw[first_valid..];
        let hma_slice = overlap::wma(raw_valid, sqrt_p);
        for (k, &v) in hma_slice.iter().enumerate() {
            hull[first_valid + k] = v;
        }
    }

    hull
}

// ---------------------------------------------------------------------------
// CHANDELIER_EXIT
// ---------------------------------------------------------------------------

/// Chandelier Exit — ATR-based trailing stop levels.
///
/// # Returns
/// `(long_exit, short_exit)` arrays.
pub fn chandelier_exit(
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

    let atr = compute_atr(high, low, close, timeperiod);

    let highest_high = math::sliding_max(high, timeperiod);
    let lowest_low = math::sliding_min(low, timeperiod);

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

// ---------------------------------------------------------------------------
// ICHIMOKU
// ---------------------------------------------------------------------------

/// Ichimoku Cloud (Ichimoku Kinko Hyo).
///
/// # Returns
/// `(tenkan, kijun, senkou_a, senkou_b, chikou)` arrays.
pub fn ichimoku(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
    displacement: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let nan = || vec![f64::NAN; n];

    if tenkan_period < 1 || kijun_period < 1 || senkou_b_period < 1 {
        return (nan(), nan(), nan(), nan(), nan());
    }

    // Helper: rolling (H+L)/2 via shared sliding_max / sliding_min
    let midpoint_rolling = |period: usize| -> Vec<f64> {
        let hh = math::sliding_max(high, period);
        let ll = math::sliding_min(low, period);
        let mut result = vec![f64::NAN; n];
        for i in 0..n {
            if !hh[i].is_nan() {
                result[i] = (hh[i] + ll[i]) / 2.0;
            }
        }
        result
    };

    let tenkan = midpoint_rolling(tenkan_period);
    let kijun = midpoint_rolling(kijun_period);
    let raw_b = midpoint_rolling(senkou_b_period);

    // Senkou A: (tenkan + kijun) / 2 shifted back `displacement` bars
    let mut senkou_a = vec![f64::NAN; n];
    if n > displacement {
        for i in displacement..n {
            if !tenkan[i].is_nan() && !kijun[i].is_nan() {
                senkou_a[i - displacement] = (tenkan[i] + kijun[i]) / 2.0;
            }
        }
    }

    // Senkou B: raw_b shifted back `displacement` bars
    let mut senkou_b = vec![f64::NAN; n];
    if n > displacement {
        senkou_b[..n - displacement].copy_from_slice(&raw_b[displacement..]);
    }

    // Chikou: close shifted forward `displacement` bars
    let mut chikou = vec![f64::NAN; n];
    if n > displacement {
        chikou[displacement..].copy_from_slice(&close[..n - displacement]);
    }

    (tenkan, kijun, senkou_a, senkou_b, chikou)
}

// ---------------------------------------------------------------------------
// PIVOT_POINTS
// ---------------------------------------------------------------------------

/// Pivot Points — support / resistance levels computed from the previous bar.
///
/// # Arguments
/// * `method` — `"classic"`, `"fibonacci"`, or `"camarilla"`. Returns all-NaN
///   vectors for unknown methods.
///
/// # Returns
/// `(pivot, r1, s1, r2, s2)` arrays. Index 0 is always `NaN` (no previous bar).
pub fn pivot_points(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    method: &str,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut pivot = vec![f64::NAN; n];
    let mut r1 = vec![f64::NAN; n];
    let mut s1 = vec![f64::NAN; n];
    let mut r2 = vec![f64::NAN; n];
    let mut s2 = vec![f64::NAN; n];

    let method_lower = method.to_lowercase();
    if !matches!(method_lower.as_str(), "classic" | "fibonacci" | "camarilla") {
        // Unknown method — return all NaN
        return (pivot, r1, s1, r2, s2);
    }

    for i in 1..n {
        let ph = high[i - 1];
        let pl = low[i - 1];
        let pc = close[i - 1];
        let hl = ph - pl;
        let p = (ph + pl + pc) / 3.0;
        pivot[i] = p;
        match method_lower.as_str() {
            "classic" => {
                r1[i] = 2.0 * p - pl;
                s1[i] = 2.0 * p - ph;
                r2[i] = p + hl;
                s2[i] = p - hl;
            }
            "fibonacci" => {
                r1[i] = p + 0.382 * hl;
                s1[i] = p - 0.382 * hl;
                r2[i] = p + 0.618 * hl;
                s2[i] = p - 0.618 * hl;
            }
            "camarilla" => {
                r1[i] = pc + 1.1 * hl / 12.0;
                s1[i] = pc - 1.1 * hl / 12.0;
                r2[i] = pc + 1.1 * hl / 6.0;
                s2[i] = pc - 1.1 * hl / 6.0;
            }
            _ => unreachable!(),
        }
    }

    (pivot, r1, s1, r2, s2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Shared test data: 10-bar OHLCV
    fn sample_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![11.0, 12.0, 13.0, 14.0, 15.0, 14.5, 15.5, 16.0, 15.0, 14.0];
        let low = vec![9.0, 10.0, 11.0, 12.0, 13.0, 12.5, 13.5, 14.0, 13.0, 12.0];
        let close = vec![10.0, 11.0, 12.0, 13.0, 14.0, 13.5, 14.5, 15.0, 14.0, 13.0];
        let volume = vec![
            100.0, 150.0, 200.0, 250.0, 300.0, 200.0, 350.0, 400.0, 180.0, 220.0,
        ];
        (high, low, close, volume)
    }

    // -----------------------------------------------------------------------
    // VWAP tests
    // -----------------------------------------------------------------------

    #[test]
    fn vwap_cumulative_basic() {
        let (h, l, c, v) = sample_ohlcv();
        let result = vwap(&h, &l, &c, &v, 0);
        assert_eq!(result.len(), h.len());
        // First bar: tp = (11+9+10)/3 = 10.0, tpv = 1000.0, vol = 100.0 => 10.0
        assert!((result[0] - 10.0).abs() < 1e-10);
        // All values should be non-NaN for cumulative
        for val in &result {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn vwap_empty_input() {
        let result = vwap(&[], &[], &[], &[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn vwap_rolling_basic() {
        let (h, l, c, v) = sample_ohlcv();
        let result = vwap(&h, &l, &c, &v, 3);
        assert_eq!(result.len(), h.len());
        // First 2 values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // From index 2 onward should be valid
        assert!(!result[2].is_nan());
    }

    // -----------------------------------------------------------------------
    // VWMA tests
    // -----------------------------------------------------------------------

    #[test]
    fn vwma_basic() {
        let (_, _, c, v) = sample_ohlcv();
        let result = vwma(&c, &v, 3);
        assert_eq!(result.len(), c.len());
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Index 2: sum(c*v, 0..3) / sum(v, 0..3) = (1000+1650+2400)/(100+150+200) = 5050/450
        let expected = (10.0 * 100.0 + 11.0 * 150.0 + 12.0 * 200.0) / (100.0 + 150.0 + 200.0);
        assert!((result[2] - expected).abs() < 1e-10);
    }

    #[test]
    fn vwma_empty_input() {
        let result = vwma(&[], &[], 3);
        assert!(result.is_empty());
    }

    #[test]
    fn vwma_period_larger_than_data() {
        let result = vwma(&[1.0, 2.0], &[100.0, 200.0], 5);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|v| v.is_nan()));
    }

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
    fn hull_ma_empty_input() {
        let result = hull_ma(&[], 4);
        assert!(result.is_empty());
    }

    #[test]
    fn hull_ma_period_larger_than_data() {
        let result = hull_ma(&[1.0, 2.0], 10);
        assert!(result.iter().all(|v| v.is_nan()));
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

    // -----------------------------------------------------------------------
    // ICHIMOKU tests
    // -----------------------------------------------------------------------

    #[test]
    fn ichimoku_basic() {
        // Use a larger dataset for ichimoku
        let n = 60;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 + 1.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 - 1.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let (tenkan, kijun, senkou_a, senkou_b, chikou) =
            ichimoku(&high, &low, &close, 9, 26, 52, 26);

        assert_eq!(tenkan.len(), n);
        assert_eq!(kijun.len(), n);
        assert_eq!(senkou_a.len(), n);
        assert_eq!(senkou_b.len(), n);
        assert_eq!(chikou.len(), n);

        // Tenkan: period 9, first valid at index 8
        assert!(tenkan[7].is_nan());
        assert!(!tenkan[8].is_nan());

        // Kijun: period 26, first valid at index 25
        assert!(kijun[24].is_nan());
        assert!(!kijun[25].is_nan());

        // Chikou: close shifted forward by 26 bars
        assert!(chikou[25].is_nan());
        assert!(!chikou[26].is_nan());
        assert!((chikou[26] - close[0]).abs() < 1e-10);
    }

    #[test]
    fn ichimoku_empty_input() {
        let (t, k, sa, sb, ch) = ichimoku(&[], &[], &[], 9, 26, 52, 26);
        assert!(t.is_empty());
        assert!(k.is_empty());
        assert!(sa.is_empty());
        assert!(sb.is_empty());
        assert!(ch.is_empty());
    }

    // -----------------------------------------------------------------------
    // PIVOT_POINTS tests
    // -----------------------------------------------------------------------

    #[test]
    fn pivot_points_classic() {
        let h = vec![10.0, 12.0, 11.0];
        let l = vec![8.0, 9.0, 8.5];
        let c = vec![9.0, 11.0, 10.0];
        let (pivot, r1, s1, r2, s2) = pivot_points(&h, &l, &c, "classic");
        assert_eq!(pivot.len(), 3);
        // Index 0 is NaN
        assert!(pivot[0].is_nan());
        // Index 1: prev bar H=10, L=8, C=9 => P=(10+8+9)/3=9.0
        assert!((pivot[1] - 9.0).abs() < 1e-10);
        // R1 = 2*P - L = 18 - 8 = 10
        assert!((r1[1] - 10.0).abs() < 1e-10);
        // S1 = 2*P - H = 18 - 10 = 8
        assert!((s1[1] - 8.0).abs() < 1e-10);
        // R2 = P + (H-L) = 9 + 2 = 11
        assert!((r2[1] - 11.0).abs() < 1e-10);
        // S2 = P - (H-L) = 9 - 2 = 7
        assert!((s2[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn pivot_points_fibonacci() {
        let h = vec![10.0, 12.0];
        let l = vec![8.0, 9.0];
        let c = vec![9.0, 11.0];
        let (pivot, r1, s1, _, _) = pivot_points(&h, &l, &c, "fibonacci");
        // Index 1: P = (10+8+9)/3 = 9.0, HL = 2
        assert!((pivot[1] - 9.0).abs() < 1e-10);
        assert!((r1[1] - (9.0 + 0.382 * 2.0)).abs() < 1e-10);
        assert!((s1[1] - (9.0 - 0.382 * 2.0)).abs() < 1e-10);
    }

    #[test]
    fn pivot_points_camarilla() {
        let h = vec![10.0, 12.0];
        let l = vec![8.0, 9.0];
        let c = vec![9.0, 11.0];
        let (pivot, r1, s1, _, _) = pivot_points(&h, &l, &c, "camarilla");
        assert!((pivot[1] - 9.0).abs() < 1e-10);
        // R1 = C + 1.1 * HL / 12 = 9 + 1.1*2/12
        assert!((r1[1] - (9.0 + 1.1 * 2.0 / 12.0)).abs() < 1e-10);
        assert!((s1[1] - (9.0 - 1.1 * 2.0 / 12.0)).abs() < 1e-10);
    }

    #[test]
    fn pivot_points_unknown_method() {
        let h = vec![10.0, 12.0];
        let l = vec![8.0, 9.0];
        let c = vec![9.0, 11.0];
        let (pivot, r1, s1, r2, s2) = pivot_points(&h, &l, &c, "unknown");
        assert!(pivot.iter().all(|v| v.is_nan()));
        assert!(r1.iter().all(|v| v.is_nan()));
        assert!(s1.iter().all(|v| v.is_nan()));
        assert!(r2.iter().all(|v| v.is_nan()));
        assert!(s2.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn pivot_points_empty_input() {
        let (p, r1, s1, r2, s2) = pivot_points(&[], &[], &[], "classic");
        assert!(p.is_empty());
        assert!(r1.is_empty());
        assert!(s1.is_empty());
        assert!(r2.is_empty());
        assert!(s2.is_empty());
    }
}
