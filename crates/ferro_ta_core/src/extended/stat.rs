//! Statistic extended indicators: MEDIAN, MEDIAN_BANDS, MODE.

/// Median of a copied window. `window` is sorted in place.
fn window_median(window: &mut [f64]) -> f64 {
    window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = window.len();
    if n % 2 == 1 {
        window[n / 2]
    } else {
        0.5 * (window[n / 2 - 1] + window[n / 2])
    }
}

/// Rolling median of `real` over `timeperiod`.
///
/// Even windows use the average of the two central values (same as NumPy).
/// The first `timeperiod - 1` entries are `NaN`. A window that contains any
/// non-finite value also yields `NaN`.
pub fn median(real: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    let mut buf = vec![0.0_f64; timeperiod];
    for end in (timeperiod - 1)..n {
        let start = end + 1 - timeperiod;
        let window = &real[start..=end];
        if window.iter().any(|v| !v.is_finite()) {
            continue;
        }
        buf.copy_from_slice(window);
        result[end] = window_median(&mut buf);
    }
    result
}

/// Median bands: rolling median of `(high + low) / 2`, ATR envelopes, and an
/// EMA of the median.
///
/// Returns `(median, upper, lower, median_ema)`.
///
/// * `timeperiod` — median / EMA window (typically 3).
/// * `atr_period` — ATR smoothing period (typically 14).
/// * `multiplier` — ATR width (`upper = median + multiplier * ATR`).
pub fn median_bands(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
    atr_period: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut source = vec![f64::NAN; n];
    for i in 0..n {
        source[i] = 0.5 * (high[i] + low[i]);
    }
    let mid = median(&source, timeperiod);
    let atr = crate::volatility::atr(high, low, close, atr_period);
    let mid_ema = crate::overlap::ema(&mid, timeperiod);
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in 0..n {
        if mid[i].is_finite() && atr[i].is_finite() {
            let width = multiplier * atr[i];
            upper[i] = mid[i] + width;
            lower[i] = mid[i] - width;
        }
    }
    (mid, upper, lower, mid_ema)
}

/// Rolling mode via equal-width discretization of each window.
///
/// Values are placed into `bins` buckets between the window min and max.
/// The output is the centre of the most populous bin. Ties take the
/// left-most bin. A constant window returns that constant. The first
/// `timeperiod - 1` entries are `NaN`.
pub fn mode(real: &[f64], timeperiod: usize, bins: usize) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || bins < 1 || n < timeperiod {
        return result;
    }
    let mut counts = vec![0_usize; bins];
    for end in (timeperiod - 1)..n {
        let window = &real[end + 1 - timeperiod..=end];
        if window.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &v in window {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        if max_v == min_v {
            result[end] = min_v;
            continue;
        }
        counts.fill(0);
        let width = (max_v - min_v) / bins as f64;
        for &v in window {
            let mut idx = ((v - min_v) / width).floor() as usize;
            if idx >= bins {
                idx = bins - 1;
            }
            counts[idx] += 1;
        }
        let mut best = 0_usize;
        let mut best_c = counts[0];
        for (i, &c) in counts.iter().enumerate().skip(1) {
            if c > best_c {
                best = i;
                best_c = c;
            }
        }
        result[end] = min_v + (best as f64 + 0.5) * width;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_window_golden() {
        let close = [1.0, 5.0, 3.0, 4.0, 2.0];
        let result = median(&close, 3);
        assert!(result[0].is_nan() && result[1].is_nan());
        assert!((result[2] - 3.0).abs() < 1e-12);
        assert!((result[3] - 4.0).abs() < 1e-12);
        assert!((result[4] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn median_even_window_averages_centres() {
        let close = [1.0, 2.0, 3.0, 4.0];
        let result = median(&close, 4);
        assert!(result[0].is_nan() && result[1].is_nan() && result[2].is_nan());
        assert!((result[3] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn median_period_one_is_identity() {
        let close = [7.0, 8.0, 9.0];
        let result = median(&close, 1);
        assert_eq!(result, close);
    }

    #[test]
    fn median_nan_in_window() {
        let close = [1.0, f64::NAN, 3.0];
        let result = median(&close, 3);
        assert!(result[2].is_nan());
    }

    #[test]
    fn median_empty_and_short() {
        assert!(median(&[], 3).is_empty());
        let short = median(&[1.0, 2.0], 3);
        assert_eq!(short.len(), 2);
        assert!(short.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn median_bands_compose_from_hl2_and_atr() {
        let high = [11.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 19.0];
        let low = [9.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0];
        let close = [10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 18.0];
        let (mid, upper, lower, mid_ema) = median_bands(&high, &low, &close, 3, 3, 2.0);
        let mut hl2 = [0.0; 8];
        for i in 0..8 {
            hl2[i] = 0.5 * (high[i] + low[i]);
        }
        let expected_mid = median(&hl2, 3);
        let atr = crate::volatility::atr(&high, &low, &close, 3);
        let expected_ema = crate::overlap::ema(&expected_mid, 3);
        for i in 0..8 {
            assert!(
                (mid[i].is_nan() && expected_mid[i].is_nan())
                    || (mid[i] - expected_mid[i]).abs() < 1e-12
            );
            if mid[i].is_finite() && atr[i].is_finite() {
                assert!((upper[i] - (mid[i] + 2.0 * atr[i])).abs() < 1e-12);
                assert!((lower[i] - (mid[i] - 2.0 * atr[i])).abs() < 1e-12);
            } else {
                assert!(upper[i].is_nan() && lower[i].is_nan());
            }
            assert!(
                (mid_ema[i].is_nan() && expected_ema[i].is_nan())
                    || (mid_ema[i] - expected_ema[i]).abs() < 1e-12
            );
        }
    }

    #[test]
    fn mode_constant_window() {
        let real = [4.0, 4.0, 4.0, 4.0];
        let result = mode(&real, 3, 10);
        assert!(result[0].is_nan() && result[1].is_nan());
        assert!((result[2] - 4.0).abs() < 1e-12);
        assert!((result[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn mode_binned_golden() {
        // Window [1, 1, 1, 2, 3], bins=2 → min=1, max=3, width=1.
        // bin0 [1, 2) gets three 1s; bin1 [2, 3] gets 2 and 3.
        // Mode is the centre of bin0: 1.5.
        let real = [1.0, 1.0, 1.0, 2.0, 3.0];
        let result = mode(&real, 5, 2);
        assert!((result[4] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn mode_empty() {
        assert!(mode(&[], 5, 10).is_empty());
    }
}
