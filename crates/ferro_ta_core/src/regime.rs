//! Regime detection and structural breaks.
//!
//! - `regime_adx`             — label trend (1) vs range (0) using ADX threshold
//! - `regime_combined`        — combine ADX + ATR-ratio for robust regime labelling
//! - `detect_breaks_cusum`    — CUSUM-based structural break detection
//! - `rolling_variance_break` — variance ratio break detection

/// Label each bar as trend (1) or range (0) based on ADX level.
///
/// Returns `Vec<i8>`: `1` = trend (ADX > threshold), `0` = range, `-1` = NaN/warmup.
pub fn regime_adx(adx: &[f64], threshold: f64) -> Vec<i8> {
    adx.iter()
        .map(|&v| {
            if v.is_nan() {
                -1i8
            } else if v > threshold {
                1i8
            } else {
                0i8
            }
        })
        .collect()
}

/// Label each bar as trend (1) or range (0) using ADX + ATR-ratio rule.
///
/// A bar is trending when: `adx[i] > adx_threshold` AND `atr[i] / close[i] > atr_pct_threshold`.
///
/// Returns `Vec<i8>`: `1` = trend, `0` = range, `-1` = NaN.
pub fn regime_combined(
    adx: &[f64],
    atr: &[f64],
    close: &[f64],
    adx_threshold: f64,
    atr_pct_threshold: f64,
) -> Vec<i8> {
    let n = adx.len();
    (0..n)
        .map(|i| {
            let av = adx[i];
            let rv = atr[i];
            let cv = close[i];
            if av.is_nan() || rv.is_nan() || cv.is_nan() || cv == 0.0 {
                -1i8
            } else if av > adx_threshold && (rv / cv) > atr_pct_threshold {
                1i8
            } else {
                0i8
            }
        })
        .collect()
}

/// Detect structural breaks using a CUSUM (cumulative sum) approach.
///
/// `window` must be >= 2. Returns `Vec<i8>`: `1` at break bars, `0` elsewhere.
pub fn detect_breaks_cusum(
    series: &[f64],
    window: usize,
    threshold: f64,
    slack: f64,
) -> Vec<i8> {
    let n = series.len();
    let mut out = vec![0i8; n];
    if n < window || window < 2 {
        return out;
    }
    let mut cusum_pos = 0.0_f64;
    let mut cusum_neg = 0.0_f64;
    for i in window..n {
        let slice = &series[(i - window)..i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 =
            slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (window - 1) as f64;
        let std = var.sqrt();
        if std == 0.0 || std.is_nan() || series[i].is_nan() {
            continue;
        }
        let z = (series[i] - mean) / std;
        cusum_pos = (cusum_pos + z - slack).max(0.0);
        cusum_neg = (cusum_neg - z - slack).max(0.0);
        if cusum_pos > threshold || cusum_neg > threshold {
            out[i] = 1;
            cusum_pos = 0.0;
            cusum_neg = 0.0;
        }
    }
    out
}

/// Detect volatility regime breaks using rolling variance ratio.
///
/// `short_window` must be >= 2, `long_window` must be > `short_window`.
/// Returns `Vec<i8>`: `1` at break bars, `0` elsewhere.
pub fn rolling_variance_break(
    series: &[f64],
    short_window: usize,
    long_window: usize,
    threshold: f64,
) -> Vec<i8> {
    let n = series.len();
    let mut out = vec![0i8; n];
    if n < long_window || short_window < 2 || long_window <= short_window {
        return out;
    }

    let variance = |slice: &[f64]| -> f64 {
        let k = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / k as f64;
        slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (k - 1) as f64
    };

    for i in long_window..n {
        let long_slice = &series[(i - long_window)..i];
        let short_slice = &series[(i - short_window)..i];
        let long_var = variance(long_slice);
        let short_var = variance(short_slice);
        if long_var == 0.0 || long_var.is_nan() || short_var.is_nan() {
            continue;
        }
        if short_var / long_var > threshold {
            out[i] = 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_adx_basic() {
        let adx = vec![f64::NAN, 20.0, 30.0, 10.0, 50.0];
        let result = regime_adx(&adx, 25.0);
        assert_eq!(result, vec![-1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_regime_combined() {
        let adx = vec![30.0, 30.0, 10.0];
        let atr = vec![1.0, 0.001, 1.0];
        let close = vec![100.0, 100.0, 100.0];
        let result = regime_combined(&adx, &atr, &close, 25.0, 0.005);
        assert_eq!(result[0], 1); // ADX>25 and ATR/close=0.01>0.005
        assert_eq!(result[1], 0); // ATR/close=0.00001 < 0.005
        assert_eq!(result[2], 0); // ADX<25
    }

    #[test]
    fn test_detect_breaks_cusum_short_input() {
        let series = vec![1.0, 2.0];
        let result = detect_breaks_cusum(&series, 5, 3.0, 0.5);
        assert!(result.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_rolling_variance_break_short_input() {
        let series = vec![1.0, 2.0, 3.0];
        let result = rolling_variance_break(&series, 2, 5, 2.0);
        assert!(result.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_empty() {
        assert!(regime_adx(&[], 25.0).is_empty());
        assert!(regime_combined(&[], &[], &[], 25.0, 0.005).is_empty());
        assert!(detect_breaks_cusum(&[], 2, 3.0, 0.5).is_empty());
        assert!(rolling_variance_break(&[], 2, 5, 2.0).is_empty());
    }
}
