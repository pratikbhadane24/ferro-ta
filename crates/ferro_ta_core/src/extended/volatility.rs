//! Volatility extended indicators (CHAIKIN_VOL, MASS, BBPERCENT, BBWIDTH,
//! HISTORICAL_VOLATILITY, ULCER_INDEX, STARC).

use crate::math;
use crate::momentum;
use crate::overlap;
use crate::volatility;

/// Chaikin Volatility: rate of change of an EMA of the high–low range.
///
/// `100 * (EMA(H-L) / EMA(H-L)[rocperiod] - 1)`.
///
/// # Arguments
/// * `high` / `low` — equal-length price slices.
/// * `timeperiod` — EMA length of the range (classic default 10).
/// * `rocperiod` — ROC lookback of that EMA (classic default 10).
pub fn chaikin_vol(high: &[f64], low: &[f64], timeperiod: usize, rocperiod: usize) -> Vec<f64> {
    let n = high.len();
    if timeperiod < 1 || rocperiod < 1 || n == 0 {
        return vec![f64::NAN; n];
    }
    let mut range = vec![0.0; n];
    for i in 0..n {
        range[i] = high[i] - low[i];
    }
    let ema_range = overlap::ema(&range, timeperiod);
    momentum::roc(&ema_range, rocperiod)
}

/// Mass Index: rolling sum of the single/double EMA ratio of the high–low range.
///
/// `ratio = EMA(H-L, timeperiod) / EMA(EMA(H-L, timeperiod), timeperiod)`,
/// then `MASS = SUM(ratio, sumperiod)`.
///
/// # Arguments
/// * `high` / `low` — equal-length price slices.
/// * `timeperiod` — EMA length (classic default 9).
/// * `sumperiod` — rolling sum length (classic default 25).
pub fn mass(high: &[f64], low: &[f64], timeperiod: usize, sumperiod: usize) -> Vec<f64> {
    let n = high.len();
    if timeperiod < 1 || sumperiod < 1 || n == 0 {
        return vec![f64::NAN; n];
    }
    let mut range = vec![0.0; n];
    for i in 0..n {
        range[i] = high[i] - low[i];
    }
    let single = overlap::ema(&range, timeperiod);
    let double = overlap::ema(&single, timeperiod);
    let mut ratio = vec![f64::NAN; n];
    for i in 0..n {
        if single[i].is_finite() && double[i].is_finite() && double[i] != 0.0 {
            ratio[i] = single[i] / double[i];
        }
    }
    sum_finite(&ratio, sumperiod)
}

/// Bollinger %B: `(close - lower) / (upper - lower)` from [`overlap::bbands`].
///
/// `NaN` when the band range is zero.
pub fn bbpercent(close: &[f64], timeperiod: usize, nbdevup: f64, nbdevdn: f64) -> Vec<f64> {
    let (upper, _middle, lower) = overlap::bbands(close, timeperiod, nbdevup, nbdevdn, 0);
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        let span = upper[i] - lower[i];
        if span != 0.0 && upper[i].is_finite() && lower[i].is_finite() {
            result[i] = (close[i] - lower[i]) / span;
        }
    }
    result
}

/// Bollinger Bandwidth: `(upper - lower) / middle` from [`overlap::bbands`].
///
/// `NaN` when the middle band is zero.
pub fn bbwidth(close: &[f64], timeperiod: usize, nbdevup: f64, nbdevdn: f64) -> Vec<f64> {
    let (upper, middle, lower) = overlap::bbands(close, timeperiod, nbdevup, nbdevdn, 0);
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        if middle[i] != 0.0 && middle[i].is_finite() && upper[i].is_finite() && lower[i].is_finite()
        {
            result[i] = (upper[i] - lower[i]) / middle[i];
        }
    }
    result
}

/// Close-to-close historical volatility, annualized and in percent.
///
/// `stddev(ln(close / close[1]), timeperiod) * sqrt(annual) * 100`
/// using population variance (same as [`crate::statistic::stddev`]).
/// First valid value is at index `timeperiod` (one bar for the return,
/// then `timeperiod` returns).
///
/// # Arguments
/// * `close` — price series (must be positive for a defined log return).
/// * `timeperiod` — return-window length (classic default 20).
/// * `annual` — annualization factor (252 trading days by default).
pub fn historical_volatility(close: &[f64], timeperiod: usize, annual: f64) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < 2 {
        return result;
    }
    let mut log_ret = vec![f64::NAN; n];
    for i in 1..n {
        let prev = close[i - 1];
        let cur = close[i];
        if prev > 0.0 && cur > 0.0 {
            log_ret[i] = (cur / prev).ln();
        }
    }
    let scale = annual.max(0.0).sqrt() * 100.0;
    let p = timeperiod as f64;
    // First full window of finite log-returns ends at index `timeperiod`.
    for i in timeperiod..n {
        let start = i + 1 - timeperiod;
        let window = &log_ret[start..=i];
        if !window.iter().all(|v| v.is_finite()) {
            continue;
        }
        let mean = window.iter().sum::<f64>() / p;
        let var = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / p;
        result[i] = var.max(0.0).sqrt() * scale;
    }
    result
}

/// Ulcer Index: RMS of percent drawdowns versus the rolling highest close.
///
/// `PD = 100 * (close - MAX(close)) / MAX(close)`,
/// `UI = sqrt(SMA(PD², timeperiod))`.
///
/// # Arguments
/// * `close` — price series.
/// * `timeperiod` — lookback for the peak and the RMS (classic default 14).
pub fn ulcer_index(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 || n == 0 {
        return vec![f64::NAN; n];
    }
    let peak = math::max(close, timeperiod);
    let mut pd2 = vec![f64::NAN; n];
    for i in 0..n {
        if peak[i].is_finite() && peak[i] != 0.0 && close[i].is_finite() {
            let pd = 100.0 * (close[i] - peak[i]) / peak[i];
            pd2[i] = pd * pd;
        }
    }
    let mut result = sma_finite(&pd2, timeperiod);
    for v in &mut result {
        if v.is_finite() {
            *v = v.sqrt();
        }
    }
    result
}

/// Stoller Average Range Channels: SMA of close ± `multiplier * ATR`.
///
/// # Arguments
/// * `high` / `low` / `close` — equal-length OHLC slices.
/// * `timeperiod` — SMA length of close (classic default 15).
/// * `atr_period` — ATR length (classic default 15).
/// * `multiplier` — ATR multiple (classic default 2.0).
pub fn starc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
    atr_period: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    if timeperiod < 1 || atr_period < 1 || n == 0 {
        let nan = vec![f64::NAN; n];
        return (nan.clone(), nan.clone(), nan);
    }
    let middle = overlap::sma(close, timeperiod);
    let atr = volatility::atr(high, low, close, atr_period);
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in 0..n {
        if middle[i].is_finite() && atr[i].is_finite() {
            let band = multiplier * atr[i];
            upper[i] = middle[i] + band;
            lower[i] = middle[i] - band;
        }
    }
    (upper, middle, lower)
}

/// Rolling sum that ignores a leading (or mid-series) NaN run.
fn sum_finite(src: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = src.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    let mut run = 0usize;
    let mut win = 0.0;
    for i in 0..n {
        if src[i].is_finite() {
            run += 1;
            win += src[i];
            if run > timeperiod {
                win -= src[i - timeperiod];
                result[i] = win;
            } else if run == timeperiod {
                result[i] = win;
            }
        } else {
            run = 0;
            win = 0.0;
        }
    }
    result
}

/// Rolling SMA that ignores a leading (or mid-series) NaN run.
fn sma_finite(src: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = src.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 {
        return result;
    }
    let p = timeperiod as f64;
    for (i, v) in sum_finite(src, timeperiod).into_iter().enumerate() {
        if v.is_finite() {
            result[i] = v / p;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        (high, low, close)
    }

    #[test]
    fn chaikin_vol_empty() {
        assert!(chaikin_vol(&[], &[], 10, 10).is_empty());
    }

    #[test]
    fn chaikin_vol_constant_range_is_zero() {
        // cargo: chaikin_vol_constant_range_is_zero
        let (h, l, _) = linear_ohlc(25);
        let result = chaikin_vol(&h, &l, 3, 3);
        // EMA of constant 2 is 2; ROC is 0 once both EMA samples exist.
        let first = result.iter().position(|v| v.is_finite());
        assert_eq!(first, Some(5));
        for &v in &result[5..] {
            assert!(v.abs() < 1e-10, "{v}");
        }
    }

    #[test]
    fn mass_empty() {
        assert!(mass(&[], &[], 9, 25).is_empty());
    }

    #[test]
    fn mass_constant_range_equals_sumperiod() {
        // cargo: mass_constant_range_equals_sumperiod
        let (h, l, _) = linear_ohlc(20);
        let result = mass(&h, &l, 3, 3);
        let first = result.iter().position(|v| v.is_finite()).expect("valid");
        assert!(first >= 4);
        for &v in &result[first..] {
            assert!((v - 3.0).abs() < 1e-10, "{v}");
        }
    }

    #[test]
    fn bbpercent_bbwidth_from_bbands() {
        // cargo: bbpercent_bbwidth_from_bbands
        let close = [1.0, 2.0, 3.0, 4.0, 5.0];
        let pct = bbpercent(&close, 3, 2.0, 2.0);
        let width = bbwidth(&close, 3, 2.0, 2.0);
        let (upper, middle, lower) = overlap::bbands(&close, 3, 2.0, 2.0, 0);
        assert!(pct[0].is_nan() && pct[1].is_nan());
        let std = (2.0_f64 / 3.0).sqrt();
        // %B[2] = (3 - (2-2s)) / (4s) = (1+2s)/(4s)
        let expected_pct = (1.0 + 2.0 * std) / (4.0 * std);
        assert!((pct[2] - expected_pct).abs() < 1e-12);
        assert!((width[2] - 2.0 * std).abs() < 1e-12);
        for i in 2..5 {
            assert!(((close[i] - lower[i]) / (upper[i] - lower[i]) - pct[i]).abs() < 1e-12);
            assert!(((upper[i] - lower[i]) / middle[i] - width[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn historical_volatility_constant_return_is_zero() {
        // cargo: historical_volatility_constant_return_is_zero
        let close: Vec<f64> = (0..8).map(|i| 2.0_f64.powi(i)).collect();
        let result = historical_volatility(&close, 3, 252.0);
        assert!(result[0].is_nan() && result[1].is_nan() && result[2].is_nan());
        for &v in &result[3..] {
            assert!(v.abs() < 1e-10, "{v}");
        }
    }

    #[test]
    fn ulcer_index_rising_series_is_zero() {
        // cargo: ulcer_index_rising_series_is_zero
        let close: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let result = ulcer_index(&close, 4);
        let first = result.iter().position(|v| v.is_finite()).expect("valid");
        for &v in &result[first..] {
            assert!(v.abs() < 1e-12, "{v}");
        }
    }

    #[test]
    fn starc_golden_linear() {
        // cargo: starc_golden_linear — SMA[i]=i, ATR=2, multiplier=1
        let (h, l, c) = linear_ohlc(10);
        let (upper, middle, lower) = starc(&h, &l, &c, 3, 3, 1.0);
        assert!(middle[0].is_nan() && middle[1].is_nan());
        // ATR first output at index 3; SMA valid from 2. Both finite from 3.
        assert!(upper[2].is_nan());
        for i in 3..10 {
            assert!(
                (middle[i] - i as f64).abs() < 1e-10,
                "mid[{i}]={}",
                middle[i]
            );
            assert!((upper[i] - (i as f64 + 2.0)).abs() < 1e-10);
            assert!((lower[i] - (i as f64 - 2.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn bbpercent_zero_width_is_nan() {
        let close = [5.0, 5.0, 5.0, 5.0];
        let pct = bbpercent(&close, 2, 2.0, 2.0);
        assert!(pct.iter().all(|v| v.is_nan()));
    }
}
