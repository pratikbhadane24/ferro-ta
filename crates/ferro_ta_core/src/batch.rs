//! Pure-Rust batch operations — apply indicators across multiple series
//! (columns) sequentially. The PyO3 wrapper can add Rayon parallelism on top.
//!
//! Input convention: `data[j]` is column *j* (one time-series). All columns
//! must have the same length.

use crate::{momentum, overlap, statistic, volatility};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Validate that every column in `data` has the same length.  Returns `Ok(n)`
/// where `n` is the common length, or `Err` with a message.
fn validate_columns(data: &[Vec<f64>]) -> Result<usize, String> {
    if data.is_empty() {
        return Ok(0);
    }
    let n = data[0].len();
    for (idx, col) in data.iter().enumerate() {
        if col.len() != n {
            return Err(format!(
                "column 0 has length {n}, but column {idx} has length {}",
                col.len()
            ));
        }
    }
    Ok(n)
}

fn validate_hlc_columns(
    high: &[Vec<f64>],
    low: &[Vec<f64>],
    close: &[Vec<f64>],
) -> Result<(usize, usize), String> {
    let n_series = high.len();
    if low.len() != n_series || close.len() != n_series {
        return Err(format!(
            "high has {} columns, low has {}, close has {} — must be equal",
            n_series,
            low.len(),
            close.len()
        ));
    }
    if n_series == 0 {
        return Ok((0, 0));
    }
    let n = high[0].len();
    for (idx, (h, (l, c))) in high
        .iter()
        .zip(low.iter().zip(close.iter()))
        .enumerate()
    {
        if h.len() != n || l.len() != n || c.len() != n {
            return Err(format!(
                "column {idx}: high len={}, low len={}, close len={} — must all be {n}",
                h.len(),
                l.len(),
                c.len()
            ));
        }
    }
    Ok((n, n_series))
}

// ---------------------------------------------------------------------------
// rolling linear regression (self-contained so core has no PyO3 dep)
// ---------------------------------------------------------------------------

fn linreg(window: &[f64]) -> (f64, f64) {
    let n = window.len() as f64;
    let sum_x: f64 = (0..window.len()).map(|i| i as f64).sum();
    let sum_y: f64 = window.iter().sum();
    let sum_xy: f64 = window.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..window.len()).map(|i| (i as f64).powi(2)).sum();
    let denom = n * sum_x2 - sum_x * sum_x;
    let slope = if denom != 0.0 {
        (n * sum_xy - sum_x * sum_y) / denom
    } else {
        0.0
    };
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

fn rolling_linreg_apply<F>(prices: &[f64], timeperiod: usize, mut map: F) -> Vec<f64>
where
    F: FnMut(f64, f64) -> f64,
{
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }

    if prices.iter().any(|value| !value.is_finite()) {
        for end in (timeperiod - 1)..n {
            let window = &prices[(end + 1 - timeperiod)..=end];
            let (slope, intercept) = linreg(window);
            result[end] = map(slope, intercept);
        }
        return result;
    }

    let period = timeperiod as f64;
    let last_x = (timeperiod - 1) as f64;
    let sum_x = last_x * period / 2.0;
    let sum_x2 = last_x * period * (2.0 * period - 1.0) / 6.0;
    let denom = period * sum_x2 - sum_x * sum_x;

    let mut sum_y = prices[..timeperiod].iter().sum::<f64>();
    let mut sum_xy = prices[..timeperiod]
        .iter()
        .enumerate()
        .map(|(idx, &value)| idx as f64 * value)
        .sum::<f64>();

    for end in (timeperiod - 1)..n {
        let slope = if denom != 0.0 {
            (period * sum_xy - sum_x * sum_y) / denom
        } else {
            0.0
        };
        let intercept = (sum_y - slope * sum_x) / period;
        result[end] = map(slope, intercept);

        if end + 1 < n {
            let outgoing = prices[end + 1 - timeperiod];
            let incoming = prices[end + 1];
            let prev_sum_y = sum_y;

            sum_y = prev_sum_y - outgoing + incoming;
            sum_xy = sum_xy - (prev_sum_y - outgoing) + last_x * incoming;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// CCI / WILLR helpers (no external dep)
// ---------------------------------------------------------------------------

fn compute_cci(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let typical_price: Vec<f64> = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .map(|((&h, &l), &c)| (h + l + c) / 3.0)
        .collect();

    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }
    for end in (timeperiod - 1)..n {
        let window = &typical_price[(end + 1 - timeperiod)..=end];
        let mean = window.iter().sum::<f64>() / timeperiod as f64;
        let mad = window
            .iter()
            .map(|&value| (value - mean).abs())
            .sum::<f64>()
            / timeperiod as f64;
        result[end] = if mad != 0.0 {
            (typical_price[end] - mean) / (0.015 * mad)
        } else {
            0.0
        };
    }
    result
}

fn compute_willr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }

    // Use simple sliding-window max/min
    for end in (timeperiod - 1)..n {
        let start = end + 1 - timeperiod;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        for i in start..=end {
            if high[i] > highest {
                highest = high[i];
            }
            if low[i] < lowest {
                lowest = low[i];
            }
        }
        let range = highest - lowest;
        result[end] = if range != 0.0 {
            -100.0 * (highest - close[end]) / range
        } else {
            -50.0
        };
    }

    result
}

// ---------------------------------------------------------------------------
// batch_sma
// ---------------------------------------------------------------------------

/// Apply SMA to each column. Returns one output column per input column.
pub fn batch_sma(data: &[Vec<f64>], timeperiod: usize) -> Result<Vec<Vec<f64>>, String> {
    if timeperiod == 0 {
        return Err("timeperiod must be >= 1".into());
    }
    validate_columns(data)?;
    Ok(data
        .iter()
        .map(|col| overlap::sma(col, timeperiod))
        .collect())
}

// ---------------------------------------------------------------------------
// batch_ema
// ---------------------------------------------------------------------------

/// Apply EMA to each column.
pub fn batch_ema(data: &[Vec<f64>], timeperiod: usize) -> Result<Vec<Vec<f64>>, String> {
    if timeperiod == 0 {
        return Err("timeperiod must be >= 1".into());
    }
    validate_columns(data)?;
    Ok(data
        .iter()
        .map(|col| overlap::ema(col, timeperiod))
        .collect())
}

// ---------------------------------------------------------------------------
// batch_rsi
// ---------------------------------------------------------------------------

/// Apply RSI to each column.
pub fn batch_rsi(data: &[Vec<f64>], timeperiod: usize) -> Result<Vec<Vec<f64>>, String> {
    if timeperiod == 0 {
        return Err("timeperiod must be >= 1".into());
    }
    validate_columns(data)?;
    Ok(data
        .iter()
        .map(|col| momentum::rsi(col, timeperiod))
        .collect())
}

// ---------------------------------------------------------------------------
// batch_atr
// ---------------------------------------------------------------------------

/// Apply ATR to each set of (high, low, close) columns.
pub fn batch_atr(
    high: &[Vec<f64>],
    low: &[Vec<f64>],
    close: &[Vec<f64>],
    timeperiod: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if timeperiod == 0 {
        return Err("timeperiod must be >= 1".into());
    }
    validate_hlc_columns(high, low, close)?;
    Ok((0..high.len())
        .map(|i| volatility::atr(&high[i], &low[i], &close[i], timeperiod))
        .collect())
}

// ---------------------------------------------------------------------------
// batch_stoch
// ---------------------------------------------------------------------------

/// Apply Stochastic to each set of (high, low, close) columns.
/// Returns `(slowk_columns, slowd_columns)`.
pub fn batch_stoch(
    high: &[Vec<f64>],
    low: &[Vec<f64>],
    close: &[Vec<f64>],
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    validate_hlc_columns(high, low, close)?;
    let mut all_k = Vec::with_capacity(high.len());
    let mut all_d = Vec::with_capacity(high.len());
    for i in 0..high.len() {
        let (k, d) = momentum::stoch(
            &high[i],
            &low[i],
            &close[i],
            fastk_period,
            slowk_period,
            slowd_period,
        );
        all_k.push(k);
        all_d.push(d);
    }
    Ok((all_k, all_d))
}

// ---------------------------------------------------------------------------
// batch_adx
// ---------------------------------------------------------------------------

/// Apply ADX to each set of (high, low, close) columns.
pub fn batch_adx(
    high: &[Vec<f64>],
    low: &[Vec<f64>],
    close: &[Vec<f64>],
    timeperiod: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if timeperiod == 0 {
        return Err("timeperiod must be >= 1".into());
    }
    validate_hlc_columns(high, low, close)?;
    Ok((0..high.len())
        .map(|i| momentum::adx(&high[i], &low[i], &close[i], timeperiod))
        .collect())
}

// ---------------------------------------------------------------------------
// run_close_indicators
// ---------------------------------------------------------------------------

fn validate_indicator_requests(names: &[String], timeperiods: &[usize]) -> Result<(), String> {
    if names.len() != timeperiods.len() {
        return Err(format!(
            "names length ({}) must equal timeperiods length ({})",
            names.len(),
            timeperiods.len()
        ));
    }
    for (name, &tp) in names.iter().zip(timeperiods.iter()) {
        if tp == 0 {
            return Err(format!("{name}: timeperiod must be >= 1"));
        }
    }
    Ok(())
}

fn compute_close_indicator(
    name: &str,
    close: &[f64],
    timeperiod: usize,
) -> Result<Vec<f64>, String> {
    match name {
        "SMA" => Ok(overlap::sma(close, timeperiod)),
        "EMA" => Ok(overlap::ema(close, timeperiod)),
        "RSI" => Ok(momentum::rsi(close, timeperiod)),
        "STDDEV" => Ok(statistic::stddev(close, timeperiod, 1.0)),
        "VAR" => Ok(statistic::stddev(close, timeperiod, 1.0)
            .into_iter()
            .map(|v| if v.is_nan() { v } else { v * v })
            .collect()),
        "LINEARREG" => {
            let last_x = (timeperiod - 1) as f64;
            Ok(rolling_linreg_apply(close, timeperiod, |slope, intercept| {
                intercept + slope * last_x
            }))
        }
        "LINEARREG_SLOPE" => Ok(rolling_linreg_apply(close, timeperiod, |slope, _| slope)),
        "LINEARREG_INTERCEPT" => {
            Ok(rolling_linreg_apply(close, timeperiod, |_, intercept| {
                intercept
            }))
        }
        "LINEARREG_ANGLE" => Ok(rolling_linreg_apply(close, timeperiod, |slope, _| {
            slope.atan() * 180.0 / std::f64::consts::PI
        })),
        "TSF" => {
            let forecast_x = timeperiod as f64;
            Ok(rolling_linreg_apply(close, timeperiod, |slope, intercept| {
                intercept + slope * forecast_x
            }))
        }
        _ => Err(format!(
            "unsupported close indicator for grouped execution: {name}"
        )),
    }
}

/// Run multiple close-only indicators on the same series.
/// Returns `Vec<Result<Vec<f64>, String>>` — one result per (name, timeperiod) pair.
pub fn run_close_indicators(
    close: &[f64],
    names: &[String],
    timeperiods: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    validate_indicator_requests(names, timeperiods)?;
    let mut results = Vec::with_capacity(names.len());
    for (name, &tp) in names.iter().zip(timeperiods.iter()) {
        results.push(compute_close_indicator(name, close, tp)?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// run_hlc_indicators
// ---------------------------------------------------------------------------

fn compute_hlc_indicator(
    name: &str,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> Result<Vec<f64>, String> {
    match name {
        "ATR" => Ok(volatility::atr(high, low, close, timeperiod)),
        "NATR" => {
            let atr_vals = volatility::atr(high, low, close, timeperiod);
            Ok(atr_vals
                .into_iter()
                .zip(close.iter())
                .map(|(a, &c)| {
                    if a.is_nan() || c == 0.0 {
                        f64::NAN
                    } else {
                        (a / c) * 100.0
                    }
                })
                .collect())
        }
        "ADX" => Ok(momentum::adx(high, low, close, timeperiod)),
        "ADXR" => Ok(momentum::adxr(high, low, close, timeperiod)),
        "CCI" => Ok(compute_cci(high, low, close, timeperiod)),
        "WILLR" => Ok(compute_willr(high, low, close, timeperiod)),
        _ => Err(format!(
            "unsupported HLC indicator for grouped execution: {name}"
        )),
    }
}

/// Run multiple HLC indicators on the same series.
pub fn run_hlc_indicators(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    names: &[String],
    timeperiods: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    validate_indicator_requests(names, timeperiods)?;
    if high.len() != low.len() || high.len() != close.len() {
        return Err("high, low, and close must have equal length".into());
    }
    let mut results = Vec::with_capacity(names.len());
    for (name, &tp) in names.iter().zip(timeperiods.iter()) {
        results.push(compute_hlc_indicator(name, high, low, close, tp)?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close_data() -> Vec<f64> {
        vec![
            44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61,
            46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
        ]
    }

    fn hlc_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close = close_data();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        (high, low, close)
    }

    #[test]
    fn test_batch_sma_basic() {
        let col1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let col2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let data = vec![col1, col2];
        let result = batch_sma(&data, 3).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0][0].is_nan());
        assert!(result[0][1].is_nan());
        assert!((result[0][2] - 2.0).abs() < 1e-10);
        assert!((result[1][2] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_sma_zero_period() {
        let data = vec![vec![1.0, 2.0]];
        assert!(batch_sma(&data, 0).is_err());
    }

    #[test]
    fn test_batch_ema_basic() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let result = batch_ema(&data, 3).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0][0].is_nan());
    }

    #[test]
    fn test_batch_rsi_basic() {
        let data = vec![close_data()];
        let result = batch_rsi(&data, 14).unwrap();
        assert_eq!(result.len(), 1);
        // First 14 values should be NaN
        for i in 0..14 {
            assert!(result[0][i].is_nan(), "index {i} should be NaN");
        }
        // Value at index 14 should be a valid RSI
        let rsi_val = result[0][14];
        assert!(!rsi_val.is_nan());
        assert!(rsi_val >= 0.0 && rsi_val <= 100.0);
    }

    #[test]
    fn test_batch_atr_basic() {
        let (h, l, c) = hlc_data();
        let high = vec![h];
        let low = vec![l];
        let close = vec![c];
        let result = batch_atr(&high, &low, &close, 14).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_batch_stoch_basic() {
        let (h, l, c) = hlc_data();
        let high = vec![h];
        let low = vec![l];
        let close = vec![c];
        let (k, d) = batch_stoch(&high, &low, &close, 5, 3, 3).unwrap();
        assert_eq!(k.len(), 1);
        assert_eq!(d.len(), 1);
        assert_eq!(k[0].len(), d[0].len());
    }

    #[test]
    fn test_batch_adx_basic() {
        let (h, l, c) = hlc_data();
        let high = vec![h];
        let low = vec![l];
        let close = vec![c];
        let result = batch_adx(&high, &low, &close, 14).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_run_close_indicators_basic() {
        let close = close_data();
        let names = vec!["SMA".to_string(), "EMA".to_string()];
        let timeperiods = vec![5, 5];
        let result = run_close_indicators(&close, &names, &timeperiods).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), close.len());
        assert_eq!(result[1].len(), close.len());
    }

    #[test]
    fn test_run_close_indicators_mismatched_lengths() {
        let close = close_data();
        let names = vec!["SMA".to_string()];
        let timeperiods = vec![5, 10]; // different length
        assert!(run_close_indicators(&close, &names, &timeperiods).is_err());
    }

    #[test]
    fn test_run_close_indicators_linreg_variants() {
        let close = close_data();
        let names = vec![
            "LINEARREG".to_string(),
            "LINEARREG_SLOPE".to_string(),
            "LINEARREG_INTERCEPT".to_string(),
            "LINEARREG_ANGLE".to_string(),
            "TSF".to_string(),
        ];
        let timeperiods = vec![5, 5, 5, 5, 5];
        let result = run_close_indicators(&close, &names, &timeperiods).unwrap();
        assert_eq!(result.len(), 5);
        // First 4 values should be NaN for period=5
        for series in &result {
            for i in 0..4 {
                assert!(series[i].is_nan());
            }
            assert!(!series[4].is_nan());
        }
    }

    #[test]
    fn test_run_hlc_indicators_basic() {
        let (h, l, c) = hlc_data();
        let names = vec!["ATR".to_string(), "CCI".to_string()];
        let timeperiods = vec![14, 14];
        let result = run_hlc_indicators(&h, &l, &c, &names, &timeperiods).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_run_hlc_indicators_unsupported() {
        let (h, l, c) = hlc_data();
        let names = vec!["UNKNOWN".to_string()];
        let timeperiods = vec![14];
        assert!(run_hlc_indicators(&h, &l, &c, &names, &timeperiods).is_err());
    }

    #[test]
    fn test_validate_hlc_mismatched_columns() {
        let high = vec![vec![1.0, 2.0]];
        let low = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // 2 cols vs 1
        let close = vec![vec![1.0, 2.0]];
        assert!(batch_atr(&high, &low, &close, 5).is_err());
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        let result = batch_sma(&data, 3).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_multiple_columns() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
        ];
        let result = batch_sma(&data, 3).unwrap();
        assert_eq!(result.len(), 3);
        // col 0: sma(3) at index 2 = (1+2+3)/3 = 2.0
        assert!((result[0][2] - 2.0).abs() < 1e-10);
        // col 1: sma(3) at index 2 = (5+4+3)/3 = 4.0
        assert!((result[1][2] - 4.0).abs() < 1e-10);
        // col 2: sma(3) at index 2 = (2+4+6)/3 = 4.0
        assert!((result[2][2] - 4.0).abs() < 1e-10);
    }
}
