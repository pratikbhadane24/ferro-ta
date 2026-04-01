//! Volatility indicators.

/// Compute the Average True Range (ATR), Wilder smoothed (TA-Lib compatible).
///
/// ATR measures market volatility by smoothing the True Range with Wilder's
/// method. Seeded with the SMA of `TR[1..=timeperiod]` (bar 0 is skipped,
/// matching TA-Lib). Returns non-negative values; the first `timeperiod`
/// indices are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Smoothing period (typically 14).
pub fn atr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if n <= timeperiod || timeperiod < 1 {
        return result;
    }
    // Seed: SMA of TR[1..=timeperiod] (TA-Lib skips TR[0]).
    // Compute TR on-the-fly to avoid a separate Vec allocation.
    let mut seed = 0.0_f64;
    for i in 1..=timeperiod {
        let hl = high[i] - low[i];
        let hpc = (high[i] - close[i - 1]).abs();
        let lpc = (low[i] - close[i - 1]).abs();
        seed += hl.max(hpc).max(lpc);
    }
    seed /= timeperiod as f64;
    result[timeperiod] = seed;
    let p = timeperiod as f64;
    for i in (timeperiod + 1)..n {
        let hl = high[i] - low[i];
        let hpc = (high[i] - close[i - 1]).abs();
        let lpc = (low[i] - close[i - 1]).abs();
        let tr = hl.max(hpc).max(lpc);
        result[i] = (result[i - 1] * (p - 1.0) + tr) / p;
    }
    result
}

/// Compute the True Range for each bar.
///
/// `TR = max(H - L, |H - C_prev|, |L - C_prev|)`. For bar 0, TR is
/// simply `H - L` (no previous close available). Returns non-negative
/// values for every bar (no `NaN` warmup).
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
pub fn trange(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }
    result[0] = high[0] - low[0];
    for i in 1..n {
        let hl = high[i] - low[i];
        let hpc = (high[i] - close[i - 1]).abs();
        let lpc = (low[i] - close[i - 1]).abs();
        result[i] = hl.max(hpc).max(lpc);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atr_nonnegative() {
        let h = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let l = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let c = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let result = atr(&h, &l, &c, 3);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0);
        }
    }
}
