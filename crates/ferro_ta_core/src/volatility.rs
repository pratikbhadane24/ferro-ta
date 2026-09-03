//! Volatility indicators.

/// True range of one bar given the previous close.
///
/// `TR = max(H - L, |H - C_prev|, |L - C_prev|)`.
#[inline]
pub(crate) fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    let hl = high - low;
    hl.max((high - prev_close).abs())
        .max((low - prev_close).abs())
}

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
    let mut seed = 0.0_f64;
    for i in 1..=timeperiod {
        seed += true_range(high[i], low[i], close[i - 1]);
    }
    seed /= timeperiod as f64;
    result[timeperiod] = seed;
    let p = timeperiod as f64;
    for i in (timeperiod + 1)..n {
        let tr = true_range(high[i], low[i], close[i - 1]);
        result[i] = (result[i - 1] * (p - 1.0) + tr) / p;
    }
    result
}

/// Compute the True Range for each bar.
///
/// `TR = max(H - L, |H - C_prev|, |L - C_prev|)`. Bar 0 is `NaN` (TA-Lib:
/// no previous close). Remaining bars are non-negative.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
pub fn trange(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }
    for i in 1..n {
        result[i] = true_range(high[i], low[i], close[i - 1]);
    }
    result
}

/// Normalized Average True Range: `ATR / close * 100`.
pub fn natr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let atr_vals = atr(high, low, close, timeperiod);
    atr_vals
        .iter()
        .zip(close.iter())
        .map(|(&a, &c)| {
            if a.is_nan() || c == 0.0 {
                f64::NAN
            } else {
                a / c * 100.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < tol
    }

    /// Bar 0 has a wide range so a seed that includes TR[0] cannot match TA-Lib.
    fn wide_first_bar() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![20.0, 12.0, 13.0, 14.0, 15.0],
            vec![5.0, 10.0, 11.0, 12.0, 13.0],
            vec![10.0, 11.0, 12.0, 13.0, 14.0],
        )
    }

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

    #[test]
    fn trange_bar0_is_nan() {
        let h = vec![11.0, 13.0, 14.0];
        let l = vec![9.0, 10.0, 11.0];
        let c = vec![10.0, 12.0, 13.0];
        let result = trange(&h, &l, &c);
        assert!(
            result[0].is_nan(),
            "TA-Lib TRANGE[0] is NaN, got {}",
            result[0]
        );
        // TR[1] = max(13-10, |13-10|, |10-10|) = 3
        assert!(approx_eq(result[1], 3.0, 1e-12));
        // TR[2] = max(14-11, |14-12|, |11-12|) = 3
        assert!(approx_eq(result[2], 3.0, 1e-12));
    }

    #[test]
    fn atr_seeds_from_tr_1_through_period() {
        let (h, l, c) = wide_first_bar();
        let result = atr(&h, &l, &c, 3);
        // TR[0]=15 is skipped. TR[1]=TR[2]=TR[3]=2 → first ATR at index 3 is 2.
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!(
            approx_eq(result[3], 2.0, 1e-12),
            "ATR[period] should be SMA(TR[1..=period])=2, got {}",
            result[3]
        );
        assert!(approx_eq(result[4], 2.0, 1e-12));
    }

    #[test]
    fn natr_uses_same_atr_seed() {
        let (h, l, c) = wide_first_bar();
        let result = natr(&h, &l, &c, 3);
        assert!(result[2].is_nan());
        assert!(approx_eq(result[3], 2.0 / 13.0 * 100.0, 1e-12));
    }
}
