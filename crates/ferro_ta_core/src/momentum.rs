//! Momentum indicators.

use crate::math::{sliding_max, sliding_min};

/// Relative Strength Index — TA-Lib compatible Wilder smoothing.
///
/// Seeds avg_gain/avg_loss with SMA of first `timeperiod` changes.
/// Uses branchless gain/loss split: `gain = diff.max(0.0)`, `loss = (-diff).max(0.0)`.
pub fn rsi(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if n <= timeperiod || timeperiod < 1 {
        return result;
    }
    let mut avg_gain = 0.0_f64;
    let mut avg_loss = 0.0_f64;
    for i in 1..=timeperiod {
        let diff = close[i] - close[i - 1];
        let abs_diff = diff.abs();
        avg_gain += (diff + abs_diff) * 0.5;
        avg_loss += (abs_diff - diff) * 0.5;
    }
    avg_gain /= timeperiod as f64;
    avg_loss /= timeperiod as f64;
    let p = timeperiod as f64;
    let rs = if avg_loss == 0.0 {
        f64::MAX
    } else {
        avg_gain / avg_loss
    };
    result[timeperiod] = 100.0 - 100.0 / (1.0 + rs);
    for i in (timeperiod + 1)..n {
        let diff = close[i] - close[i - 1];
        let abs_diff = diff.abs();
        let gain = (diff + abs_diff) * 0.5;
        let loss = (abs_diff - diff) * 0.5;
        avg_gain = (avg_gain * (p - 1.0) + gain) / p;
        avg_loss = (avg_loss * (p - 1.0) + loss) / p;
        let rs = if avg_loss == 0.0 {
            f64::MAX
        } else {
            avg_gain / avg_loss
        };
        result[i] = 100.0 - 100.0 / (1.0 + rs);
    }
    result
}

/// Momentum — `close[i] - close[i - timeperiod]`.
pub fn mom(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 {
        return result;
    }
    for i in timeperiod..n {
        result[i] = close[i] - close[i - timeperiod];
    }
    result
}

/// Stochastic Oscillator — TA-Lib compatible.
///
/// Returns `(slowk, slowd)`.
///  - Fast %K[i] = 100 * (close[i] - min(low, fastk_period)) / (max(high, fastk_period) - min(low, fastk_period))
///  - Slow %K = SMA(fast %K, slowk_period)
///  - Slow %D = SMA(slow %K, slowd_period)
///
/// Uses O(n) sliding max/min via monotonic deques.
pub fn stoch(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let nan_pair = || (vec![f64::NAN; n], vec![f64::NAN; n]);
    if n == 0 || fastk_period < 1 || slowk_period < 1 || slowd_period < 1 {
        return nan_pair();
    }
    if n < fastk_period {
        return nan_pair();
    }

    let max_h = sliding_max(high, fastk_period);
    let min_l = sliding_min(low, fastk_period);

    let mut slowk = vec![f64::NAN; n];
    let mut slowd = vec![f64::NAN; n];

    // Fast %K is valid from index fastk_period-1 onward.
    let fastk_start = fastk_period - 1;
    let mut fastk_valid = vec![0.0; n - fastk_start];
    for i in fastk_start..n {
        let range = max_h[i] - min_l[i];
        fastk_valid[i - fastk_start] = if range != 0.0 {
            100.0 * (close[i] - min_l[i]) / range
        } else {
            0.0
        };
    }

    // Slow %K = SMA(fastk_valid, slowk_period); write directly into `slowk` offset by `fastk_start`.
    crate::overlap::sma_into(&fastk_valid, slowk_period, &mut slowk, fastk_start);

    // Slow %D = SMA(slowk, slowd_period).
    // The valid part of slowk starts at `fastk_start + slowk_period - 1`.
    let slowk_valid_start = fastk_start + slowk_period - 1;
    let slowd_valid_start = slowk_valid_start + slowd_period - 1;

    if slowk_valid_start < n {
        let slowk_valid_slice = &slowk[slowk_valid_start..];
        crate::overlap::sma_into(
            slowk_valid_slice,
            slowd_period,
            &mut slowd,
            slowk_valid_start,
        );
    }

    // TA-Lib pads BOTH slowk and slowd with NaNs up to the point where both are valid.
    if slowd_valid_start < n {
        for v in slowk.iter_mut().take(slowd_valid_start) {
            *v = f64::NAN;
        }
    } else {
        for v in slowk.iter_mut().take(n) {
            *v = f64::NAN;
        }
    }

    (slowk, slowd)
}

// ---------------------------------------------------------------------------
// ADX family
// ---------------------------------------------------------------------------

/// Return type for ADX inner (pdm_s, mdm_s, plus_di, minus_di, dx, adx).
type AdxInnerOutput = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Fused inner function for ADX-family indicators.
/// Returns a tuple of (pdm_s, mdm_s, plus_di, minus_di, dx, adx).
fn adx_inner(high: &[f64], low: &[f64], close: &[f64], period: usize) -> AdxInnerOutput {
    let n = high.len();
    let mut b_pdm = vec![f64::NAN; n];
    let mut b_mdm = vec![f64::NAN; n];
    let mut b_pdi = vec![f64::NAN; n];
    let mut b_mdi = vec![f64::NAN; n];
    let mut b_dx = vec![f64::NAN; n];
    let mut b_adx = vec![f64::NAN; n];

    if n < period || period < 1 || n < 2 {
        return (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx);
    }

    let m = n - 1;
    let mut tr = vec![0.0_f64; m];
    let mut pdm = vec![0.0_f64; m];
    let mut mdm = vec![0.0_f64; m];

    for i in 0..m {
        let j = i + 1;
        let h_diff = high[j] - high[i];
        let l_diff = low[i] - low[j];
        let hl = high[j] - low[j];
        let hpc = (high[j] - close[i]).abs();
        let lpc = (low[j] - close[i]).abs();
        tr[i] = hl.max(hpc).max(lpc);
        pdm[i] = if h_diff > l_diff && h_diff > 0.0 {
            h_diff
        } else {
            0.0
        };
        mdm[i] = if l_diff > h_diff && l_diff > 0.0 {
            l_diff
        } else {
            0.0
        };
    }

    if m < period {
        return (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx);
    }

    let mut tr_s = tr[..period].iter().sum::<f64>();
    let mut pdm_s = pdm[..period].iter().sum::<f64>();
    let mut mdm_s = mdm[..period].iter().sum::<f64>();

    // Initial seeded values at index `period`
    b_pdm[period] = pdm_s;
    b_mdm[period] = mdm_s;
    if tr_s != 0.0 {
        b_pdi[period] = 100.0 * pdm_s / tr_s;
        b_mdi[period] = 100.0 * mdm_s / tr_s;
        let s = b_pdi[period] + b_mdi[period];
        b_dx[period] = if s != 0.0 {
            100.0 * (b_pdi[period] - b_mdi[period]).abs() / s
        } else {
            0.0
        };
    }

    let decay = (period - 1) as f64 / period as f64;
    for i in period..m {
        tr_s = tr_s * decay + tr[i];
        pdm_s = pdm_s * decay + pdm[i];
        mdm_s = mdm_s * decay + mdm[i];

        b_pdm[i + 1] = pdm_s;
        b_mdm[i + 1] = mdm_s;
        if tr_s != 0.0 {
            b_pdi[i + 1] = 100.0 * pdm_s / tr_s;
            b_mdi[i + 1] = 100.0 * mdm_s / tr_s;
            let s = b_pdi[i + 1] + b_mdi[i + 1];
            b_dx[i + 1] = if s != 0.0 {
                100.0 * (b_pdi[i + 1] - b_mdi[i + 1]).abs() / s
            } else {
                0.0
            };
        }
    }

    // Wilder smooth DX to get ADX
    let adx_start = period + period - 1;
    if n > adx_start {
        let mut dx_sum = 0.0;
        let mut valid_dx = true;
        for v in b_dx.iter().skip(period).take(period) {
            if v.is_nan() {
                valid_dx = false;
                break;
            }
            dx_sum += v;
        }
        if valid_dx {
            let mut adx_s = dx_sum / period as f64;
            b_adx[adx_start] = adx_s;
            let alpha = 1.0 / period as f64;
            for i in adx_start + 1..n {
                adx_s = adx_s + alpha * (b_dx[i] - adx_s);
                b_adx[i] = adx_s;
            }
        }
    }

    (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx)
}

/// Plus Directional Movement (Wilder smoothed). Output length = n (bar 0 is NaN).
pub fn plus_dm(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let closes = vec![0.0_f64; n];
    let (pdm, _, _, _, _, _) = adx_inner(high, low, &closes, timeperiod);
    pdm
}

/// Minus Directional Movement (Wilder smoothed). Output length = n (bar 0 is NaN).
pub fn minus_dm(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let closes = vec![0.0_f64; n];
    let (_, mdm, _, _, _, _) = adx_inner(high, low, &closes, timeperiod);
    mdm
}

/// Plus Directional Indicator (Wilder smoothed). Output length = n.
pub fn plus_di(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, pdi, _, _, _) = adx_inner(high, low, close, timeperiod);
    pdi
}

/// Minus Directional Indicator (Wilder smoothed). Output length = n.
pub fn minus_di(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, mdi, _, _) = adx_inner(high, low, close, timeperiod);
    mdi
}

/// Directional Movement Index: 100 * |+DI − −DI| / (+DI + −DI).
pub fn dx(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, _, dx_vals, _) = adx_inner(high, low, close, timeperiod);
    dx_vals
}

/// Average Directional Movement Index (Wilder smoothing of DX).
pub fn adx(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, _, _, adx_vals) = adx_inner(high, low, close, timeperiod);
    adx_vals
}

/// ADX Rating: (ADX[i] + ADX[i − timeperiod]) / 2.
pub fn adxr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let adx_vals = adx(high, low, close, timeperiod);
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        if !adx_vals[i].is_nan() && !adx_vals[i - timeperiod].is_nan() {
            result[i] = (adx_vals[i] + adx_vals[i - timeperiod]) / 2.0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rsi_range() {
        let prices: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let result = rsi(&prices, 14);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0 && *v <= 100.0);
        }
    }

    #[test]
    fn mom_basic() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mom(&prices, 2);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn stoch_basic() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5];
        let low = vec![9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0];
        let (slowk, slowd) = stoch(&high, &low, &close, 3, 3, 3);
        // Check that valid values are in [0, 100]
        for v in slowk.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0 && *v <= 100.0, "slowk out of range: {v}");
        }
        for v in slowd.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0 && *v <= 100.0, "slowd out of range: {v}");
        }
    }

    #[test]
    fn adx_nonnegative() {
        let h: Vec<f64> = (1..=50).map(|i| i as f64 + 1.0).collect();
        let l: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let c: Vec<f64> = (1..=50).map(|i| i as f64 + 0.5).collect();
        let result = adx(&h, &l, &c, 14);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0);
        }
    }
}
