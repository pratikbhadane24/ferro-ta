//! Momentum indicators.

/// Compute the Relative Strength Index (RSI).
///
/// Returns values in the range `[0, 100]`. Uses Wilder's smoothing method
/// (TA-Lib compatible), seeding avg_gain/avg_loss with the SMA of the first
/// `timeperiod` price changes. The first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Lookback period (typically 14).
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

/// Compute the Momentum indicator: `close[i] - close[i - timeperiod]`.
///
/// Returns a `Vec<f64>` of length `n`. The first `timeperiod` values are `NaN`.
/// Positive values indicate upward price movement over the lookback window.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Number of bars to look back (must be >= 1).
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

/// Compute the Stochastic Oscillator (TA-Lib compatible).
///
/// Returns `(slow_k, slow_d)`, both in the range `[0, 100]`.
///  - Fast %K = 100 * (close - lowest low) / (highest high - lowest low)
///  - Slow %K = SMA(fast %K, `slowk_period`)
///  - Slow %D = SMA(slow %K, `slowd_period`)
///
/// Uses O(n) sliding max/min via monotonic deques. Both outputs are
/// `NaN`-padded until slow %D becomes valid (TA-Lib convention).
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `fastk_period` - Lookback for highest high / lowest low.
/// * `slowk_period` - SMA period applied to fast %K.
/// * `slowd_period` - SMA period applied to slow %K.
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

    let mut slowk = vec![f64::NAN; n];
    let mut slowd = vec![f64::NAN; n];

    // Fused pass: compute fast %K inline with sliding max/min.
    // For typical small windows (5-14), inline scan beats VecDeque overhead.
    let fastk_start = fastk_period - 1;
    let fk_len = n - fastk_start;
    let mut fastk_valid = vec![0.0_f64; fk_len];

    for i in fastk_start..n {
        // Inline sliding max(high) and min(low) over [i - fastk_period + 1 .. i].
        let win_start = i + 1 - fastk_period;
        let mut hh = high[win_start];
        let mut ll = low[win_start];
        for j in (win_start + 1)..=i {
            let h = high[j];
            let l = low[j];
            if h > hh {
                hh = h;
            }
            if l < ll {
                ll = l;
            }
        }
        let range = hh - ll;
        fastk_valid[i - fastk_start] = if range != 0.0 {
            100.0 * (close[i] - ll) / range
        } else {
            0.0
        };
    }

    // Slow %K = SMA(fastk_valid, slowk_period).
    crate::overlap::sma_into(&fastk_valid, slowk_period, &mut slowk, fastk_start);

    // Slow %D = SMA(slowk, slowd_period).
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

/// Compute all six ADX-family outputs in a single pass.
///
/// Returns `(plus_dm, minus_dm, plus_di, minus_di, dx, adx)`.
/// Use this when you need multiple ADX-family outputs to avoid redundant
/// computation. All values are in `[0, 100]` except DM which is unbounded.
/// Warmup: DI/DX valid from index `timeperiod`; ADX from `2 * timeperiod - 1`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period (typically 14).
pub fn adx_all(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> AdxInnerOutput {
    adx_inner(high, low, close, timeperiod)
}

/// Internal helper for plus_dm and minus_dm that doesn't allocate dummy close prices.
/// Returns (plus_dm, minus_dm) smoothed with Wilder's method.
fn dm_only_inner(high: &[f64], low: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut b_pdm = vec![f64::NAN; n];
    let mut b_mdm = vec![f64::NAN; n];

    if n < period || period < 1 || n < 2 {
        return (b_pdm, b_mdm);
    }

    let m = n - 1;
    let mut pdm = vec![0.0_f64; m];
    let mut mdm = vec![0.0_f64; m];

    for i in 0..m {
        let j = i + 1;
        let h_diff = high[j] - high[i];
        let l_diff = low[i] - low[j];
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
        return (b_pdm, b_mdm);
    }

    let mut pdm_s = pdm[..period].iter().sum::<f64>();
    let mut mdm_s = mdm[..period].iter().sum::<f64>();

    b_pdm[period] = pdm_s;
    b_mdm[period] = mdm_s;

    let decay = (period - 1) as f64 / period as f64;
    for i in period..m {
        pdm_s = pdm_s * decay + pdm[i];
        mdm_s = mdm_s * decay + mdm[i];
        b_pdm[i + 1] = pdm_s;
        b_mdm[i + 1] = mdm_s;
    }

    (b_pdm, b_mdm)
}

/// Compute the Plus Directional Movement (+DM), Wilder smoothed.
///
/// Measures upward price movement. Returns a `Vec<f64>` of length `n`;
/// the first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` - High and low price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn plus_dm(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let (pdm, _) = dm_only_inner(high, low, timeperiod);
    pdm
}

/// Compute the Minus Directional Movement (-DM), Wilder smoothed.
///
/// Measures downward price movement. Returns a `Vec<f64>` of length `n`;
/// the first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` - High and low price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn minus_dm(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, mdm) = dm_only_inner(high, low, timeperiod);
    mdm
}

/// Compute the Plus Directional Indicator (+DI), Wilder smoothed.
///
/// `+DI = 100 * smoothed(+DM) / smoothed(TR)`. Returns values in `[0, 100]`.
/// The first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn plus_di(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, pdi, _, _, _) = adx_inner(high, low, close, timeperiod);
    pdi
}

/// Compute the Minus Directional Indicator (-DI), Wilder smoothed.
///
/// `-DI = 100 * smoothed(-DM) / smoothed(TR)`. Returns values in `[0, 100]`.
/// The first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn minus_di(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, mdi, _, _) = adx_inner(high, low, close, timeperiod);
    mdi
}

/// Compute the Directional Movement Index (DX).
///
/// `DX = 100 * |+DI - -DI| / (+DI + -DI)`. Returns values in `[0, 100]`.
/// The first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn dx(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, _, dx_vals, _) = adx_inner(high, low, close, timeperiod);
    dx_vals
}

/// Compute the Average Directional Movement Index (ADX).
///
/// ADX is Wilder's smoothing of DX, measuring trend strength regardless of
/// direction. Returns values in `[0, 100]`. The first `2 * timeperiod - 1`
/// values are `NaN` (DX warmup + ADX smoothing warmup).
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period (typically 14).
pub fn adx(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, _, _, adx_vals) = adx_inner(high, low, close, timeperiod);
    adx_vals
}

/// Compute the ADX Rating (ADXR).
///
/// `ADXR[i] = (ADX[i] + ADX[i - timeperiod]) / 2`. Smooths ADX further
/// by averaging current ADX with its value `timeperiod` bars ago.
/// Returns values in `[0, 100]`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period (typically 14).
pub fn adxr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    // Reuse adx_all to compute ADX once, then derive ADXR from it
    let (_, _, _, _, _, adx_vals) = adx_inner(high, low, close, timeperiod);
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
