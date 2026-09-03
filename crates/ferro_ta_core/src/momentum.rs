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

/// Fast Stochastic. Returns `(fastk, fastd)`.
///
/// Fast %K is the raw stochastic; %D is SMA(fast %K, `fastd_period`)
/// (TA-Lib default `fastd_matype=0`). Both outputs are `NaN`-padded until
/// %D is valid.
pub fn stochf(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    // slowk_period=1 leaves Fast %K unsmoothed; slowd_period applies SMA %D.
    stoch(high, low, close, fastk_period, 1, fastd_period)
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

// ---------------------------------------------------------------------------
// Rate of Change variants
// ---------------------------------------------------------------------------

/// Rate of Change: `(close[i] - close[i-p]) / close[i-p] * 100`.
pub fn roc(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 {
        return result;
    }
    for i in timeperiod..n {
        let prev = close[i - timeperiod];
        if prev != 0.0 {
            result[i] = (close[i] - prev) / prev * 100.0;
        }
    }
    result
}

/// Rate of Change Percentage: `(close[i] - close[i-p]) / close[i-p]`.
pub fn rocp(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 {
        return result;
    }
    for i in timeperiod..n {
        let prev = close[i - timeperiod];
        if prev != 0.0 {
            result[i] = (close[i] - prev) / prev;
        }
    }
    result
}

/// Rate of Change Ratio: `close[i] / close[i-p]`.
pub fn rocr(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 {
        return result;
    }
    for i in timeperiod..n {
        let prev = close[i - timeperiod];
        if prev != 0.0 {
            result[i] = close[i] / prev;
        }
    }
    result
}

/// Rate of Change Ratio x 100: `close[i] / close[i-p] * 100`.
pub fn rocr100(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 {
        return result;
    }
    for i in timeperiod..n {
        let prev = close[i - timeperiod];
        if prev != 0.0 {
            result[i] = close[i] / prev * 100.0;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Williams %R
// ---------------------------------------------------------------------------

/// Williams %R: `-100 * (HH - close) / (HH - LL)` over the window.
/// Returns values in `[-100, 0]`.
pub fn willr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }
    for i in (timeperiod - 1)..n {
        let start = i + 1 - timeperiod;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        for j in start..=i {
            if high[j] > highest {
                highest = high[j];
            }
            if low[j] < lowest {
                lowest = low[j];
            }
        }
        let range = highest - lowest;
        result[i] = if range != 0.0 {
            -100.0 * (highest - close[i]) / range
        } else {
            -50.0
        };
    }
    result
}

// ---------------------------------------------------------------------------
// Aroon
// ---------------------------------------------------------------------------

/// Aroon indicator. Returns `(aroon_down, aroon_up)`.
pub fn aroon(high: &[f64], low: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut aroon_down = vec![f64::NAN; n];
    let mut aroon_up = vec![f64::NAN; n];
    if timeperiod == 0 || n <= timeperiod {
        return (aroon_down, aroon_up);
    }
    let period_f = timeperiod as f64;
    let window_size = timeperiod + 1;
    for i in timeperiod..n {
        let start = i + 1 - window_size;
        let mut max_val = high[start];
        let mut min_val = low[start];
        let mut max_idx = 0usize;
        let mut min_idx = 0usize;
        for j in 0..window_size {
            if high[start + j] >= max_val {
                max_val = high[start + j];
                max_idx = j;
            }
            if low[start + j] <= min_val {
                min_val = low[start + j];
                min_idx = j;
            }
        }
        aroon_up[i] = 100.0 * (max_idx as f64) / period_f;
        aroon_down[i] = 100.0 * (min_idx as f64) / period_f;
    }
    (aroon_down, aroon_up)
}

/// Aroon Oscillator: `aroon_up - aroon_down`.
pub fn aroonosc(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let (down, up) = aroon(high, low, timeperiod);
    up.iter()
        .zip(down.iter())
        .map(|(&u, &d)| {
            if u.is_nan() || d.is_nan() {
                f64::NAN
            } else {
                u - d
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CCI
// ---------------------------------------------------------------------------

/// Commodity Channel Index: `(tp - SMA(tp)) / (0.015 * MAD)`.
pub fn cci(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }
    let tp: Vec<f64> = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .map(|((&h, &l), &c)| (h + l + c) / 3.0)
        .collect();
    for i in (timeperiod - 1)..n {
        let window = &tp[(i + 1 - timeperiod)..=i];
        let mean: f64 = window.iter().sum::<f64>() / timeperiod as f64;
        let mad: f64 = window.iter().map(|&x| (x - mean).abs()).sum::<f64>() / timeperiod as f64;
        result[i] = if mad != 0.0 {
            (tp[i] - mean) / (0.015 * mad)
        } else {
            0.0
        };
    }
    result
}

// ---------------------------------------------------------------------------
// BOP
// ---------------------------------------------------------------------------

/// Balance of Power: `(close - open) / (high - low)`.
pub fn bop(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    open.iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .map(|(((&o, &h), &l), &c)| {
            let range = h - l;
            if range != 0.0 {
                (c - o) / range
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stochastic RSI
// ---------------------------------------------------------------------------

/// Stochastic RSI. Returns `(fastk, fastd)`.
pub fn stochrsi(
    close: &[f64],
    timeperiod: usize,
    fastk_period: usize,
    fastd_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan_pair = || (vec![f64::NAN; n], vec![f64::NAN; n]);
    if timeperiod == 0 || fastk_period == 0 || fastd_period == 0 {
        return nan_pair();
    }

    let rsi_vals = rsi(close, timeperiod);
    let rsi_warmup = timeperiod;
    let k_warmup = rsi_warmup + fastk_period - 1;
    let d_warmup = k_warmup + fastd_period - 1;

    let mut fastk = vec![f64::NAN; n];
    let mut fastd = vec![f64::NAN; n];

    for i in k_warmup..n {
        if rsi_vals[i].is_nan() {
            continue;
        }
        let start = i + 1 - fastk_period;
        if (start..=i).any(|j| rsi_vals[j].is_nan()) {
            continue;
        }
        let mx = rsi_vals[start..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mn = rsi_vals[start..=i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        fastk[i] = if mx != mn {
            100.0 * (rsi_vals[i] - mn) / (mx - mn)
        } else {
            50.0
        };
    }

    for i in d_warmup..n {
        let start = i + 1 - fastd_period;
        let window = &fastk[start..=i];
        if window.iter().all(|v| !v.is_nan()) {
            fastd[i] = window.iter().sum::<f64>() / fastd_period as f64;
        }
    }
    (fastk, fastd)
}

// ---------------------------------------------------------------------------
// APO / PPO
// ---------------------------------------------------------------------------

/// Absolute Price Oscillator: `fast EMA - slow EMA`.
pub fn apo(close: &[f64], fastperiod: usize, slowperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if fastperiod == 0 || slowperiod == 0 || fastperiod >= slowperiod {
        return result;
    }
    let fast = crate::overlap::ema(close, fastperiod);
    let slow = crate::overlap::ema(close, slowperiod);
    let warmup = slowperiod - 1;
    for i in warmup..n {
        if !fast[i].is_nan() && !slow[i].is_nan() {
            result[i] = fast[i] - slow[i];
        }
    }
    result
}

/// Percentage Price Oscillator: `(fast EMA - slow EMA) / slow EMA * 100`.
/// Returns `(ppo_line, signal_line, histogram)`.
pub fn ppo(
    close: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan3 = || (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    if fastperiod == 0 || slowperiod == 0 || signalperiod == 0 || fastperiod >= slowperiod {
        return nan3();
    }
    let fast = crate::overlap::ema(close, fastperiod);
    let slow = crate::overlap::ema(close, slowperiod);
    let warmup = slowperiod - 1;

    let mut ppo_line = vec![f64::NAN; n];
    for i in warmup..n {
        if !fast[i].is_nan() && !slow[i].is_nan() && slow[i] != 0.0 {
            ppo_line[i] = (fast[i] - slow[i]) / slow[i] * 100.0;
        }
    }

    // Signal line = EMA of PPO line, seeded from the first finite PPO window
    // so leading NaNs do not poison the recurrence.
    let signal = crate::overlap::ema_from_first_finite(&ppo_line, signalperiod);
    let mut signal_line = vec![f64::NAN; n];
    let mut hist = vec![f64::NAN; n];
    let sig_warmup = warmup + signalperiod - 1;
    for i in sig_warmup..n {
        if !ppo_line[i].is_nan() && !signal[i].is_nan() {
            signal_line[i] = signal[i];
            hist[i] = ppo_line[i] - signal[i];
        }
    }
    (ppo_line, signal_line, hist)
}

// ---------------------------------------------------------------------------
// CMO
// ---------------------------------------------------------------------------

/// Chande Momentum Oscillator (TA-Lib / Wilder).
///
/// Same gain/loss seeding as RSI: SMA of the first `timeperiod` changes,
/// then Wilder smoothing. Output is
/// `100 * (avg_gain - avg_loss) / (avg_gain + avg_loss)`.
/// The first `timeperiod` values are `NaN`. A flat window (gain+loss = 0)
/// yields `0.0`.
pub fn cmo(close: &[f64], timeperiod: usize) -> Vec<f64> {
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
    let denom = avg_gain + avg_loss;
    result[timeperiod] = if denom != 0.0 {
        100.0 * (avg_gain - avg_loss) / denom
    } else {
        0.0
    };
    for i in (timeperiod + 1)..n {
        let diff = close[i] - close[i - 1];
        let abs_diff = diff.abs();
        let gain = (diff + abs_diff) * 0.5;
        let loss = (abs_diff - diff) * 0.5;
        avg_gain = (avg_gain * (p - 1.0) + gain) / p;
        avg_loss = (avg_loss * (p - 1.0) + loss) / p;
        let denom = avg_gain + avg_loss;
        result[i] = if denom != 0.0 {
            100.0 * (avg_gain - avg_loss) / denom
        } else {
            0.0
        };
    }
    result
}

// ---------------------------------------------------------------------------
// TRIX
// ---------------------------------------------------------------------------

/// TRIX: 1-period rate of change of triple-smoothed EMA.
pub fn trix(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 {
        return result;
    }
    let warmup = 3 * (timeperiod - 1);

    // Triple EMA: each stage seeds from the first finite window of the prior.
    let ema1 = crate::overlap::ema_from_first_finite(close, timeperiod);
    let ema2 = crate::overlap::ema_from_first_finite(&ema1, timeperiod);
    let ema3 = crate::overlap::ema_from_first_finite(&ema2, timeperiod);

    for i in (warmup + 1)..n {
        let prev = ema3[i - 1];
        if !ema3[i].is_nan() && !prev.is_nan() && prev != 0.0 {
            result[i] = (ema3[i] - prev) / prev * 100.0;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Ultimate Oscillator
// ---------------------------------------------------------------------------

/// Ultimate Oscillator: weighted average of buying pressure over three periods.
pub fn ultosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod1 == 0 || timeperiod2 == 0 || timeperiod3 == 0 || n < 2 {
        return result;
    }
    let max_period = timeperiod1.max(timeperiod2).max(timeperiod3);
    if n <= max_period {
        return result;
    }

    let mut bp = vec![0.0_f64; n];
    let mut tr = vec![0.0_f64; n];
    for i in 1..n {
        let true_low = low[i].min(close[i - 1]);
        let true_high = high[i].max(close[i - 1]);
        bp[i] = close[i] - true_low;
        tr[i] = true_high - true_low;
    }

    for i in max_period..n {
        let avg = |period: usize| -> f64 {
            let sum_bp: f64 = bp[(i + 1 - period)..=i].iter().sum();
            let sum_tr: f64 = tr[(i + 1 - period)..=i].iter().sum();
            if sum_tr != 0.0 {
                sum_bp / sum_tr
            } else {
                0.0
            }
        };
        result[i] =
            100.0 * (4.0 * avg(timeperiod1) + 2.0 * avg(timeperiod2) + avg(timeperiod3)) / 7.0;
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
    fn stochf_golden_sma_d() {
        // Hand-computed STOCHF(fastk=3, fastd=2) — TA-Lib fastd_matype=0 (SMA).
        // Both outputs are NaN-padded until %D is valid (lookback = 3-1+2-1 = 3).
        //
        // Fast %K:
        //   i=2: HH=12, LL=8, K=100*(10-8)/4 = 50
        //   i=3: HH=13, LL=8, K=100*(12-8)/5 = 80
        //   i=4: HH=14, LL=8, K=100*(13-8)/6 = 250/3
        //   i=5: HH=15, LL=10, K=100*(14-10)/5 = 80
        // SMA %D period 2:
        //   i=3: (50+80)/2 = 65
        //   i=4: (80+250/3)/2 = 245/3
        //   i=5: (250/3+80)/2 = 245/3
        let high = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0];
        let low = [8.0, 9.0, 8.0, 10.0, 11.0, 12.0];
        let close = [9.0, 11.0, 10.0, 12.0, 13.0, 14.0];
        let (fastk, fastd) = stochf(&high, &low, &close, 3, 2);
        let first_valid = 3; // fastk_period-1 + fastd_period-1
        for i in 0..first_valid {
            assert!(
                fastk[i].is_nan(),
                "expected NaN fastk at {i}, got {}",
                fastk[i]
            );
            assert!(
                fastd[i].is_nan(),
                "expected NaN fastd at {i}, got {}",
                fastd[i]
            );
        }
        let expected_k = [80.0, 250.0 / 3.0, 80.0];
        let expected_d = [65.0, 245.0 / 3.0, 245.0 / 3.0];
        for (offset, (&exp_k, &exp_d)) in expected_k.iter().zip(expected_d.iter()).enumerate() {
            let i = first_valid + offset;
            assert!(
                (fastk[i] - exp_k).abs() < 1e-10,
                "fastk[{i}]: got {} expected {exp_k}",
                fastk[i]
            );
            assert!(
                (fastd[i] - exp_d).abs() < 1e-10,
                "fastd[{i}]: got {} expected {exp_d} (must be SMA, not EMA)",
                fastd[i]
            );
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

    #[test]
    fn trix_golden_period3() {
        // Hand-computed TRIX(3) on 1..=10.
        // Triple EMA as in tema_golden_period3: EMA3 = [NaN×6, 4, 5, 6, 7]
        // TRIX[i] = (EMA3[i] - EMA3[i-1]) / EMA3[i-1] * 100
        // First value after 3*(3-1)+1 = 7: 25, 20, 100/6
        let prices: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let period = 3;
        let result = trix(&prices, period);
        let first_valid = 3 * (period - 1) + 1;
        for (i, &v) in result.iter().enumerate().take(first_valid) {
            assert!(v.is_nan(), "expected NaN warmup at {i}, got {v}");
        }
        let expected = [25.0, 20.0, 100.0 / 6.0];
        for (offset, &exp) in expected.iter().enumerate() {
            let i = first_valid + offset;
            assert!(
                result[i].is_finite(),
                "expected finite TRIX at {i}, got {}",
                result[i]
            );
            assert!(
                (result[i] - exp).abs() < 1e-10,
                "TRIX[{i}]: got {} expected {exp}",
                result[i]
            );
        }
    }

    #[test]
    fn ppo_signal_from_first_finite() {
        // PPO(2,3,2) on 1..=10. Fast/slow EMAs are SMA-seeded on raw prices
        // (no leading NaN), so the PPO line is already finite after slow-1.
        // The signal must seed from that first finite PPO window, not from
        // leading NaNs (which would poison the entire signal line).
        let prices: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let (ppo_line, signal, hist) = ppo(&prices, 2, 3, 2);
        let line_start = 3 - 1;
        let sig_start = line_start + 2 - 1;
        for (i, &v) in ppo_line.iter().enumerate().take(line_start) {
            assert!(v.is_nan(), "expected NaN PPO line at {i}, got {v}");
        }
        for (i, &v) in signal.iter().enumerate().take(sig_start) {
            assert!(v.is_nan(), "expected NaN PPO signal at {i}, got {v}");
        }
        // PPO[2] = (2.5 - 2) / 2 * 100 = 25
        // PPO[3] = (3.5 - 3) / 3 * 100 = 50/3
        // signal seed = SMA(25, 50/3) = 125/6 at index 3
        assert!((ppo_line[2] - 25.0).abs() < 1e-10);
        assert!((ppo_line[3] - 50.0 / 3.0).abs() < 1e-10);
        assert!(
            signal[sig_start].is_finite(),
            "expected finite PPO signal at {sig_start}, got {}",
            signal[sig_start]
        );
        assert!((signal[3] - 125.0 / 6.0).abs() < 1e-10);
        assert!((hist[3] - (ppo_line[3] - signal[3])).abs() < 1e-10);
        for i in sig_start..prices.len() {
            assert!(signal[i].is_finite(), "PPO signal NaN at {i}");
            assert!((hist[i] - (ppo_line[i] - signal[i])).abs() < 1e-10);
        }
    }

    #[test]
    fn cmo_wilder_golden_period3() {
        // Hand-computed CMO(3) with Wilder gain/loss (same seed as RSI).
        // close = [1, 2, 3, 2, 4, 3, 5]
        // changes: +1, +1, -1, +2, -1, +2
        //
        // Seed (first 3 changes): avg_gain=2/3, avg_loss=1/3
        // CMO[3] = 100*(2/3-1/3)/(2/3+1/3) = 100/3
        // i=4: g=10/9, l=2/9 → 200/3
        // i=5: g=20/27, l=13/27 → 700/33
        // i=6: g=94/81, l=26/81 → 170/3
        //
        // A plain rolling sum (the old bug) also yields 100/3 at index 3,
        // but 50.0 at index 4 — so 200/3 is the Wilder lock-in.
        let close = [1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0];
        let result = cmo(&close, 3);
        for (i, &v) in result.iter().enumerate().take(3) {
            assert!(v.is_nan(), "expected NaN warmup at {i}, got {v}");
        }
        let expected = [100.0 / 3.0, 200.0 / 3.0, 700.0 / 33.0, 170.0 / 3.0];
        for (offset, &exp) in expected.iter().enumerate() {
            let i = 3 + offset;
            assert!(
                result[i].is_finite(),
                "expected finite CMO at {i}, got {}",
                result[i]
            );
            assert!(
                (result[i] - exp).abs() < 1e-10,
                "CMO[{i}]: got {} expected {exp} (must be Wilder, not rolling sum)",
                result[i]
            );
        }
        // Same seed as RSI: CMO = 2*RSI - 100 on this non-flat series.
        let rsi_vals = rsi(&close, 3);
        for i in 3..close.len() {
            assert!((result[i] - (2.0 * rsi_vals[i] - 100.0)).abs() < 1e-10);
        }
    }
}
