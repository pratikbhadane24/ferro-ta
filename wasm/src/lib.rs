/*!
# ferro-ta WASM bindings

WebAssembly bindings for the ferro-ta technical analysis library.

All functions accept `Float64Array` inputs and return `Float64Array` (or a
`js_sys::Array` of `Float64Array` for multi-output indicators such as `BBANDS`
and `MACD`).

## Overlap Studies
- [`sma`] — Simple Moving Average
- [`ema`] — Exponential Moving Average
- [`bbands`] — Bollinger Bands (returns `[upper, middle, lower]`)

## Momentum Indicators
- [`rsi`] — Relative Strength Index (Wilder smoothing)
- [`macd`] — Moving Average Convergence/Divergence (returns `[macd, signal, hist]`)
- [`mom`] — Momentum (close[i] - close[i-period])
- [`stochf`] — Fast Stochastic (returns `[fastk, fastd]`)

## Volatility Indicators
- [`atr`] — Average True Range (Wilder smoothing)

## Volume Indicators
- [`obv`] — On-Balance Volume
*/

use js_sys::{Array, Float64Array};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy a `Float64Array` into a `Vec<f64>`.
fn to_vec(arr: &Float64Array) -> Vec<f64> {
    let n = arr.length() as usize;
    let mut v = vec![0.0f64; n];
    arr.copy_to(&mut v);
    v
}

/// Create a `Float64Array` from a `Vec<f64>`.
fn from_vec(v: Vec<f64>) -> Float64Array {
    // Safety: Float64Array::view requires the backing Vec to stay alive for the
    // duration of the copy.  We immediately copy via `Float64Array::from` so
    // there is no aliasing.
    let arr = Float64Array::new_with_length(v.len() as u32);
    arr.copy_from(&v);
    arr
}

// ---------------------------------------------------------------------------
// SMA — Simple Moving Average
// ---------------------------------------------------------------------------

/// Simple Moving Average.
///
/// # Arguments
/// - `close` – `Float64Array` of close prices.
/// - `timeperiod` – look-back window (default 30, minimum 1).
///
/// # Returns
/// `Float64Array` with the first `timeperiod - 1` values set to `NaN`.
#[wasm_bindgen]
pub fn sma(close: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(close);
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if timeperiod == 0 || n < timeperiod {
        return from_vec(result);
    }

    // Seed: sum of first window
    let mut window_sum: f64 = prices[..timeperiod].iter().sum();
    result[timeperiod - 1] = window_sum / timeperiod as f64;

    for i in timeperiod..n {
        window_sum += prices[i] - prices[i - timeperiod];
        result[i] = window_sum / timeperiod as f64;
    }

    from_vec(result)
}

// ---------------------------------------------------------------------------
// EMA — Exponential Moving Average
// ---------------------------------------------------------------------------

/// Exponential Moving Average (SMA-seeded).
///
/// # Arguments
/// - `close` – `Float64Array` of close prices.
/// - `timeperiod` – look-back period (default 30, minimum 1).
///
/// # Returns
/// `Float64Array` with the first `timeperiod - 1` values set to `NaN`.
#[wasm_bindgen]
pub fn ema(close: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(close);
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if timeperiod == 0 || n < timeperiod {
        return from_vec(result);
    }

    let k = 2.0 / (timeperiod as f64 + 1.0);

    // Seed with SMA of first window
    let seed: f64 = prices[..timeperiod].iter().sum::<f64>() / timeperiod as f64;
    result[timeperiod - 1] = seed;
    let mut prev = seed;

    for i in timeperiod..n {
        let val = prices[i] * k + prev * (1.0 - k);
        result[i] = val;
        prev = val;
    }

    from_vec(result)
}

// ---------------------------------------------------------------------------
// BBANDS — Bollinger Bands
// ---------------------------------------------------------------------------

/// Bollinger Bands (SMA ± k × rolling standard deviation).
///
/// # Arguments
/// - `close` – `Float64Array` of close prices.
/// - `timeperiod` – look-back window (default 5, minimum 1).
/// - `nbdevup` – multiplier for the upper band (default 2.0).
/// - `nbdevdn` – multiplier for the lower band (default 2.0).
///
/// # Returns
/// A `js_sys::Array` containing three `Float64Array` elements:
/// `[upperband, middleband, lowerband]`.
#[wasm_bindgen]
pub fn bbands(
    close: &Float64Array,
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
) -> Array {
    let prices = to_vec(close);
    let n = prices.len();
    let mut upper = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];

    if timeperiod == 0 || n < timeperiod {
        let out = Array::new();
        out.push(&from_vec(upper));
        out.push(&from_vec(middle));
        out.push(&from_vec(lower));
        return out;
    }

    for i in (timeperiod - 1)..n {
        let window = &prices[(i + 1 - timeperiod)..=i];
        let mean = window.iter().sum::<f64>() / timeperiod as f64;
        let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / timeperiod as f64;
        let stddev = variance.sqrt();
        middle[i] = mean;
        upper[i] = mean + nbdevup * stddev;
        lower[i] = mean - nbdevdn * stddev;
    }

    let out = Array::new();
    out.push(&from_vec(upper));
    out.push(&from_vec(middle));
    out.push(&from_vec(lower));
    out
}

// ---------------------------------------------------------------------------
// RSI — Relative Strength Index (Wilder smoothing, TA-Lib compatible)
// ---------------------------------------------------------------------------

/// Relative Strength Index (Wilder smoothing).
///
/// # Arguments
/// - `close` – `Float64Array` of close prices.
/// - `timeperiod` – look-back period (default 14, minimum 1).
///
/// # Returns
/// `Float64Array` — values in `[0, 100]`; first `timeperiod` values are `NaN`.
#[wasm_bindgen]
pub fn rsi(close: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(close);
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if timeperiod == 0 || n <= timeperiod {
        return from_vec(result);
    }

    // Compute gains and losses
    let diffs: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

    // Seed average gain / loss over first `timeperiod` bars
    let mut avg_gain: f64 = diffs[..timeperiod]
        .iter()
        .map(|&d| if d > 0.0 { d } else { 0.0 })
        .sum::<f64>()
        / timeperiod as f64;
    let mut avg_loss: f64 = diffs[..timeperiod]
        .iter()
        .map(|&d| if d < 0.0 { -d } else { 0.0 })
        .sum::<f64>()
        / timeperiod as f64;

    // First RSI value at index `timeperiod`
    let rs = if avg_loss == 0.0 { f64::INFINITY } else { avg_gain / avg_loss };
    result[timeperiod] = 100.0 - 100.0 / (1.0 + rs);

    // Wilder smoothing for remaining values
    for i in (timeperiod + 1)..n {
        let diff = diffs[i - 1];
        let gain = if diff > 0.0 { diff } else { 0.0 };
        let loss = if diff < 0.0 { -diff } else { 0.0 };
        avg_gain = (avg_gain * (timeperiod as f64 - 1.0) + gain) / timeperiod as f64;
        avg_loss = (avg_loss * (timeperiod as f64 - 1.0) + loss) / timeperiod as f64;
        let rs = if avg_loss == 0.0 { f64::INFINITY } else { avg_gain / avg_loss };
        result[i] = 100.0 - 100.0 / (1.0 + rs);
    }

    from_vec(result)
}

// ---------------------------------------------------------------------------
// ATR — Average True Range (Wilder smoothing)
// ---------------------------------------------------------------------------

/// Average True Range (Wilder smoothing, TA-Lib compatible).
///
/// # Arguments
/// - `high`  – `Float64Array` of high prices.
/// - `low`   – `Float64Array` of low prices.
/// - `close` – `Float64Array` of close prices.
/// - `timeperiod` – look-back period (default 14, minimum 1).
///
/// # Returns
/// `Float64Array`; first `timeperiod` values are `NaN`.
#[wasm_bindgen]
pub fn atr(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    timeperiod: usize,
) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let n = h.len();
    let mut result = vec![f64::NAN; n];

    if timeperiod == 0 || n <= timeperiod {
        return from_vec(result);
    }
    if l.len() != n || c.len() != n {
        return from_vec(result);
    }

    // True Range for each bar
    let mut tr = vec![0.0f64; n];
    tr[0] = h[0] - l[0]; // first bar: no previous close
    for i in 1..n {
        let hl = h[i] - l[i];
        let hpc = (h[i] - c[i - 1]).abs();
        let lpc = (l[i] - c[i - 1]).abs();
        tr[i] = hl.max(hpc).max(lpc);
    }

    // Seed: SMA of first `timeperiod` true ranges
    let seed: f64 = tr[1..=timeperiod].iter().sum::<f64>() / timeperiod as f64;
    result[timeperiod] = seed;
    let mut prev = seed;

    for i in (timeperiod + 1)..n {
        let val = (prev * (timeperiod as f64 - 1.0) + tr[i]) / timeperiod as f64;
        result[i] = val;
        prev = val;
    }

    from_vec(result)
}

// ---------------------------------------------------------------------------
// OBV — On-Balance Volume
// ---------------------------------------------------------------------------

/// On-Balance Volume.
///
/// # Arguments
/// - `close`  – `Float64Array` of close prices.
/// - `volume` – `Float64Array` of volume values.
///
/// # Returns
/// `Float64Array` — cumulative OBV.
#[wasm_bindgen]
pub fn obv(close: &Float64Array, volume: &Float64Array) -> Float64Array {
    let c = to_vec(close);
    let v = to_vec(volume);
    let n = c.len();
    let mut result = vec![0.0f64; n];

    if n == 0 || v.len() != n {
        return from_vec(result);
    }

    result[0] = v[0];
    for i in 1..n {
        if c[i] > c[i - 1] {
            result[i] = result[i - 1] + v[i];
        } else if c[i] < c[i - 1] {
            result[i] = result[i - 1] - v[i];
        } else {
            result[i] = result[i - 1];
        }
    }

    from_vec(result)
}

// ---------------------------------------------------------------------------
// MOM — Momentum
// ---------------------------------------------------------------------------

/// Momentum — difference between current close and close *timeperiod* bars ago.
///
/// # Arguments
/// - `close`      – `Float64Array` of close prices.
/// - `timeperiod` – look-back window (default 10, minimum 1).
///
/// # Returns
/// `Float64Array`; first `timeperiod` values are `NaN`.
#[wasm_bindgen]
pub fn mom(close: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(close);
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if timeperiod == 0 || n <= timeperiod {
        return from_vec(result);
    }

    for i in timeperiod..n {
        result[i] = prices[i] - prices[i - timeperiod];
    }

    from_vec(result)
}

// ---------------------------------------------------------------------------
// STOCHF — Fast Stochastic Oscillator
// ---------------------------------------------------------------------------

/// Fast Stochastic Oscillator.
///
/// # Arguments
/// - `high`        – `Float64Array` of high prices.
/// - `low`         – `Float64Array` of low prices.
/// - `close`       – `Float64Array` of close prices.
/// - `fastk_period` – fast-%K look-back window (default 5, minimum 1).
/// - `fastd_period` – fast-%D SMA smoothing period (default 3, minimum 1).
///
/// # Returns
/// A `js_sys::Array` containing two `Float64Array` elements: `[fastk, fastd]`.
#[wasm_bindgen]
pub fn stochf(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    fastk_period: usize,
    fastd_period: usize,
) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let n = c.len();

    let nan_out = || {
        let out = Array::new();
        out.push(&from_vec(vec![f64::NAN; n]));
        out.push(&from_vec(vec![f64::NAN; n]));
        out
    };

    if fastk_period == 0 || fastd_period == 0 || n < fastk_period {
        return nan_out();
    }
    if h.len() != n || l.len() != n {
        return nan_out();
    }

    // Fast %K: (close - lowest_low) / (highest_high - lowest_low) * 100
    let mut fastk = vec![f64::NAN; n];
    for i in (fastk_period - 1)..n {
        let low_min = l[(i + 1 - fastk_period)..=i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let high_max = h[(i + 1 - fastk_period)..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = high_max - low_min;
        fastk[i] = if range > 0.0 {
            (c[i] - low_min) / range * 100.0
        } else {
            50.0 // all bars at same price — neutral
        };
    }

    // Fast %D: SMA(fastd_period) of fast %K
    let mut fastd = vec![f64::NAN; n];
    let k_start = fastk_period - 1;
    if n >= k_start + fastd_period {
        for i in (k_start + fastd_period - 1)..n {
            let window = &fastk[(i + 1 - fastd_period)..=i];
            if window.iter().all(|x| x.is_finite()) {
                fastd[i] = window.iter().sum::<f64>() / fastd_period as f64;
            }
        }
    }

    let out = Array::new();
    out.push(&from_vec(fastk));
    out.push(&from_vec(fastd));
    out
}

// ---------------------------------------------------------------------------
// MACD — Moving Average Convergence/Divergence
// ---------------------------------------------------------------------------

/// Moving Average Convergence/Divergence.
///
/// # Arguments
/// - `close`        – `Float64Array` of close prices.
/// - `fastperiod`   – fast EMA period (default 12).
/// - `slowperiod`   – slow EMA period (default 26).
/// - `signalperiod` – signal EMA period (default 9).
///
/// # Returns
/// A `js_sys::Array` containing three `Float64Array` elements:
/// `[macd_line, signal_line, histogram]`.
#[wasm_bindgen]
pub fn macd(
    close: &Float64Array,
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> Array {
    let prices = to_vec(close);
    let n = prices.len();
    let nan_result = || {
        let out = Array::new();
        out.push(&from_vec(vec![f64::NAN; n]));
        out.push(&from_vec(vec![f64::NAN; n]));
        out.push(&from_vec(vec![f64::NAN; n]));
        out
    };

    if fastperiod == 0 || slowperiod == 0 || signalperiod == 0 || fastperiod >= slowperiod {
        return nan_result();
    }
    if n < slowperiod {
        return nan_result();
    }

    // Helper: SMA-seeded EMA
    let ema_vec = |data: &[f64], period: usize| -> Vec<f64> {
        let len = data.len();
        let mut result = vec![f64::NAN; len];
        if period == 0 || len < period {
            return result;
        }
        let k = 2.0 / (period as f64 + 1.0);
        let seed: f64 = data[..period].iter().sum::<f64>() / period as f64;
        result[period - 1] = seed;
        for i in period..len {
            result[i] = data[i] * k + result[i - 1] * (1.0 - k);
        }
        result
    };

    let fast_ema = ema_vec(&prices, fastperiod);
    let slow_ema = ema_vec(&prices, slowperiod);

    // MACD line = fast EMA − slow EMA (valid from index slowperiod - 1)
    let mut macd_line = vec![f64::NAN; n];
    for i in (slowperiod - 1)..n {
        if fast_ema[i].is_finite() && slow_ema[i].is_finite() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }

    // Signal line = EMA(signalperiod) of macd_line, seeded at index slowperiod - 1
    let macd_start = slowperiod - 1;
    let mut signal_line = vec![f64::NAN; n];
    let signal_seed_end = macd_start + signalperiod;
    if signal_seed_end > n {
        let out = Array::new();
        out.push(&from_vec(macd_line.clone()));
        out.push(&from_vec(signal_line));
        out.push(&from_vec(vec![f64::NAN; n]));
        return out;
    }

    // Seed: SMA of first signalperiod MACD values
    let seed: f64 = macd_line[macd_start..signal_seed_end]
        .iter()
        .sum::<f64>()
        / signalperiod as f64;
    signal_line[signal_seed_end - 1] = seed;
    let k = 2.0 / (signalperiod as f64 + 1.0);
    for i in signal_seed_end..n {
        if macd_line[i].is_finite() {
            signal_line[i] = macd_line[i] * k + signal_line[i - 1] * (1.0 - k);
        }
    }

    // Histogram = MACD − signal
    let mut histogram = vec![f64::NAN; n];
    for i in (signal_seed_end - 1)..n {
        if macd_line[i].is_finite() && signal_line[i].is_finite() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    let out = Array::new();
    out.push(&from_vec(macd_line));
    out.push(&from_vec(signal_line));
    out.push(&from_vec(histogram));
    out
}

// ---------------------------------------------------------------------------
// WASM tests (run with `wasm-pack test --node`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    fn make_arr(v: &[f64]) -> Float64Array {
        let arr = Float64Array::new_with_length(v.len() as u32);
        arr.copy_from(v);
        arr
    }

    fn get_finite(arr: &Float64Array) -> Vec<f64> {
        let mut v = vec![0.0f64; arr.length() as usize];
        arr.copy_to(&mut v);
        v.into_iter().filter(|x| x.is_finite()).collect()
    }

    // -----------------------------------------------------------------------
    // SMA tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_sma_output_length() {
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = sma(&close, 3);
        assert_eq!(out.length(), 5);
    }

    #[wasm_bindgen_test]
    fn test_sma_known_value() {
        // SMA(3) of [1,2,3,4,5]: first valid at index 2 = (1+2+3)/3 = 2.0
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = sma(&close, 3);
        let vals: Vec<f64> = {
            let mut v = vec![0.0f64; 5];
            out.copy_to(&mut v);
            v
        };
        assert!(vals[0].is_nan());
        assert!(vals[1].is_nan());
        assert!((vals[2] - 2.0).abs() < 1e-10);
        assert!((vals[3] - 3.0).abs() < 1e-10);
        assert!((vals[4] - 4.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // EMA tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_ema_output_length() {
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = ema(&close, 3);
        assert_eq!(out.length(), 5);
    }

    #[wasm_bindgen_test]
    fn test_ema_seed_equals_sma() {
        // Seed of EMA(3) at index 2 should equal SMA(3) = 2.0
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = ema(&close, 3);
        let mut vals = vec![0.0f64; 5];
        out.copy_to(&mut vals);
        assert!((vals[2] - 2.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // BBANDS tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_bbands_returns_three_arrays() {
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = bbands(&close, 3, 2.0, 2.0);
        assert_eq!(out.length(), 3);
    }

    #[wasm_bindgen_test]
    fn test_bbands_middle_equals_sma() {
        let data = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10];
        let close = make_arr(&data);
        let bands = bbands(&close, 3, 2.0, 2.0);

        // Middle band should equal SMA(3)
        let middle = Float64Array::from(bands.get(1));
        let sma_out = sma(&close, 3);

        let mut m = vec![0.0f64; 7];
        middle.copy_to(&mut m);
        let mut s = vec![0.0f64; 7];
        sma_out.copy_to(&mut s);

        for i in 2..7 {
            assert!((m[i] - s[i]).abs() < 1e-10, "middle[{i}] != sma[{i}]");
        }
    }

    #[wasm_bindgen_test]
    fn test_bbands_upper_greater_than_lower() {
        let data = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10];
        let close = make_arr(&data);
        let bands = bbands(&close, 3, 2.0, 2.0);
        let upper = Float64Array::from(bands.get(0));
        let lower = Float64Array::from(bands.get(2));
        let mut u = vec![0.0f64; 7];
        let mut l = vec![0.0f64; 7];
        upper.copy_to(&mut u);
        lower.copy_to(&mut l);
        for i in 2..7 {
            assert!(u[i] >= l[i], "upper[{i}] < lower[{i}]");
        }
    }

    // -----------------------------------------------------------------------
    // RSI tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_rsi_output_length() {
        let close = make_arr(&[
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
            43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33,
        ]);
        let out = rsi(&close, 14);
        assert_eq!(out.length(), 15);
    }

    #[wasm_bindgen_test]
    fn test_rsi_range_0_to_100() {
        let close = make_arr(&[
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
            43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33,
        ]);
        let out = rsi(&close, 5);
        let finite = get_finite(&out);
        for v in finite {
            assert!(v >= 0.0 && v <= 100.0, "RSI out of range: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // ATR tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_atr_output_length() {
        let high  = make_arr(&[45.0, 46.0, 47.0, 46.0, 45.0, 44.0, 45.0]);
        let low   = make_arr(&[43.0, 44.0, 45.0, 44.0, 43.0, 42.0, 43.0]);
        let close = make_arr(&[44.0, 45.0, 46.0, 45.0, 44.0, 43.0, 44.0]);
        let out = atr(&high, &low, &close, 3);
        assert_eq!(out.length(), 7);
    }

    #[wasm_bindgen_test]
    fn test_atr_all_positive() {
        let high  = make_arr(&[45.0, 46.0, 47.0, 46.0, 45.0, 44.0, 45.0]);
        let low   = make_arr(&[43.0, 44.0, 45.0, 44.0, 43.0, 42.0, 43.0]);
        let close = make_arr(&[44.0, 45.0, 46.0, 45.0, 44.0, 43.0, 44.0]);
        let out = atr(&high, &low, &close, 3);
        let finite = get_finite(&out);
        assert!(!finite.is_empty());
        for v in finite {
            assert!(v > 0.0, "ATR should be positive, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // OBV tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_obv_output_length() {
        let close  = make_arr(&[10.0, 11.0, 10.0, 12.0, 11.0]);
        let volume = make_arr(&[100.0, 200.0, 150.0, 300.0, 250.0]);
        let out = obv(&close, &volume);
        assert_eq!(out.length(), 5);
    }

    #[wasm_bindgen_test]
    fn test_obv_known_values() {
        // close: 10 → 11 (up, +200) → 10 (dn, -150) → 12 (up, +300) → 11 (dn, -250)
        // OBV:   100, 300, 150, 450, 200
        let close  = make_arr(&[10.0, 11.0, 10.0, 12.0, 11.0]);
        let volume = make_arr(&[100.0, 200.0, 150.0, 300.0, 250.0]);
        let out = obv(&close, &volume);
        let mut vals = vec![0.0f64; 5];
        out.copy_to(&mut vals);
        assert!((vals[0] - 100.0).abs() < 1e-10);
        assert!((vals[1] - 300.0).abs() < 1e-10);
        assert!((vals[2] - 150.0).abs() < 1e-10);
        assert!((vals[3] - 450.0).abs() < 1e-10);
        assert!((vals[4] - 200.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // MACD tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_macd_returns_three_arrays() {
        let data = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
            43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33,
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
            43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33,
        ];
        let close = make_arr(&data);
        let out = macd(&close, 3, 5, 2);
        assert_eq!(out.length(), 3);
    }

    #[wasm_bindgen_test]
    fn test_macd_output_length() {
        let data: Vec<f64> = (1..=30).map(|x| x as f64 * 1.0).collect();
        let close = make_arr(&data);
        let out = macd(&close, 3, 5, 2);
        let macd_line = Float64Array::from(out.get(0));
        assert_eq!(macd_line.length(), 30);
    }

    #[wasm_bindgen_test]
    fn test_macd_finite_values_after_warmup() {
        // With fastperiod=3, slowperiod=5, signalperiod=2:
        // MACD line valid from index 4; signal from index 5.
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let close = make_arr(&data);
        let out = macd(&close, 3, 5, 2);
        let signal = Float64Array::from(out.get(1));
        let finite = get_finite(&signal);
        assert!(!finite.is_empty(), "signal should have finite values");
    }

    #[wasm_bindgen_test]
    fn test_macd_histogram_is_macd_minus_signal() {
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let close = make_arr(&data);
        let out = macd(&close, 3, 5, 2);
        let macd_arr = Float64Array::from(out.get(0));
        let sig_arr  = Float64Array::from(out.get(1));
        let hist_arr = Float64Array::from(out.get(2));

        let n = macd_arr.length() as usize;
        let mut m = vec![0.0f64; n];
        let mut s = vec![0.0f64; n];
        let mut h = vec![0.0f64; n];
        macd_arr.copy_to(&mut m);
        sig_arr.copy_to(&mut s);
        hist_arr.copy_to(&mut h);

        for i in 0..n {
            if m[i].is_finite() && s[i].is_finite() {
                assert!((h[i] - (m[i] - s[i])).abs() < 1e-10,
                    "histogram[{i}] != macd[{i}] - signal[{i}]");
            }
        }
    }

    // -----------------------------------------------------------------------
    // MOM tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_mom_output_length() {
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let out = mom(&close, 3);
        assert_eq!(out.length(), 7);
    }

    #[wasm_bindgen_test]
    fn test_mom_known_values() {
        // MOM(2) of [1,2,3,4,5]: NaN, NaN, 2.0, 2.0, 2.0
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = mom(&close, 2);
        let mut vals = vec![0.0f64; 5];
        out.copy_to(&mut vals);
        assert!(vals[0].is_nan());
        assert!(vals[1].is_nan());
        assert!((vals[2] - 2.0).abs() < 1e-10, "MOM[2] should be 2.0");
        assert!((vals[3] - 2.0).abs() < 1e-10, "MOM[3] should be 2.0");
        assert!((vals[4] - 2.0).abs() < 1e-10, "MOM[4] should be 2.0");
    }

    // -----------------------------------------------------------------------
    // STOCHF tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_stochf_returns_two_arrays() {
        let h = make_arr(&[10.0, 11.0, 12.0, 11.0, 10.0, 12.0, 13.0]);
        let l = make_arr(&[8.0,  9.0, 10.0,  9.0,  8.0, 10.0, 11.0]);
        let c = make_arr(&[9.0, 10.0, 11.0, 10.0,  9.0, 11.0, 12.0]);
        let out = stochf(&h, &l, &c, 3, 2);
        assert_eq!(out.length(), 2);
    }

    #[wasm_bindgen_test]
    fn test_stochf_output_length() {
        let h = make_arr(&[10.0, 11.0, 12.0, 11.0, 10.0, 12.0, 13.0]);
        let l = make_arr(&[8.0,  9.0, 10.0,  9.0,  8.0, 10.0, 11.0]);
        let c = make_arr(&[9.0, 10.0, 11.0, 10.0,  9.0, 11.0, 12.0]);
        let out = stochf(&h, &l, &c, 3, 2);
        let fastk = Float64Array::from(out.get(0));
        assert_eq!(fastk.length(), 7);
    }

    #[wasm_bindgen_test]
    fn test_stochf_fastk_in_0_to_100() {
        let h = make_arr(&[10.0, 11.0, 12.0, 11.0, 10.0, 12.0, 13.0]);
        let l = make_arr(&[8.0,  9.0, 10.0,  9.0,  8.0, 10.0, 11.0]);
        let c = make_arr(&[9.0, 10.0, 11.0, 10.0,  9.0, 11.0, 12.0]);
        let out = stochf(&h, &l, &c, 3, 2);
        let fastk = Float64Array::from(out.get(0));
        let finite = get_finite(&fastk);
        assert!(!finite.is_empty(), "fastk should have finite values");
        for v in finite {
            assert!(v >= 0.0 && v <= 100.0, "fastk value {v} out of [0, 100]");
        }
    }
}
