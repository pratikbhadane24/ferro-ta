/*!
# ferro-ta WASM bindings

WebAssembly bindings for the ferro-ta technical analysis library.

All functions accept `Float64Array` inputs and return `Float64Array` (or a
`js_sys::Array` of `Float64Array` for multi-output indicators such as `BBANDS`
and `MACD`).

## Overlap Studies
- [`sma`] — Simple Moving Average
- [`ema`] — Exponential Moving Average
- [`wma`] — Weighted Moving Average
- [`bbands`] — Bollinger Bands (returns `[upper, middle, lower]`)

## Momentum Indicators
- [`rsi`] — Relative Strength Index (Wilder smoothing)
- [`macd`] — Moving Average Convergence/Divergence (returns `[macd, signal, hist]`)
- [`mom`] — Momentum (close[i] - close[i-period])
- [`stochf`] — Fast Stochastic (returns `[fastk, fastd]`)
- [`adx`] — Average Directional Movement Index

## Volatility Indicators
- [`atr`] — Average True Range (Wilder smoothing)

## Volume Indicators
- [`obv`] — On-Balance Volume
- [`mfi`] — Money Flow Index
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
    from_vec(ferro_ta_core::overlap::sma(&prices, timeperiod))
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
    from_vec(ferro_ta_core::overlap::ema(&prices, timeperiod))
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
    let (upper, middle, lower) = ferro_ta_core::overlap::bbands(&prices, timeperiod, nbdevup, nbdevdn);
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
    from_vec(ferro_ta_core::momentum::rsi(&prices, timeperiod))
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
    from_vec(ferro_ta_core::volatility::atr(&h, &l, &c, timeperiod))
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
    from_vec(ferro_ta_core::volume::obv(&c, &v))
}

// ---------------------------------------------------------------------------
// WMA — Weighted Moving Average
// ---------------------------------------------------------------------------

/// Weighted Moving Average.
///
/// # Arguments
/// - `close` – `Float64Array` of close prices.
/// - `timeperiod` – look-back window (default 30, minimum 1).
///
/// # Returns
/// `Float64Array` with the first `timeperiod - 1` values set to `NaN`.
#[wasm_bindgen]
pub fn wma(close: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(close);
    from_vec(ferro_ta_core::overlap::wma(&prices, timeperiod))
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
    from_vec(ferro_ta_core::momentum::mom(&prices, timeperiod))
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
    // stoch with slowk_period=1 yields fastk as slowk, fastd as slowd
    let (fastk, fastd) = ferro_ta_core::momentum::stoch(&h, &l, &c, fastk_period, 1, fastd_period);
    let out = Array::new();
    out.push(&from_vec(fastk));
    out.push(&from_vec(fastd));
    out
}

// ---------------------------------------------------------------------------
// ADX — Average Directional Movement Index
// ---------------------------------------------------------------------------

/// Average Directional Movement Index (Wilder smoothing).
///
/// # Arguments
/// - `high`       – `Float64Array` of high prices.
/// - `low`        – `Float64Array` of low prices.
/// - `close`      – `Float64Array` of close prices.
/// - `timeperiod` – look-back period (default 14, minimum 1).
///
/// # Returns
/// `Float64Array`; warm-up values are `NaN`.
#[wasm_bindgen]
pub fn adx(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    timeperiod: usize,
) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    if h.len() != l.len() || h.len() != c.len() {
        return from_vec(vec![f64::NAN; c.len()]);
    }
    from_vec(ferro_ta_core::momentum::adx(&h, &l, &c, timeperiod))
}

// ---------------------------------------------------------------------------
// MFI — Money Flow Index
// ---------------------------------------------------------------------------

/// Money Flow Index.
///
/// # Arguments
/// - `high`       – `Float64Array` of high prices.
/// - `low`        – `Float64Array` of low prices.
/// - `close`      – `Float64Array` of close prices.
/// - `volume`     – `Float64Array` of volume values.
/// - `timeperiod` – look-back period (default 14, minimum 1).
///
/// # Returns
/// `Float64Array`; warm-up values are `NaN`.
#[wasm_bindgen]
pub fn mfi(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    volume: &Float64Array,
    timeperiod: usize,
) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let v = to_vec(volume);
    let n = c.len();
    if h.len() != n || l.len() != n || v.len() != n {
        return from_vec(vec![f64::NAN; n]);
    }
    from_vec(ferro_ta_core::volume::mfi(&h, &l, &c, &v, timeperiod))
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
    let (macd_line, signal_line, histogram) =
        ferro_ta_core::overlap::macd(&prices, fastperiod, slowperiod, signalperiod);
    let out = Array::new();
    out.push(&from_vec(macd_line));
    out.push(&from_vec(signal_line));
    out.push(&from_vec(histogram));
    out
}

// ---------------------------------------------------------------------------
// CommissionModel — advanced commission and tax model for Indian and global markets
// ---------------------------------------------------------------------------

/// Advanced commission and tax model (WASM binding).
///
/// All `_rate` fields are fractions (e.g. 0.001 = 0.1%).
/// Per-unit fields (`flat_per_order`, `per_lot`) are in base currency units (e.g. INR).
///
/// Use the static factory methods for built-in presets, or construct and
/// set fields individually.
#[wasm_bindgen]
pub struct CommissionModel {
    inner: ferro_ta_core::commission::CommissionModel,
}

#[wasm_bindgen]
impl CommissionModel {
    /// Create a zero-commission model.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { inner: ferro_ta_core::commission::CommissionModel::default() }
    }

    // ---- Field getters/setters ------------------------------------------

    #[wasm_bindgen(getter)] pub fn flat_per_order(&self) -> f64 { self.inner.flat_per_order }
    #[wasm_bindgen(setter)] pub fn set_flat_per_order(&mut self, v: f64) { self.inner.flat_per_order = v; }

    #[wasm_bindgen(getter)] pub fn rate_of_value(&self) -> f64 { self.inner.rate_of_value }
    #[wasm_bindgen(setter)] pub fn set_rate_of_value(&mut self, v: f64) { self.inner.rate_of_value = v; }

    #[wasm_bindgen(getter)] pub fn per_lot(&self) -> f64 { self.inner.per_lot }
    #[wasm_bindgen(setter)] pub fn set_per_lot(&mut self, v: f64) { self.inner.per_lot = v; }

    #[wasm_bindgen(getter)] pub fn max_brokerage(&self) -> f64 { self.inner.max_brokerage }
    #[wasm_bindgen(setter)] pub fn set_max_brokerage(&mut self, v: f64) { self.inner.max_brokerage = v; }

    #[wasm_bindgen(getter)] pub fn stt_rate(&self) -> f64 { self.inner.stt_rate }
    #[wasm_bindgen(setter)] pub fn set_stt_rate(&mut self, v: f64) { self.inner.stt_rate = v; }

    #[wasm_bindgen(getter)] pub fn stt_on_buy(&self) -> bool { self.inner.stt_on_buy }
    #[wasm_bindgen(setter)] pub fn set_stt_on_buy(&mut self, v: bool) { self.inner.stt_on_buy = v; }

    #[wasm_bindgen(getter)] pub fn stt_on_sell(&self) -> bool { self.inner.stt_on_sell }
    #[wasm_bindgen(setter)] pub fn set_stt_on_sell(&mut self, v: bool) { self.inner.stt_on_sell = v; }

    #[wasm_bindgen(getter)] pub fn exchange_charges_rate(&self) -> f64 { self.inner.exchange_charges_rate }
    #[wasm_bindgen(setter)] pub fn set_exchange_charges_rate(&mut self, v: f64) { self.inner.exchange_charges_rate = v; }

    #[wasm_bindgen(getter)] pub fn regulatory_charges_rate(&self) -> f64 { self.inner.regulatory_charges_rate }
    #[wasm_bindgen(setter)] pub fn set_regulatory_charges_rate(&mut self, v: f64) { self.inner.regulatory_charges_rate = v; }

    #[wasm_bindgen(getter)] pub fn gst_rate(&self) -> f64 { self.inner.gst_rate }
    #[wasm_bindgen(setter)] pub fn set_gst_rate(&mut self, v: f64) { self.inner.gst_rate = v; }

    #[wasm_bindgen(getter)] pub fn stamp_duty_rate(&self) -> f64 { self.inner.stamp_duty_rate }
    #[wasm_bindgen(setter)] pub fn set_stamp_duty_rate(&mut self, v: f64) { self.inner.stamp_duty_rate = v; }

    #[wasm_bindgen(getter)] pub fn lot_size(&self) -> f64 { self.inner.lot_size }
    #[wasm_bindgen(setter)] pub fn set_lot_size(&mut self, v: f64) { self.inner.lot_size = v; }

    // ---- Compute --------------------------------------------------------

    /// Total transaction cost in absolute currency units.
    pub fn total_cost(&self, trade_value: f64, num_lots: f64, is_buy: bool) -> f64 {
        self.inner.total_cost(trade_value, num_lots, is_buy)
    }

    /// Cost as fraction of `initial_capital` (for normalised equity loops).
    pub fn cost_fraction(&self, trade_value: f64, num_lots: f64, is_buy: bool, initial_capital: f64) -> f64 {
        self.inner.cost_fraction(trade_value, num_lots, is_buy, initial_capital)
    }

    // ---- Presets (static constructors) ----------------------------------

    /// Zero-commission model.
    pub fn zero() -> CommissionModel {
        CommissionModel { inner: ferro_ta_core::commission::CommissionModel::zero() }
    }

    /// Indian equity delivery preset.
    pub fn equity_delivery_india() -> CommissionModel {
        CommissionModel { inner: ferro_ta_core::commission::CommissionModel::equity_delivery_india() }
    }

    /// Indian equity intraday preset.
    pub fn equity_intraday_india() -> CommissionModel {
        CommissionModel { inner: ferro_ta_core::commission::CommissionModel::equity_intraday_india() }
    }

    /// Indian index futures preset.
    pub fn futures_india() -> CommissionModel {
        CommissionModel { inner: ferro_ta_core::commission::CommissionModel::futures_india() }
    }

    /// Indian index options preset.
    pub fn options_india() -> CommissionModel {
        CommissionModel { inner: ferro_ta_core::commission::CommissionModel::options_india() }
    }

    /// Simple proportional model (no taxes, `rate` fraction both ways).
    pub fn proportional(rate: f64) -> CommissionModel {
        CommissionModel { inner: ferro_ta_core::commission::CommissionModel::proportional(rate) }
    }

    // ---- JSON (minimal manual serialization — no serde in WASM) ----------

    /// Serialize key fields to a JSON string (no serde dependency).
    pub fn to_json_string(&self) -> String {
        let m = &self.inner;
        format!(
            r#"{{"flat_per_order":{},"rate_of_value":{},"per_lot":{},"max_brokerage":{},"stt_rate":{},"stt_on_buy":{},"stt_on_sell":{},"exchange_charges_rate":{},"regulatory_charges_rate":{},"gst_rate":{},"stamp_duty_rate":{},"lot_size":{},"spread_bps":{},"short_borrow_rate_annual":{}}}"#,
            m.flat_per_order, m.rate_of_value, m.per_lot, m.max_brokerage,
            m.stt_rate, m.stt_on_buy, m.stt_on_sell,
            m.exchange_charges_rate, m.regulatory_charges_rate,
            m.gst_rate, m.stamp_duty_rate, m.lot_size,
            m.spread_bps, m.short_borrow_rate_annual,
        )
    }
}

// ===========================================================================
// Price Transform
// ===========================================================================

/// Average Price: (open + high + low + close) / 4.
#[wasm_bindgen]
pub fn avgprice(
    open: &Float64Array,
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
) -> Float64Array {
    let o = to_vec(open);
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::price_transform::avgprice(&o, &h, &l, &c))
}

/// Median Price: (high + low) / 2.
#[wasm_bindgen]
pub fn medprice(high: &Float64Array, low: &Float64Array) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    from_vec(ferro_ta_core::price_transform::medprice(&h, &l))
}

/// Typical Price: (high + low + close) / 3.
#[wasm_bindgen]
pub fn typprice(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::price_transform::typprice(&h, &l, &c))
}

/// Weighted Close Price: (high + low + close * 2) / 4.
#[wasm_bindgen]
pub fn wclprice(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::price_transform::wclprice(&h, &l, &c))
}

// ===========================================================================
// Alerts
// ===========================================================================

/// Fire an alert when series crosses a threshold level.
/// direction: 1 = cross above, -1 = cross below.
/// Returns Int8Array: 1 at crossing bars, 0 elsewhere.
#[wasm_bindgen]
pub fn check_threshold(series: &Float64Array, level: f64, direction: i32) -> js_sys::Int8Array {
    let s = to_vec(series);
    let result = ferro_ta_core::alerts::check_threshold(&s, level, direction);
    let arr = js_sys::Int8Array::new_with_length(result.len() as u32);
    arr.copy_from(&result);
    arr
}

/// Detect cross-over/cross-under events between fast and slow series.
/// Returns Int8Array: 1 = bullish, -1 = bearish, 0 = none.
#[wasm_bindgen]
pub fn check_cross(fast: &Float64Array, slow: &Float64Array) -> js_sys::Int8Array {
    let f = to_vec(fast);
    let s = to_vec(slow);
    let result = ferro_ta_core::alerts::check_cross(&f, &s);
    let arr = js_sys::Int8Array::new_with_length(result.len() as u32);
    arr.copy_from(&result);
    arr
}

/// Collect bar indices where mask is non-zero.
#[wasm_bindgen]
pub fn collect_alert_bars(mask: &js_sys::Int8Array) -> Float64Array {
    let n = mask.length() as usize;
    let mut m = vec![0i8; n];
    mask.copy_to(&mut m);
    let result = ferro_ta_core::alerts::collect_alert_bars(&m);
    let out: Vec<f64> = result.into_iter().map(|v| v as f64).collect();
    from_vec(out)
}

// ===========================================================================
// Signals
// ===========================================================================

/// Compute fractional rank of each element (1-based, ascending).
#[wasm_bindgen]
pub fn rank_series(x: &Float64Array) -> Float64Array {
    let xv = to_vec(x);
    from_vec(ferro_ta_core::signals::rank_values(&xv))
}

/// Return indices of the N largest values.
#[wasm_bindgen]
pub fn top_n_indices(x: &Float64Array, n: usize) -> Float64Array {
    let xv = to_vec(x);
    let result = ferro_ta_core::signals::top_n_indices(&xv, n);
    let out: Vec<f64> = result.into_iter().map(|v| v as f64).collect();
    from_vec(out)
}

/// Return indices of the N smallest values.
#[wasm_bindgen]
pub fn bottom_n_indices(x: &Float64Array, n: usize) -> Float64Array {
    let xv = to_vec(x);
    let result = ferro_ta_core::signals::bottom_n_indices(&xv, n);
    let out: Vec<f64> = result.into_iter().map(|v| v as f64).collect();
    from_vec(out)
}

// ===========================================================================
// Crypto
// ===========================================================================

/// Cumulative PnL from funding rate payments.
#[wasm_bindgen]
pub fn funding_cumulative_pnl(
    position_size: &Float64Array,
    funding_rate: &Float64Array,
) -> Float64Array {
    let pos = to_vec(position_size);
    let rate = to_vec(funding_rate);
    from_vec(ferro_ta_core::crypto::funding_cumulative_pnl(&pos, &rate))
}

/// Assign sequential integer labels based on fixed period size.
#[wasm_bindgen]
pub fn continuous_bar_labels(n_bars: usize, period_bars: usize) -> Float64Array {
    let result = ferro_ta_core::crypto::continuous_bar_labels(n_bars, period_bars);
    let out: Vec<f64> = result.into_iter().map(|v| v as f64).collect();
    from_vec(out)
}

// ===========================================================================
// Math Ops
// ===========================================================================

/// Rolling sum over timeperiod bars.
#[wasm_bindgen]
pub fn rolling_sum(real: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(real);
    from_vec(ferro_ta_core::math_ops::rolling_sum(&prices, timeperiod))
}

/// Rolling maximum over timeperiod bars.
#[wasm_bindgen]
pub fn rolling_max(real: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(real);
    from_vec(ferro_ta_core::math_ops::rolling_max(&prices, timeperiod))
}

/// Rolling minimum over timeperiod bars.
#[wasm_bindgen]
pub fn rolling_min(real: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(real);
    from_vec(ferro_ta_core::math_ops::rolling_min(&prices, timeperiod))
}

/// Index of rolling maximum over timeperiod bars.
#[wasm_bindgen]
pub fn rolling_maxindex(real: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(real);
    let result = ferro_ta_core::math_ops::rolling_maxindex(&prices, timeperiod);
    let out: Vec<f64> = result.into_iter().map(|v| v as f64).collect();
    from_vec(out)
}

/// Index of rolling minimum over timeperiod bars.
#[wasm_bindgen]
pub fn rolling_minindex(real: &Float64Array, timeperiod: usize) -> Float64Array {
    let prices = to_vec(real);
    let result = ferro_ta_core::math_ops::rolling_minindex(&prices, timeperiod);
    let out: Vec<f64> = result.into_iter().map(|v| v as f64).collect();
    from_vec(out)
}

// ===========================================================================
// Regime
// ===========================================================================

/// Label bars as trend (1), range (0), or NaN (-1) based on ADX threshold.
#[wasm_bindgen]
pub fn regime_adx(adx: &Float64Array, threshold: f64) -> js_sys::Int8Array {
    let a = to_vec(adx);
    let result = ferro_ta_core::regime::regime_adx(&a, threshold);
    let arr = js_sys::Int8Array::new_with_length(result.len() as u32);
    arr.copy_from(&result);
    arr
}

/// Label bars using ADX + ATR-ratio combined rule.
#[wasm_bindgen]
pub fn regime_combined(
    adx: &Float64Array,
    atr: &Float64Array,
    close: &Float64Array,
    adx_threshold: f64,
    atr_pct_threshold: f64,
) -> js_sys::Int8Array {
    let a = to_vec(adx);
    let r = to_vec(atr);
    let c = to_vec(close);
    let result = ferro_ta_core::regime::regime_combined(&a, &r, &c, adx_threshold, atr_pct_threshold);
    let arr = js_sys::Int8Array::new_with_length(result.len() as u32);
    arr.copy_from(&result);
    arr
}

/// Detect structural breaks using CUSUM approach.
#[wasm_bindgen]
pub fn detect_breaks_cusum(
    series: &Float64Array,
    window: usize,
    threshold: f64,
    slack: f64,
) -> js_sys::Int8Array {
    let s = to_vec(series);
    let result = ferro_ta_core::regime::detect_breaks_cusum(&s, window, threshold, slack);
    let arr = js_sys::Int8Array::new_with_length(result.len() as u32);
    arr.copy_from(&result);
    arr
}

/// Detect volatility regime breaks using rolling variance ratio.
#[wasm_bindgen]
pub fn rolling_variance_break(
    series: &Float64Array,
    short_window: usize,
    long_window: usize,
    threshold: f64,
) -> js_sys::Int8Array {
    let s = to_vec(series);
    let result = ferro_ta_core::regime::rolling_variance_break(&s, short_window, long_window, threshold);
    let arr = js_sys::Int8Array::new_with_length(result.len() as u32);
    arr.copy_from(&result);
    arr
}

// ===========================================================================
// Chunked
// ===========================================================================

/// Remove first overlap elements from an array.
#[wasm_bindgen]
pub fn trim_overlap(chunk_out: &Float64Array, overlap: usize) -> Float64Array {
    let s = to_vec(chunk_out);
    from_vec(ferro_ta_core::chunked::trim_overlap(&s, overlap))
}

/// Compute (start, end) index pairs for chunked processing.
/// Returns flat Float64Array: [start0, end0, start1, end1, ...].
#[wasm_bindgen]
pub fn make_chunk_ranges(n: usize, chunk_size: usize, overlap: usize) -> Float64Array {
    let result = ferro_ta_core::chunked::make_chunk_ranges(n, chunk_size, overlap);
    let out: Vec<f64> = result.into_iter().map(|v| v as f64).collect();
    from_vec(out)
}

/// Forward-fill NaN values in a 1-D array.
#[wasm_bindgen]
pub fn forward_fill_nan(values: &Float64Array) -> Float64Array {
    let input = to_vec(values);
    from_vec(ferro_ta_core::chunked::forward_fill_nan(&input))
}

// ===========================================================================
// Extended Indicators (Sprint 2)
// ===========================================================================

/// Volume Weighted Average Price (cumulative or rolling).
#[wasm_bindgen]
pub fn vwap(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    volume: &Float64Array,
    timeperiod: usize,
) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let v = to_vec(volume);
    from_vec(ferro_ta_core::extended::vwap(&h, &l, &c, &v, timeperiod))
}

/// Volume Weighted Moving Average.
#[wasm_bindgen]
pub fn vwma(close: &Float64Array, volume: &Float64Array, timeperiod: usize) -> Float64Array {
    let c = to_vec(close);
    let v = to_vec(volume);
    from_vec(ferro_ta_core::extended::vwma(&c, &v, timeperiod))
}

/// ATR-based Supertrend indicator.
/// Returns `[supertrend_line, direction_as_f64]`.
#[wasm_bindgen]
pub fn supertrend(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    timeperiod: usize,
    multiplier: f64,
) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let (line, direction) = ferro_ta_core::extended::supertrend(&h, &l, &c, timeperiod, multiplier);
    let dir_f64: Vec<f64> = direction.iter().map(|&d| d as f64).collect();
    let out = Array::new();
    out.push(&from_vec(line));
    out.push(&from_vec(dir_f64));
    out
}

/// Donchian Channels — rolling highest high / lowest low.
/// Returns `[upper, middle, lower]`.
#[wasm_bindgen]
pub fn donchian(high: &Float64Array, low: &Float64Array, timeperiod: usize) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let (upper, middle, lower) = ferro_ta_core::extended::donchian(&h, &l, timeperiod);
    let out = Array::new();
    out.push(&from_vec(upper));
    out.push(&from_vec(middle));
    out.push(&from_vec(lower));
    out
}

/// Choppiness Index — measures market choppiness vs trending.
#[wasm_bindgen]
pub fn choppiness_index(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    timeperiod: usize,
) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::extended::choppiness_index(&h, &l, &c, timeperiod))
}

/// Keltner Channels — EMA +/- (multiplier x ATR).
/// Returns `[upper, middle, lower]`.
#[wasm_bindgen]
pub fn keltner_channels(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    timeperiod: usize,
    atr_period: usize,
    multiplier: f64,
) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let (upper, middle, lower) =
        ferro_ta_core::extended::keltner_channels(&h, &l, &c, timeperiod, atr_period, multiplier);
    let out = Array::new();
    out.push(&from_vec(upper));
    out.push(&from_vec(middle));
    out.push(&from_vec(lower));
    out
}

/// Hull Moving Average (HMA).
#[wasm_bindgen]
pub fn hull_ma(close: &Float64Array, timeperiod: usize) -> Float64Array {
    let c = to_vec(close);
    from_vec(ferro_ta_core::extended::hull_ma(&c, timeperiod))
}

/// Chandelier Exit — ATR-based trailing stop levels.
/// Returns `[long_exit, short_exit]`.
#[wasm_bindgen]
pub fn chandelier_exit(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    timeperiod: usize,
    multiplier: f64,
) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let (long_exit, short_exit) =
        ferro_ta_core::extended::chandelier_exit(&h, &l, &c, timeperiod, multiplier);
    let out = Array::new();
    out.push(&from_vec(long_exit));
    out.push(&from_vec(short_exit));
    out
}

/// Ichimoku Cloud (Ichimoku Kinko Hyo).
/// Returns `[tenkan, kijun, senkou_a, senkou_b, chikou]`.
#[wasm_bindgen]
pub fn ichimoku(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    tenkan: usize,
    kijun: usize,
    senkou_b: usize,
    displacement: usize,
) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let (tenkan_out, kijun_out, senkou_a_out, senkou_b_out, chikou_out) =
        ferro_ta_core::extended::ichimoku(&h, &l, &c, tenkan, kijun, senkou_b, displacement);
    let out = Array::new();
    out.push(&from_vec(tenkan_out));
    out.push(&from_vec(kijun_out));
    out.push(&from_vec(senkou_a_out));
    out.push(&from_vec(senkou_b_out));
    out.push(&from_vec(chikou_out));
    out
}

/// Pivot Points — support / resistance levels.
/// Returns `[pivot, r1, s1, r2, s2]`.
#[wasm_bindgen(js_name = "pivot_points")]
pub fn pivot_points(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    method: &str,
) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let (pivot, r1, s1, r2, s2) = ferro_ta_core::extended::pivot_points(&h, &l, &c, method);
    let out = Array::new();
    out.push(&from_vec(pivot));
    out.push(&from_vec(r1));
    out.push(&from_vec(s1));
    out.push(&from_vec(r2));
    out.push(&from_vec(s2));
    out
}

// ===========================================================================
// Portfolio Analytics (Sprint 2)
// ===========================================================================

/// Full-sample OLS beta of asset vs benchmark returns.
#[wasm_bindgen]
pub fn beta_full(asset_returns: &Float64Array, benchmark_returns: &Float64Array) -> f64 {
    let a = to_vec(asset_returns);
    let b = to_vec(benchmark_returns);
    ferro_ta_core::portfolio::beta_full(&a, &b)
}

/// Rolling beta of asset vs benchmark over a sliding window.
#[wasm_bindgen]
pub fn rolling_beta(
    asset: &Float64Array,
    benchmark: &Float64Array,
    window: usize,
) -> Float64Array {
    let a = to_vec(asset);
    let b = to_vec(benchmark);
    from_vec(ferro_ta_core::portfolio::rolling_beta(&a, &b, window))
}

/// Drawdown series and maximum drawdown for an equity curve.
/// Returns `[dd_array, max_dd_as_single_element]`.
#[wasm_bindgen]
pub fn drawdown_series(equity: &Float64Array) -> Array {
    let eq = to_vec(equity);
    let (dd, max_dd) = ferro_ta_core::portfolio::drawdown_series(&eq);
    let out = Array::new();
    out.push(&from_vec(dd));
    out.push(&from_vec(vec![max_dd]));
    out
}

/// Relative strength of asset vs benchmark (cumulative return ratio).
#[wasm_bindgen]
pub fn relative_strength(
    asset_returns: &Float64Array,
    benchmark_returns: &Float64Array,
) -> Float64Array {
    let a = to_vec(asset_returns);
    let b = to_vec(benchmark_returns);
    from_vec(ferro_ta_core::portfolio::relative_strength(&a, &b))
}

/// Spread between two series: a - hedge * b.
#[wasm_bindgen]
pub fn spread(a: &Float64Array, b: &Float64Array, hedge: f64) -> Float64Array {
    let av = to_vec(a);
    let bv = to_vec(b);
    from_vec(ferro_ta_core::portfolio::spread(&av, &bv, hedge))
}

/// Ratio between two series: a / b (NaN where b is zero).
#[wasm_bindgen]
pub fn ratio(a: &Float64Array, b: &Float64Array) -> Float64Array {
    let av = to_vec(a);
    let bv = to_vec(b);
    from_vec(ferro_ta_core::portfolio::ratio(&av, &bv))
}

/// Rolling Z-score of a 1-D series.
#[wasm_bindgen]
pub fn zscore_series(x: &Float64Array, window: usize) -> Float64Array {
    let xv = to_vec(x);
    from_vec(ferro_ta_core::portfolio::zscore_series(&xv, window))
}

// ===========================================================================
// Attribution (Sprint 2)
// ===========================================================================

/// Trade-level statistics from trade PnL and hold durations.
/// Returns `[win_rate, avg_win, avg_loss, profit_factor, avg_hold_bars]` as Float64Array.
#[wasm_bindgen]
pub fn trade_stats(pnl: &Float64Array, hold_bars: &Float64Array) -> Array {
    let p = to_vec(pnl);
    let h = to_vec(hold_bars);
    let (win_rate, avg_win, avg_loss, profit_factor, avg_hold) =
        ferro_ta_core::attribution::trade_stats(&p, &h);
    let out = Array::new();
    out.push(&from_vec(vec![win_rate, avg_win, avg_loss, profit_factor, avg_hold]));
    out
}

/// Group per-bar returns by month index and sum each month's contribution.
/// Returns `[months_as_f64, contributions]`.
#[wasm_bindgen]
pub fn monthly_contribution(
    bar_returns: &Float64Array,
    month_index: &Float64Array,
) -> Array {
    let ret = to_vec(bar_returns);
    let mi_f64 = to_vec(month_index);
    let mi: Vec<i64> = mi_f64.iter().map(|&v| v as i64).collect();
    let (months, contributions) = ferro_ta_core::attribution::monthly_contribution(&ret, &mi);
    let months_f64: Vec<f64> = months.iter().map(|&m| m as f64).collect();
    let out = Array::new();
    out.push(&from_vec(months_f64));
    out.push(&from_vec(contributions));
    out
}

/// Attribute per-bar returns to each signal label.
/// Returns `[labels_as_f64, contributions]`.
#[wasm_bindgen]
pub fn signal_attribution(
    bar_returns: &Float64Array,
    signal_labels: &Float64Array,
) -> Array {
    let ret = to_vec(bar_returns);
    let sl_f64 = to_vec(signal_labels);
    let sl: Vec<i64> = sl_f64.iter().map(|&v| v as i64).collect();
    let (labels, contributions) = ferro_ta_core::attribution::signal_attribution(&ret, &sl);
    let labels_f64: Vec<f64> = labels.iter().map(|&l| l as f64).collect();
    let out = Array::new();
    out.push(&from_vec(labels_f64));
    out.push(&from_vec(contributions));
    out
}

/// Extract trade PnL and hold durations from positions and strategy returns.
/// Returns `[pnl, hold_durations]`.
#[wasm_bindgen]
pub fn extract_trades(
    positions: &Float64Array,
    strategy_returns: &Float64Array,
) -> Array {
    let pos = to_vec(positions);
    let sr = to_vec(strategy_returns);
    let (pnl, hold) = ferro_ta_core::attribution::extract_trades(&pos, &sr);
    let out = Array::new();
    out.push(&from_vec(pnl));
    out.push(&from_vec(hold));
    out
}

// ===========================================================================
// Resampling (Sprint 2)
// ===========================================================================

/// Aggregate OHLCV data into volume bars of a fixed volume threshold.
/// Returns `[open, high, low, close, volume]`.
#[wasm_bindgen]
pub fn volume_bars(
    open: &Float64Array,
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    volume: &Float64Array,
    volume_threshold: f64,
) -> Array {
    let o = to_vec(open);
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let v = to_vec(volume);
    let (ro, rh, rl, rc, rv) =
        ferro_ta_core::resampling::volume_bars(&o, &h, &l, &c, &v, volume_threshold);
    let out = Array::new();
    out.push(&from_vec(ro));
    out.push(&from_vec(rh));
    out.push(&from_vec(rl));
    out.push(&from_vec(rc));
    out.push(&from_vec(rv));
    out
}

/// Aggregate OHLCV bars by integer group labels.
/// Returns `[open, high, low, close, volume]`.
#[wasm_bindgen]
pub fn ohlcv_agg(
    open: &Float64Array,
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    volume: &Float64Array,
    labels: &Float64Array,
) -> Array {
    let o = to_vec(open);
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let v = to_vec(volume);
    let lbl_f64 = to_vec(labels);
    let lbl: Vec<i64> = lbl_f64.iter().map(|&x| x as i64).collect();
    let (ro, rh, rl, rc, rv) =
        ferro_ta_core::resampling::ohlcv_agg(&o, &h, &l, &c, &v, &lbl);
    let out = Array::new();
    out.push(&from_vec(ro));
    out.push(&from_vec(rh));
    out.push(&from_vec(rl));
    out.push(&from_vec(rc));
    out.push(&from_vec(rv));
    out
}

// ===========================================================================
// Aggregation (Sprint 2)
// ===========================================================================

/// Aggregate tick/trade data into tick bars (every N ticks become one bar).
/// Returns `[open, high, low, close, volume]`.
#[wasm_bindgen]
pub fn aggregate_tick_bars(
    price: &Float64Array,
    size: &Float64Array,
    ticks_per_bar: usize,
) -> Array {
    let p = to_vec(price);
    let s = to_vec(size);
    let (o, h, l, c, v) = ferro_ta_core::aggregation::aggregate_tick_bars(&p, &s, ticks_per_bar);
    let out = Array::new();
    out.push(&from_vec(o));
    out.push(&from_vec(h));
    out.push(&from_vec(l));
    out.push(&from_vec(c));
    out.push(&from_vec(v));
    out
}

/// Aggregate tick data into volume bars (fixed volume threshold).
/// Returns `[open, high, low, close, volume]`.
#[wasm_bindgen]
pub fn aggregate_volume_bars_ticks(
    price: &Float64Array,
    size: &Float64Array,
    volume_threshold: f64,
) -> Array {
    let p = to_vec(price);
    let s = to_vec(size);
    let (o, h, l, c, v) =
        ferro_ta_core::aggregation::aggregate_volume_bars_ticks(&p, &s, volume_threshold);
    let out = Array::new();
    out.push(&from_vec(o));
    out.push(&from_vec(h));
    out.push(&from_vec(l));
    out.push(&from_vec(c));
    out.push(&from_vec(v));
    out
}

/// Aggregate tick data into time bars using pre-computed integer bucket labels.
/// Returns `[open, high, low, close, volume, labels_as_f64]`.
#[wasm_bindgen]
pub fn aggregate_time_bars(
    price: &Float64Array,
    size: &Float64Array,
    labels: &Float64Array,
) -> Array {
    let p = to_vec(price);
    let s = to_vec(size);
    let lbl_f64 = to_vec(labels);
    let lbl: Vec<i64> = lbl_f64.iter().map(|&x| x as i64).collect();
    let (o, h, l, c, v, out_labels) =
        ferro_ta_core::aggregation::aggregate_time_bars(&p, &s, &lbl);
    let labels_out: Vec<f64> = out_labels.iter().map(|&x| x as f64).collect();
    let out = Array::new();
    out.push(&from_vec(o));
    out.push(&from_vec(h));
    out.push(&from_vec(l));
    out.push(&from_vec(c));
    out.push(&from_vec(v));
    out.push(&from_vec(labels_out));
    out
}

// ===========================================================================
// Cycle Indicators
// ===========================================================================

#[wasm_bindgen]
pub fn ht_trendline(close: &Float64Array) -> Float64Array {
    let c = to_vec(close);
    from_vec(ferro_ta_core::cycle::ht_trendline(&c))
}

#[wasm_bindgen]
pub fn ht_dcperiod(close: &Float64Array) -> Float64Array {
    let c = to_vec(close);
    from_vec(ferro_ta_core::cycle::ht_dcperiod(&c))
}

#[wasm_bindgen]
pub fn ht_dcphase(close: &Float64Array) -> Float64Array {
    let c = to_vec(close);
    from_vec(ferro_ta_core::cycle::ht_dcphase(&c))
}

#[wasm_bindgen]
pub fn ht_phasor(close: &Float64Array) -> Array {
    let c = to_vec(close);
    let (inphase, quad) = ferro_ta_core::cycle::ht_phasor(&c);
    let arr = Array::new();
    arr.push(&from_vec(inphase)); arr.push(&from_vec(quad));
    arr
}

#[wasm_bindgen]
pub fn ht_sine(close: &Float64Array) -> Array {
    let c = to_vec(close);
    let (sine, leadsine) = ferro_ta_core::cycle::ht_sine(&c);
    let arr = Array::new();
    arr.push(&from_vec(sine)); arr.push(&from_vec(leadsine));
    arr
}

#[wasm_bindgen]
pub fn ht_trendmode(close: &Float64Array) -> Float64Array {
    let c = to_vec(close);
    let result = ferro_ta_core::cycle::ht_trendmode(&c);
    let out: Vec<f64> = result.into_iter().map(|v| v as f64).collect();
    from_vec(out)
}

// ===========================================================================
// Volume (additional exports)
// ===========================================================================

#[wasm_bindgen]
pub fn ad(high: &Float64Array, low: &Float64Array, close: &Float64Array, volume: &Float64Array) -> Float64Array {
    let h = to_vec(high); let l = to_vec(low); let c = to_vec(close); let v = to_vec(volume);
    from_vec(ferro_ta_core::volume::ad(&h, &l, &c, &v))
}

#[wasm_bindgen]
pub fn adosc(high: &Float64Array, low: &Float64Array, close: &Float64Array, volume: &Float64Array, fastperiod: usize, slowperiod: usize) -> Float64Array {
    let h = to_vec(high); let l = to_vec(low); let c = to_vec(close); let v = to_vec(volume);
    from_vec(ferro_ta_core::volume::adosc(&h, &l, &c, &v, fastperiod, slowperiod))
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
        // OBV starts at 0: 0, 200, 50, 350, 100
        let close  = make_arr(&[10.0, 11.0, 10.0, 12.0, 11.0]);
        let volume = make_arr(&[100.0, 200.0, 150.0, 300.0, 250.0]);
        let out = obv(&close, &volume);
        let mut vals = vec![0.0f64; 5];
        out.copy_to(&mut vals);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 200.0).abs() < 1e-10);
        assert!((vals[2] - 50.0).abs() < 1e-10);
        assert!((vals[3] - 350.0).abs() < 1e-10);
        assert!((vals[4] - 100.0).abs() < 1e-10);
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

    // -----------------------------------------------------------------------
    // WMA tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_wma_output_length() {
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = wma(&close, 3);
        assert_eq!(out.length(), 5);
    }

    #[wasm_bindgen_test]
    fn test_wma_known_value() {
        // WMA(3) at index 2 = (1*1 + 2*2 + 3*3) / 6 = 14/6
        let close = make_arr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = wma(&close, 3);
        let mut vals = vec![0.0f64; 5];
        out.copy_to(&mut vals);
        assert!(vals[0].is_nan());
        assert!(vals[1].is_nan());
        assert!((vals[2] - (14.0 / 6.0)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // ADX tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_adx_output_length() {
        let h = make_arr(&[10.0, 11.0, 12.0, 13.0, 13.5, 14.0, 14.5, 15.0]);
        let l = make_arr(&[9.0, 9.5, 10.5, 11.5, 12.0, 12.5, 13.0, 13.5]);
        let c = make_arr(&[9.5, 10.5, 11.5, 12.0, 13.0, 13.5, 14.0, 14.5]);
        let out = adx(&h, &l, &c, 3);
        assert_eq!(out.length(), 8);
    }

    #[wasm_bindgen_test]
    fn test_adx_values_in_range() {
        let h = make_arr(&[10.0, 11.0, 12.0, 13.0, 13.5, 14.0, 14.5, 15.0]);
        let l = make_arr(&[9.0, 9.5, 10.5, 11.5, 12.0, 12.5, 13.0, 13.5]);
        let c = make_arr(&[9.5, 10.5, 11.5, 12.0, 13.0, 13.5, 14.0, 14.5]);
        let out = adx(&h, &l, &c, 3);
        for v in get_finite(&out) {
            assert!((0.0..=100.0).contains(&v), "ADX out of range: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // MFI tests
    // -----------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_mfi_output_length() {
        let h = make_arr(&[10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 13.5]);
        let l = make_arr(&[9.0, 9.5, 10.5, 10.0, 11.0, 11.5, 12.0]);
        let c = make_arr(&[9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 13.0]);
        let v = make_arr(&[100.0, 110.0, 120.0, 130.0, 125.0, 140.0, 150.0]);
        let out = mfi(&h, &l, &c, &v, 3);
        assert_eq!(out.length(), 7);
    }

    #[wasm_bindgen_test]
    fn test_mfi_values_in_range() {
        let h = make_arr(&[10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 13.5]);
        let l = make_arr(&[9.0, 9.5, 10.5, 10.0, 11.0, 11.5, 12.0]);
        let c = make_arr(&[9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 13.0]);
        let v = make_arr(&[100.0, 110.0, 120.0, 130.0, 125.0, 140.0, 150.0]);
        let out = mfi(&h, &l, &c, &v, 3);
        for val in get_finite(&out) {
            assert!((0.0..=100.0).contains(&val), "MFI out of range: {val}");
        }
    }
}
