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

// ===========================================================================
// Momentum (additional exports)
// ===========================================================================

/// Full Stochastic Oscillator (slow %K and slow %D).
#[wasm_bindgen]
pub fn stoch(
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let (slowk, slowd) = ferro_ta_core::momentum::stoch(&h, &l, &c, fastk_period, slowk_period, slowd_period);
    let out = Array::new();
    out.push(&from_vec(slowk));
    out.push(&from_vec(slowd));
    out
}

/// Plus Directional Movement (+DM).
#[wasm_bindgen]
pub fn plus_dm(high: &Float64Array, low: &Float64Array, timeperiod: usize) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    from_vec(ferro_ta_core::momentum::plus_dm(&h, &l, timeperiod))
}

/// Minus Directional Movement (-DM).
#[wasm_bindgen]
pub fn minus_dm(high: &Float64Array, low: &Float64Array, timeperiod: usize) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    from_vec(ferro_ta_core::momentum::minus_dm(&h, &l, timeperiod))
}

/// Plus Directional Indicator (+DI).
#[wasm_bindgen]
pub fn plus_di(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod: usize) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::momentum::plus_di(&h, &l, &c, timeperiod))
}

/// Minus Directional Indicator (-DI).
#[wasm_bindgen]
pub fn minus_di(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod: usize) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::momentum::minus_di(&h, &l, &c, timeperiod))
}

/// Directional Movement Index (DX).
#[wasm_bindgen]
pub fn dx(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod: usize) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::momentum::dx(&h, &l, &c, timeperiod))
}

/// Average Directional Movement Index Rating (ADXR).
#[wasm_bindgen]
pub fn adxr(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod: usize) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::momentum::adxr(&h, &l, &c, timeperiod))
}

/// All ADX components: returns [+DM, -DM, +DI, -DI, DX, ADX].
#[wasm_bindgen]
pub fn adx_all(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod: usize) -> Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    let (pdm, mdm, pdi, mdi, dxv, adxv) = ferro_ta_core::momentum::adx_all(&h, &l, &c, timeperiod);
    let out = Array::new();
    out.push(&from_vec(pdm));
    out.push(&from_vec(mdm));
    out.push(&from_vec(pdi));
    out.push(&from_vec(mdi));
    out.push(&from_vec(dxv));
    out.push(&from_vec(adxv));
    out
}

// ===========================================================================
// Overlap Studies (additional exports)
// ===========================================================================

/// Double Exponential Moving Average.
#[wasm_bindgen]
pub fn dema(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::overlap::dema(&to_vec(close), timeperiod))
}

/// Triple Exponential Moving Average.
#[wasm_bindgen]
pub fn tema(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::overlap::tema(&to_vec(close), timeperiod))
}

/// Triangular Moving Average.
#[wasm_bindgen]
pub fn trima(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::overlap::trima(&to_vec(close), timeperiod))
}

/// Kaufman Adaptive Moving Average.
#[wasm_bindgen]
pub fn kama(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::overlap::kama(&to_vec(close), timeperiod))
}

/// Tillson T3.
#[wasm_bindgen]
pub fn t3(close: &Float64Array, timeperiod: usize, vfactor: f64) -> Float64Array {
    from_vec(ferro_ta_core::overlap::t3(&to_vec(close), timeperiod, vfactor))
}

/// Parabolic SAR.
#[wasm_bindgen]
pub fn sar(high: &Float64Array, low: &Float64Array, acceleration: f64, maximum: f64) -> Float64Array {
    from_vec(ferro_ta_core::overlap::sar(&to_vec(high), &to_vec(low), acceleration, maximum))
}

/// Parabolic SAR Extended.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn sarext(
    high: &Float64Array, low: &Float64Array,
    startvalue: f64, offsetonreverse: f64,
    accelerationinitlong: f64, accelerationlong: f64, accelerationmaxlong: f64,
    accelerationinitshort: f64, accelerationshort: f64, accelerationmaxshort: f64,
) -> Float64Array {
    from_vec(ferro_ta_core::overlap::sarext(
        &to_vec(high), &to_vec(low),
        startvalue, offsetonreverse,
        accelerationinitlong, accelerationlong, accelerationmaxlong,
        accelerationinitshort, accelerationshort, accelerationmaxshort,
    ))
}

/// MESA Adaptive Moving Average. Returns [mama, fama].
#[wasm_bindgen]
pub fn mama(close: &Float64Array, fastlimit: f64, slowlimit: f64) -> Array {
    let (m, f) = ferro_ta_core::overlap::mama(&to_vec(close), fastlimit, slowlimit);
    let out = Array::new();
    out.push(&from_vec(m));
    out.push(&from_vec(f));
    out
}

/// Midpoint over rolling window.
#[wasm_bindgen]
pub fn midpoint(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::overlap::midpoint(&to_vec(close), timeperiod))
}

/// MidPrice over rolling window.
#[wasm_bindgen]
pub fn midprice(high: &Float64Array, low: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::overlap::midprice(&to_vec(high), &to_vec(low), timeperiod))
}

/// MACD with fixed 12/26 periods. Returns [macd, signal, histogram].
#[wasm_bindgen]
pub fn macdfix(close: &Float64Array, signalperiod: usize) -> Array {
    let (m, s, h) = ferro_ta_core::overlap::macdfix(&to_vec(close), signalperiod);
    let out = Array::new();
    out.push(&from_vec(m));
    out.push(&from_vec(s));
    out.push(&from_vec(h));
    out
}

/// MACD with configurable MA types. Returns [macd, signal, histogram].
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn macdext(
    close: &Float64Array, fastperiod: usize, fastmatype: u8,
    slowperiod: usize, slowmatype: u8, signalperiod: usize, signalmatype: u8,
) -> Array {
    let (m, s, h) = ferro_ta_core::overlap::macdext(
        &to_vec(close), fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype,
    );
    let out = Array::new();
    out.push(&from_vec(m));
    out.push(&from_vec(s));
    out.push(&from_vec(h));
    out
}

/// Generic Moving Average (matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=T3).
#[wasm_bindgen]
pub fn ma(close: &Float64Array, timeperiod: usize, matype: u8) -> Float64Array {
    from_vec(ferro_ta_core::overlap::ma(&to_vec(close), timeperiod, matype))
}

/// Moving Average with Variable Period.
#[wasm_bindgen]
pub fn mavp(close: &Float64Array, periods: &Float64Array, minperiod: usize, maxperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::overlap::mavp(&to_vec(close), &to_vec(periods), minperiod, maxperiod))
}

// ===========================================================================
// Momentum (additional exports — new core indicators)
// ===========================================================================

/// Rate of Change: `(close[i] - close[i-p]) / close[i-p] * 100`.
#[wasm_bindgen]
pub fn roc(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::roc(&to_vec(close), timeperiod))
}

/// Rate of Change Percentage.
#[wasm_bindgen]
pub fn rocp(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::rocp(&to_vec(close), timeperiod))
}

/// Rate of Change Ratio.
#[wasm_bindgen]
pub fn rocr(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::rocr(&to_vec(close), timeperiod))
}

/// Rate of Change Ratio x 100.
#[wasm_bindgen]
pub fn rocr100(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::rocr100(&to_vec(close), timeperiod))
}

/// Williams %R.
#[wasm_bindgen]
pub fn willr(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::willr(&to_vec(high), &to_vec(low), &to_vec(close), timeperiod))
}

/// Aroon indicator. Returns [aroon_down, aroon_up].
#[wasm_bindgen]
pub fn aroon(high: &Float64Array, low: &Float64Array, timeperiod: usize) -> Array {
    let (down, up) = ferro_ta_core::momentum::aroon(&to_vec(high), &to_vec(low), timeperiod);
    let out = Array::new();
    out.push(&from_vec(down));
    out.push(&from_vec(up));
    out
}

/// Aroon Oscillator.
#[wasm_bindgen]
pub fn aroonosc(high: &Float64Array, low: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::aroonosc(&to_vec(high), &to_vec(low), timeperiod))
}

/// Commodity Channel Index.
#[wasm_bindgen]
pub fn cci(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::cci(&to_vec(high), &to_vec(low), &to_vec(close), timeperiod))
}

/// Balance of Power.
#[wasm_bindgen]
pub fn bop(open: &Float64Array, high: &Float64Array, low: &Float64Array, close: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::momentum::bop(&to_vec(open), &to_vec(high), &to_vec(low), &to_vec(close)))
}

/// Stochastic RSI. Returns [fastk, fastd].
#[wasm_bindgen]
pub fn stochrsi(close: &Float64Array, timeperiod: usize, fastk_period: usize, fastd_period: usize) -> Array {
    let (k, d) = ferro_ta_core::momentum::stochrsi(&to_vec(close), timeperiod, fastk_period, fastd_period);
    let out = Array::new();
    out.push(&from_vec(k));
    out.push(&from_vec(d));
    out
}

/// Absolute Price Oscillator.
#[wasm_bindgen]
pub fn apo(close: &Float64Array, fastperiod: usize, slowperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::apo(&to_vec(close), fastperiod, slowperiod))
}

/// Percentage Price Oscillator. Returns [ppo, signal, histogram].
#[wasm_bindgen]
pub fn ppo(close: &Float64Array, fastperiod: usize, slowperiod: usize, signalperiod: usize) -> Array {
    let (p, s, h) = ferro_ta_core::momentum::ppo(&to_vec(close), fastperiod, slowperiod, signalperiod);
    let out = Array::new();
    out.push(&from_vec(p));
    out.push(&from_vec(s));
    out.push(&from_vec(h));
    out
}

/// Chande Momentum Oscillator.
#[wasm_bindgen]
pub fn cmo(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::cmo(&to_vec(close), timeperiod))
}

/// TRIX: 1-period rate of change of triple-smoothed EMA.
#[wasm_bindgen]
pub fn trix_indicator(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::trix(&to_vec(close), timeperiod))
}

/// Ultimate Oscillator.
#[wasm_bindgen]
pub fn ultosc(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod1: usize, timeperiod2: usize, timeperiod3: usize) -> Float64Array {
    from_vec(ferro_ta_core::momentum::ultosc(&to_vec(high), &to_vec(low), &to_vec(close), timeperiod1, timeperiod2, timeperiod3))
}

// ===========================================================================
// Volatility (additional exports)
// ===========================================================================

/// True Range.
#[wasm_bindgen]
pub fn trange(high: &Float64Array, low: &Float64Array, close: &Float64Array) -> Float64Array {
    let h = to_vec(high);
    let l = to_vec(low);
    let c = to_vec(close);
    from_vec(ferro_ta_core::volatility::trange(&h, &l, &c))
}

/// Normalized Average True Range: ATR / close * 100.
#[wasm_bindgen]
pub fn natr(high: &Float64Array, low: &Float64Array, close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::volatility::natr(&to_vec(high), &to_vec(low), &to_vec(close), timeperiod))
}

// ===========================================================================
// Statistic (additional exports)
// ===========================================================================

/// Rolling population standard deviation scaled by `nbdev`.
#[wasm_bindgen]
pub fn stddev(close: &Float64Array, timeperiod: usize, nbdev: f64) -> Float64Array {
    from_vec(ferro_ta_core::statistic::stddev(&to_vec(close), timeperiod, nbdev))
}

/// Rolling population variance scaled by `nbdev²`.
#[wasm_bindgen]
pub fn var(close: &Float64Array, timeperiod: usize, nbdev: f64) -> Float64Array {
    from_vec(ferro_ta_core::statistic::var(&to_vec(close), timeperiod, nbdev))
}

/// Linear regression fitted value.
#[wasm_bindgen]
pub fn linearreg(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::statistic::linearreg(&to_vec(close), timeperiod))
}

/// Linear regression slope.
#[wasm_bindgen]
pub fn linearreg_slope(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::statistic::linearreg_slope(&to_vec(close), timeperiod))
}

/// Linear regression intercept.
#[wasm_bindgen]
pub fn linearreg_intercept(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::statistic::linearreg_intercept(&to_vec(close), timeperiod))
}

/// Linear regression angle in degrees.
#[wasm_bindgen]
pub fn linearreg_angle(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::statistic::linearreg_angle(&to_vec(close), timeperiod))
}

/// Time Series Forecast.
#[wasm_bindgen]
pub fn tsf(close: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::statistic::tsf(&to_vec(close), timeperiod))
}

/// Rolling beta (return-based regression).
#[wasm_bindgen]
pub fn beta_rolling(real0: &Float64Array, real1: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::statistic::beta(&to_vec(real0), &to_vec(real1), timeperiod))
}

/// Rolling Pearson correlation.
#[wasm_bindgen]
pub fn correl(real0: &Float64Array, real1: &Float64Array, timeperiod: usize) -> Float64Array {
    from_vec(ferro_ta_core::statistic::correl(&to_vec(real0), &to_vec(real1), timeperiod))
}

// ===========================================================================
// Streaming / Stateful API
// ===========================================================================

/// Streaming Simple Moving Average.
#[wasm_bindgen]
pub struct WasmStreamingSMA {
    inner: ferro_ta_core::streaming::StreamingSMA,
}

#[wasm_bindgen]
impl WasmStreamingSMA {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Result<WasmStreamingSMA, JsError> {
        let inner = ferro_ta_core::streaming::StreamingSMA::new(period)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, value: f64) -> f64 { self.inner.update(value) }
    pub fn reset(&mut self) { self.inner.reset(); }
    #[wasm_bindgen(getter)]
    pub fn period(&self) -> usize { self.inner.period() }
}

/// Streaming Exponential Moving Average.
#[wasm_bindgen]
pub struct WasmStreamingEMA {
    inner: ferro_ta_core::streaming::StreamingEMA,
}

#[wasm_bindgen]
impl WasmStreamingEMA {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Result<WasmStreamingEMA, JsError> {
        let inner = ferro_ta_core::streaming::StreamingEMA::new(period)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, value: f64) -> f64 { self.inner.update(value) }
    pub fn reset(&mut self) { self.inner.reset(); }
    #[wasm_bindgen(getter)]
    pub fn period(&self) -> usize { self.inner.period() }
}

/// Streaming Relative Strength Index.
#[wasm_bindgen]
pub struct WasmStreamingRSI {
    inner: ferro_ta_core::streaming::StreamingRSI,
}

#[wasm_bindgen]
impl WasmStreamingRSI {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Result<WasmStreamingRSI, JsError> {
        let inner = ferro_ta_core::streaming::StreamingRSI::new(period)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, value: f64) -> f64 { self.inner.update(value) }
    pub fn reset(&mut self) { self.inner.reset(); }
    #[wasm_bindgen(getter)]
    pub fn period(&self) -> usize { self.inner.period() }
}

/// Streaming Average True Range.
#[wasm_bindgen]
pub struct WasmStreamingATR {
    inner: ferro_ta_core::streaming::StreamingATR,
}

#[wasm_bindgen]
impl WasmStreamingATR {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Result<WasmStreamingATR, JsError> {
        let inner = ferro_ta_core::streaming::StreamingATR::new(period)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        self.inner.update(high, low, close)
    }
    pub fn reset(&mut self) { self.inner.reset(); }
    #[wasm_bindgen(getter)]
    pub fn period(&self) -> usize { self.inner.period() }
}

/// Streaming Bollinger Bands. Returns [upper, middle, lower] from `update()`.
#[wasm_bindgen]
pub struct WasmStreamingBBands {
    inner: ferro_ta_core::streaming::StreamingBBands,
}

#[wasm_bindgen]
impl WasmStreamingBBands {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize, nbdevup: f64, nbdevdn: f64) -> Result<WasmStreamingBBands, JsError> {
        let inner = ferro_ta_core::streaming::StreamingBBands::new(period, nbdevup, nbdevdn)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, value: f64) -> Array {
        let (u, m, l) = self.inner.update(value);
        let out = Array::new();
        out.push(&JsValue::from_f64(u));
        out.push(&JsValue::from_f64(m));
        out.push(&JsValue::from_f64(l));
        out
    }
    pub fn reset(&mut self) { self.inner.reset(); }
    #[wasm_bindgen(getter)]
    pub fn period(&self) -> usize { self.inner.period() }
}

/// Streaming MACD. Returns [macd, signal, histogram] from `update()`.
#[wasm_bindgen]
pub struct WasmStreamingMACD {
    inner: ferro_ta_core::streaming::StreamingMACD,
}

#[wasm_bindgen]
impl WasmStreamingMACD {
    #[wasm_bindgen(constructor)]
    pub fn new(fastperiod: usize, slowperiod: usize, signalperiod: usize) -> Result<WasmStreamingMACD, JsError> {
        let inner = ferro_ta_core::streaming::StreamingMACD::new(fastperiod, slowperiod, signalperiod)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, value: f64) -> Array {
        let (m, s, h) = self.inner.update(value);
        let out = Array::new();
        out.push(&JsValue::from_f64(m));
        out.push(&JsValue::from_f64(s));
        out.push(&JsValue::from_f64(h));
        out
    }
    pub fn reset(&mut self) { self.inner.reset(); }
    #[wasm_bindgen(getter)]
    pub fn fast_period(&self) -> usize { self.inner.fast_period() }
    #[wasm_bindgen(getter)]
    pub fn slow_period(&self) -> usize { self.inner.slow_period() }
    #[wasm_bindgen(getter)]
    pub fn signal_period(&self) -> usize { self.inner.signal_period() }
}

/// Streaming Stochastic Oscillator. Returns [slowk, slowd] from `update()`.
#[wasm_bindgen]
pub struct WasmStreamingStoch {
    inner: ferro_ta_core::streaming::StreamingStoch,
}

#[wasm_bindgen]
impl WasmStreamingStoch {
    #[wasm_bindgen(constructor)]
    pub fn new(fastk_period: usize, slowk_period: usize, slowd_period: usize) -> Result<WasmStreamingStoch, JsError> {
        let inner = ferro_ta_core::streaming::StreamingStoch::new(fastk_period, slowk_period, slowd_period)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Array {
        let (sk, sd) = self.inner.update(high, low, close);
        let out = Array::new();
        out.push(&JsValue::from_f64(sk));
        out.push(&JsValue::from_f64(sd));
        out
    }
    pub fn reset(&mut self) { self.inner.reset(); }
    #[wasm_bindgen(getter)]
    pub fn period(&self) -> usize { self.inner.period() }
}

/// Streaming cumulative VWAP.
#[wasm_bindgen]
pub struct WasmStreamingVWAP {
    inner: ferro_ta_core::streaming::StreamingVWAP,
}

#[wasm_bindgen]
impl WasmStreamingVWAP {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmStreamingVWAP {
        Self { inner: ferro_ta_core::streaming::StreamingVWAP::new() }
    }
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        self.inner.update(high, low, close, volume)
    }
    pub fn reset(&mut self) { self.inner.reset(); }
}

/// Streaming Supertrend. Returns [line, direction] from `update()`.
#[wasm_bindgen]
pub struct WasmStreamingSupertrend {
    inner: ferro_ta_core::streaming::StreamingSupertrend,
}

#[wasm_bindgen]
impl WasmStreamingSupertrend {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize, multiplier: f64) -> Result<WasmStreamingSupertrend, JsError> {
        let inner = ferro_ta_core::streaming::StreamingSupertrend::new(period, multiplier)
            .map_err(|e| JsError::new(&e.0))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Array {
        let (line, dir) = self.inner.update(high, low, close);
        let out = Array::new();
        out.push(&JsValue::from_f64(line));
        out.push(&JsValue::from_f64(dir as f64));
        out
    }
    pub fn reset(&mut self) { self.inner.reset(); }
    #[wasm_bindgen(getter)]
    pub fn period(&self) -> usize { self.inner.period() }
}

// ===========================================================================
// Batch Operations
// ===========================================================================

/// Convert a js_sys::Array of Float64Array into Vec<Vec<f64>>.
fn array_of_f64arr_to_vecs(arr: &Array) -> Vec<Vec<f64>> {
    (0..arr.length())
        .map(|i| {
            let item: Float64Array = arr.get(i).unchecked_into();
            to_vec(&item)
        })
        .collect()
}

/// Convert Vec<Vec<f64>> into a js_sys::Array of Float64Array.
fn vecs_to_array_of_f64arr(data: Vec<Vec<f64>>) -> Array {
    let out = Array::new();
    for v in data {
        out.push(&from_vec(v));
    }
    out
}

/// Batch SMA: compute SMA on each column of 2D data.
#[wasm_bindgen]
pub fn batch_sma(data: &Array, timeperiod: usize) -> Array {
    let vecs = array_of_f64arr_to_vecs(data);
    match ferro_ta_core::batch::batch_sma(&vecs, timeperiod) {
        Ok(r) => vecs_to_array_of_f64arr(r),
        Err(_) => Array::new(),
    }
}

/// Batch EMA: compute EMA on each column of 2D data.
#[wasm_bindgen]
pub fn batch_ema(data: &Array, timeperiod: usize) -> Array {
    let vecs = array_of_f64arr_to_vecs(data);
    match ferro_ta_core::batch::batch_ema(&vecs, timeperiod) {
        Ok(r) => vecs_to_array_of_f64arr(r),
        Err(_) => Array::new(),
    }
}

/// Batch RSI: compute RSI on each column of 2D data.
#[wasm_bindgen]
pub fn batch_rsi(data: &Array, timeperiod: usize) -> Array {
    let vecs = array_of_f64arr_to_vecs(data);
    match ferro_ta_core::batch::batch_rsi(&vecs, timeperiod) {
        Ok(r) => vecs_to_array_of_f64arr(r),
        Err(_) => Array::new(),
    }
}

// ===========================================================================
// Portfolio (additional exports)
// ===========================================================================

/// Portfolio volatility: sqrt(w' * cov * w).
#[wasm_bindgen]
pub fn portfolio_volatility(cov_matrix: &Array, weights: &Float64Array) -> f64 {
    let cov = array_of_f64arr_to_vecs(cov_matrix);
    let w = to_vec(weights);
    ferro_ta_core::portfolio::portfolio_volatility(&cov, &w)
}

/// Pairwise correlation matrix.
#[wasm_bindgen]
pub fn correlation_matrix(data: &Array) -> Array {
    let vecs = array_of_f64arr_to_vecs(data);
    vecs_to_array_of_f64arr(ferro_ta_core::portfolio::correlation_matrix(&vecs))
}

/// Weighted composite of multiple series.
#[wasm_bindgen]
pub fn compose_weighted(data: &Array, weights: &Float64Array) -> Float64Array {
    let vecs = array_of_f64arr_to_vecs(data);
    let w = to_vec(weights);
    from_vec(ferro_ta_core::portfolio::compose_weighted(&vecs, &w))
}

// ===========================================================================
// Crypto (additional exports)
// ===========================================================================

/// Mark session boundaries from nanosecond timestamps.
#[wasm_bindgen]
pub fn mark_session_boundaries(timestamps_ns: &Float64Array) -> Float64Array {
    let ts: Vec<i64> = to_vec(timestamps_ns).iter().map(|&v| v as i64).collect();
    let result = ferro_ta_core::crypto::mark_session_boundaries(&ts);
    from_vec(result.iter().map(|&v| v as f64).collect())
}

// ===========================================================================
// Chunked (additional exports)
// ===========================================================================

/// Stitch multiple chunks into a single array.
#[wasm_bindgen]
pub fn stitch_chunks(chunks: &Array) -> Float64Array {
    let vecs = array_of_f64arr_to_vecs(chunks);
    let slices: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    from_vec(ferro_ta_core::chunked::stitch_chunks(&slices))
}

// ===========================================================================
// Math Operators & Transforms
// ===========================================================================

/// Element-wise addition.
#[wasm_bindgen]
pub fn math_add(a: &Float64Array, b: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::math::add(&to_vec(a), &to_vec(b)))
}

/// Element-wise subtraction.
#[wasm_bindgen]
pub fn math_sub(a: &Float64Array, b: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::math::sub(&to_vec(a), &to_vec(b)))
}

/// Element-wise multiplication.
#[wasm_bindgen]
pub fn math_mult(a: &Float64Array, b: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::math::mult(&to_vec(a), &to_vec(b)))
}

/// Element-wise division.
#[wasm_bindgen]
pub fn math_div(a: &Float64Array, b: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::math::div(&to_vec(a), &to_vec(b)))
}

macro_rules! math_transform_wrapper {
    ($wasm_name:ident, $core_name:ident) => {
        #[wasm_bindgen]
        pub fn $wasm_name(real: &Float64Array) -> Float64Array {
            from_vec(ferro_ta_core::math::$core_name(&to_vec(real)))
        }
    };
}

math_transform_wrapper!(transform_acos, math_acos);
math_transform_wrapper!(transform_asin, math_asin);
math_transform_wrapper!(transform_atan, math_atan);
math_transform_wrapper!(transform_ceil, math_ceil);
math_transform_wrapper!(transform_cos, math_cos);
math_transform_wrapper!(transform_cosh, math_cosh);
math_transform_wrapper!(transform_exp, math_exp);
math_transform_wrapper!(transform_floor, math_floor);
math_transform_wrapper!(transform_ln, math_ln);
math_transform_wrapper!(transform_log10, math_log10);
math_transform_wrapper!(transform_sin, math_sin);
math_transform_wrapper!(transform_sinh, math_sinh);
math_transform_wrapper!(transform_sqrt, math_sqrt);
math_transform_wrapper!(transform_tan, math_tan);
math_transform_wrapper!(transform_tanh, math_tanh);

// ===========================================================================
// Candlestick Patterns (61 functions via macro)
// ===========================================================================

/// Convert a `Vec<i32>` into a `js_sys::Int32Array`.
fn from_i32_vec(v: Vec<i32>) -> js_sys::Int32Array {
    let arr = js_sys::Int32Array::new_with_length(v.len() as u32);
    arr.copy_from(&v);
    arr
}

macro_rules! cdl_wrapper {
    ($($name:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        pub fn $name(
            open: &Float64Array,
            high: &Float64Array,
            low: &Float64Array,
            close: &Float64Array,
        ) -> js_sys::Int32Array {
            let o = to_vec(open);
            let h = to_vec(high);
            let l = to_vec(low);
            let c = to_vec(close);
            from_i32_vec(ferro_ta_core::pattern::$name(&o, &h, &l, &c))
        }
    )*};
}

cdl_wrapper!(
    cdl2crows,
    cdl3blackcrows,
    cdl3inside,
    cdl3linestrike,
    cdl3outside,
    cdl3starsinsouth,
    cdl3whitesoldiers,
    cdlabandonedbaby,
    cdladvanceblock,
    cdlbelthold,
    cdlbreakaway,
    cdlclosingmarubozu,
    cdlconcealbabyswall,
    cdlcounterattack,
    cdldarkcloudcover,
    cdldoji,
    cdldojistar,
    cdldragonflydoji,
    cdlengulfing,
    cdleveningdojistar,
    cdleveningstar,
    cdlgapsidesidewhite,
    cdlgravestonedoji,
    cdlhammer,
    cdlhangingman,
    cdlharami,
    cdlharamicross,
    cdlhighwave,
    cdlhikkake,
    cdlhikkakemod,
    cdlhomingpigeon,
    cdlidentical3crows,
    cdlinneck,
    cdlinvertedhammer,
    cdlkicking,
    cdlkickingbylength,
    cdlladderbottom,
    cdllongleggeddoji,
    cdllongline,
    cdlmarubozu,
    cdlmatchinglow,
    cdlmathold,
    cdlmorningdojistar,
    cdlmorningstar,
    cdlonneck,
    cdlpiercing,
    cdlrickshawman,
    cdlrisefall3methods,
    cdlseparatinglines,
    cdlshootingstar,
    cdlshortline,
    cdlspinningtop,
    cdlstalledpattern,
    cdlsticksandwich,
    cdltakuri,
    cdltasukigap,
    cdlthrusting,
    cdltristar,
    cdlunique3river,
    cdlupsidegap2crows,
    cdlxsidegap3methods,
);

// ===========================================================================
// Signals (additional)
// ===========================================================================

/// Rank values (percentile ranking [0, 100]).
#[wasm_bindgen]
pub fn rank_values(x: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::signals::rank_values(&to_vec(x)))
}

/// Composite rank across multiple signal arrays.
#[wasm_bindgen]
pub fn compose_rank(signals: &Array) -> Float64Array {
    let vecs = array_of_f64arr_to_vecs(signals);
    let slices: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    from_vec(ferro_ta_core::signals::compose_rank(&slices))
}

// ===========================================================================
// Batch (additional)
// ===========================================================================

/// Batch ATR across multiple HLC column sets.
#[wasm_bindgen]
pub fn batch_atr(high: &Array, low: &Array, close: &Array, timeperiod: usize) -> Array {
    let h = array_of_f64arr_to_vecs(high);
    let l = array_of_f64arr_to_vecs(low);
    let c = array_of_f64arr_to_vecs(close);
    match ferro_ta_core::batch::batch_atr(&h, &l, &c, timeperiod) {
        Ok(r) => vecs_to_array_of_f64arr(r),
        Err(_) => Array::new(),
    }
}

/// Batch Stochastic across multiple HLC column sets. Returns [Array[slowk_cols], Array[slowd_cols]].
#[wasm_bindgen]
pub fn batch_stoch(high: &Array, low: &Array, close: &Array, fastk_period: usize, slowk_period: usize, slowd_period: usize) -> Array {
    let h = array_of_f64arr_to_vecs(high);
    let l = array_of_f64arr_to_vecs(low);
    let c = array_of_f64arr_to_vecs(close);
    match ferro_ta_core::batch::batch_stoch(&h, &l, &c, fastk_period, slowk_period, slowd_period) {
        Ok((sk, sd)) => {
            let out = Array::new();
            out.push(&vecs_to_array_of_f64arr(sk));
            out.push(&vecs_to_array_of_f64arr(sd));
            out
        }
        Err(_) => Array::new(),
    }
}

/// Batch ADX across multiple HLC column sets.
#[wasm_bindgen]
pub fn batch_adx(high: &Array, low: &Array, close: &Array, timeperiod: usize) -> Array {
    let h = array_of_f64arr_to_vecs(high);
    let l = array_of_f64arr_to_vecs(low);
    let c = array_of_f64arr_to_vecs(close);
    match ferro_ta_core::batch::batch_adx(&h, &l, &c, timeperiod) {
        Ok(r) => vecs_to_array_of_f64arr(r),
        Err(_) => Array::new(),
    }
}

// ===========================================================================
// Options Analytics
// ===========================================================================

fn parse_option_kind(kind: &str) -> ferro_ta_core::options::OptionKind {
    match kind.to_lowercase().as_str() {
        "put" | "p" => ferro_ta_core::options::OptionKind::Put,
        _ => ferro_ta_core::options::OptionKind::Call,
    }
}

fn parse_pricing_model(model: &str) -> ferro_ta_core::options::PricingModel {
    match model.to_lowercase().as_str() {
        "black76" | "b76" => ferro_ta_core::options::PricingModel::Black76,
        _ => ferro_ta_core::options::PricingModel::BlackScholes,
    }
}

/// Black-Scholes-Merton option price.
#[wasm_bindgen]
pub fn black_scholes_price(
    spot: f64, strike: f64, rate: f64, dividend_yield: f64,
    time_to_expiry: f64, volatility: f64, kind: &str,
) -> f64 {
    ferro_ta_core::options::pricing::black_scholes_price(
        spot, strike, rate, dividend_yield, time_to_expiry, volatility, parse_option_kind(kind),
    )
}

/// Black-76 option price (futures).
#[wasm_bindgen]
pub fn black_76_price(
    forward: f64, strike: f64, rate: f64,
    time_to_expiry: f64, volatility: f64, kind: &str,
) -> f64 {
    ferro_ta_core::options::pricing::black_76_price(
        forward, strike, rate, time_to_expiry, volatility, parse_option_kind(kind),
    )
}

/// Black-Scholes Greeks. Returns [delta, gamma, vega, theta, rho].
#[wasm_bindgen]
pub fn black_scholes_greeks(
    spot: f64, strike: f64, rate: f64, dividend_yield: f64,
    time_to_expiry: f64, volatility: f64, kind: &str,
) -> Array {
    let g = ferro_ta_core::options::greeks::black_scholes_greeks(
        spot, strike, rate, dividend_yield, time_to_expiry, volatility, parse_option_kind(kind),
    );
    let out = Array::new();
    out.push(&JsValue::from_f64(g.delta));
    out.push(&JsValue::from_f64(g.gamma));
    out.push(&JsValue::from_f64(g.vega));
    out.push(&JsValue::from_f64(g.theta));
    out.push(&JsValue::from_f64(g.rho));
    out
}

/// Black-76 Greeks. Returns [delta, gamma, vega, theta, rho].
#[wasm_bindgen]
pub fn black_76_greeks(
    forward: f64, strike: f64, rate: f64,
    time_to_expiry: f64, volatility: f64, kind: &str,
) -> Array {
    let g = ferro_ta_core::options::greeks::black_76_greeks(
        forward, strike, rate, time_to_expiry, volatility, parse_option_kind(kind),
    );
    let out = Array::new();
    out.push(&JsValue::from_f64(g.delta));
    out.push(&JsValue::from_f64(g.gamma));
    out.push(&JsValue::from_f64(g.vega));
    out.push(&JsValue::from_f64(g.theta));
    out.push(&JsValue::from_f64(g.rho));
    out
}

/// Implied volatility via Newton-Raphson.
#[wasm_bindgen]
pub fn implied_volatility(
    model: &str, underlying: f64, strike: f64, rate: f64, carry: f64,
    time_to_expiry: f64, kind: &str, target_price: f64,
    initial_guess: f64, tolerance: f64, max_iterations: usize,
) -> f64 {
    use ferro_ta_core::options::*;
    let contract = OptionContract {
        model: parse_pricing_model(model),
        underlying, strike, rate, carry, time_to_expiry,
        kind: parse_option_kind(kind),
    };
    let config = IvSolverConfig { initial_guess, tolerance, max_iterations };
    iv::implied_volatility(contract, target_price, config)
}

/// IV Rank over a rolling window.
#[wasm_bindgen]
pub fn iv_rank(iv_series: &Float64Array, window: usize) -> Float64Array {
    from_vec(ferro_ta_core::options::iv::iv_rank(&to_vec(iv_series), window))
}

/// IV Percentile over a rolling window.
#[wasm_bindgen]
pub fn iv_percentile(iv_series: &Float64Array, window: usize) -> Float64Array {
    from_vec(ferro_ta_core::options::iv::iv_percentile(&to_vec(iv_series), window))
}

/// IV Z-Score over a rolling window.
#[wasm_bindgen]
pub fn iv_zscore(iv_series: &Float64Array, window: usize) -> Float64Array {
    from_vec(ferro_ta_core::options::iv::iv_zscore(&to_vec(iv_series), window))
}

/// ATM index in a strikes array.
#[wasm_bindgen]
pub fn atm_index(strikes: &Float64Array, reference_price: f64) -> f64 {
    match ferro_ta_core::options::chain::atm_index(&to_vec(strikes), reference_price) {
        Some(idx) => idx as f64,
        None => f64::NAN,
    }
}

/// Label moneyness of strikes. Returns Int8Array.
#[wasm_bindgen]
pub fn label_moneyness(strikes: &Float64Array, reference_price: f64, kind: &str) -> js_sys::Int8Array {
    let result = ferro_ta_core::options::chain::label_moneyness(
        &to_vec(strikes), reference_price, parse_option_kind(kind),
    );
    let arr = js_sys::Int8Array::new_with_length(result.len() as u32);
    arr.copy_from(&result);
    arr
}

/// Model-dispatched option price (model: "bs" or "b76").
#[wasm_bindgen]
pub fn model_price(
    model: &str, underlying: f64, strike: f64, rate: f64, carry: f64,
    time_to_expiry: f64, volatility: f64, kind: &str,
) -> f64 {
    use ferro_ta_core::options::*;
    let input = OptionEvaluation {
        contract: OptionContract {
            model: parse_pricing_model(model), underlying, strike, rate, carry, time_to_expiry,
            kind: parse_option_kind(kind),
        },
        volatility,
    };
    pricing::model_price(input)
}

/// Model-dispatched Greeks. Returns [delta, gamma, vega, theta, rho].
#[wasm_bindgen]
pub fn model_greeks(
    model: &str, underlying: f64, strike: f64, rate: f64, carry: f64,
    time_to_expiry: f64, volatility: f64, kind: &str,
) -> Array {
    use ferro_ta_core::options::*;
    let input = OptionEvaluation {
        contract: OptionContract {
            model: parse_pricing_model(model), underlying, strike, rate, carry, time_to_expiry,
            kind: parse_option_kind(kind),
        },
        volatility,
    };
    let g = greeks::model_greeks(input);
    let out = Array::new();
    out.push(&JsValue::from_f64(g.delta));
    out.push(&JsValue::from_f64(g.gamma));
    out.push(&JsValue::from_f64(g.vega));
    out.push(&JsValue::from_f64(g.theta));
    out.push(&JsValue::from_f64(g.rho));
    out
}

/// Model theta (numerical).
#[wasm_bindgen]
pub fn model_theta(
    model: &str, underlying: f64, strike: f64, rate: f64, carry: f64,
    time_to_expiry: f64, volatility: f64, kind: &str,
) -> f64 {
    use ferro_ta_core::options::*;
    let input = OptionEvaluation {
        contract: OptionContract {
            model: parse_pricing_model(model), underlying, strike, rate, carry, time_to_expiry,
            kind: parse_option_kind(kind),
        },
        volatility,
    };
    greeks::model_theta(input)
}

/// Price lower bound.
#[wasm_bindgen]
pub fn price_lower_bound(
    model: &str, underlying: f64, strike: f64, rate: f64, carry: f64,
    time_to_expiry: f64, kind: &str,
) -> f64 {
    use ferro_ta_core::options::*;
    let contract = OptionContract {
        model: parse_pricing_model(model), underlying, strike, rate, carry, time_to_expiry,
        kind: parse_option_kind(kind),
    };
    pricing::price_lower_bound(contract)
}

/// Price upper bound.
#[wasm_bindgen]
pub fn price_upper_bound(
    model: &str, underlying: f64, strike: f64, rate: f64, carry: f64,
    time_to_expiry: f64, kind: &str,
) -> f64 {
    use ferro_ta_core::options::*;
    let contract = OptionContract {
        model: parse_pricing_model(model), underlying, strike, rate, carry, time_to_expiry,
        kind: parse_option_kind(kind),
    };
    pricing::price_upper_bound(contract)
}

/// Select strike by offset from ATM.
#[wasm_bindgen]
pub fn select_strike_by_offset(strikes: &Float64Array, reference_price: f64, offset: i32) -> f64 {
    match ferro_ta_core::options::chain::select_strike_by_offset(
        &to_vec(strikes), reference_price, offset as isize,
    ) {
        Some(v) => v,
        None => f64::NAN,
    }
}

/// Smile metrics. Returns [atm_iv, risk_reversal_25d, butterfly_25d, skew_slope, convexity].
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn smile_metrics(
    strikes: &Float64Array, vols: &Float64Array, reference_price: f64,
    rate: f64, carry: f64, time_to_expiry: f64, model: &str,
) -> Array {
    let m = ferro_ta_core::options::surface::smile_metrics(
        &to_vec(strikes), &to_vec(vols), reference_price,
        rate, carry, time_to_expiry, parse_pricing_model(model),
    );
    let out = Array::new();
    out.push(&JsValue::from_f64(m.atm_iv));
    out.push(&JsValue::from_f64(m.risk_reversal_25d));
    out.push(&JsValue::from_f64(m.butterfly_25d));
    out.push(&JsValue::from_f64(m.skew_slope));
    out.push(&JsValue::from_f64(m.convexity));
    out
}

/// Linear interpolation helper.
#[wasm_bindgen]
pub fn linear_interpolate(xs: &Float64Array, ys: &Float64Array, target: f64) -> f64 {
    ferro_ta_core::options::surface::linear_interpolate(&to_vec(xs), &to_vec(ys), target)
}

/// Select strike by delta target.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn select_strike_by_delta(
    strikes: &Float64Array, vols: &Float64Array,
    model: &str, reference_price: f64, rate: f64, carry: f64,
    time_to_expiry: f64, kind: &str, target_delta: f64,
) -> f64 {
    use ferro_ta_core::options::*;
    let ctx = ChainGreeksContext {
        model: parse_pricing_model(model),
        reference_price, rate, carry, time_to_expiry,
        kind: parse_option_kind(kind),
    };
    match chain::select_strike_by_delta(&to_vec(strikes), &to_vec(vols), ctx, target_delta) {
        Some(v) => v,
        None => f64::NAN,
    }
}

/// ATM implied volatility interpolated from strikes/vols.
#[wasm_bindgen]
pub fn atm_iv(strikes: &Float64Array, vols: &Float64Array, reference_price: f64) -> f64 {
    ferro_ta_core::options::surface::atm_iv(&to_vec(strikes), &to_vec(vols), reference_price)
}

/// Term structure slope.
#[wasm_bindgen]
pub fn term_structure_slope(tenors: &Float64Array, atm_ivs: &Float64Array) -> f64 {
    ferro_ta_core::options::surface::term_structure_slope(&to_vec(tenors), &to_vec(atm_ivs))
}

// ===========================================================================
// Futures Analytics
// ===========================================================================

/// Futures basis: future - spot.
#[wasm_bindgen]
pub fn futures_basis(spot: f64, future: f64) -> f64 {
    ferro_ta_core::futures::basis::basis(spot, future)
}

/// Annualized basis.
#[wasm_bindgen]
pub fn annualized_basis(spot: f64, future: f64, time_to_expiry: f64) -> f64 {
    ferro_ta_core::futures::basis::annualized_basis(spot, future, time_to_expiry)
}

/// Implied carry rate.
#[wasm_bindgen]
pub fn implied_carry_rate(spot: f64, future: f64, time_to_expiry: f64) -> f64 {
    ferro_ta_core::futures::basis::implied_carry_rate(spot, future, time_to_expiry)
}

/// Carry spread.
#[wasm_bindgen]
pub fn carry_spread(spot: f64, future: f64, rate: f64, time_to_expiry: f64) -> f64 {
    ferro_ta_core::futures::basis::carry_spread(spot, future, rate, time_to_expiry)
}

/// Calendar spreads between consecutive futures prices.
#[wasm_bindgen]
pub fn calendar_spreads(futures_prices: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::futures::curve::calendar_spreads(&to_vec(futures_prices)))
}

/// Curve slope (linear regression).
#[wasm_bindgen]
pub fn curve_slope(tenors: &Float64Array, futures_prices: &Float64Array) -> f64 {
    ferro_ta_core::futures::curve::curve_slope(&to_vec(tenors), &to_vec(futures_prices))
}

/// Curve summary. Returns [front_basis, average_basis, slope, is_contango (1.0 or 0.0)].
#[wasm_bindgen]
pub fn curve_summary(spot: f64, tenors: &Float64Array, futures_prices: &Float64Array) -> Array {
    let s = ferro_ta_core::futures::curve::curve_summary(spot, &to_vec(tenors), &to_vec(futures_prices));
    let out = Array::new();
    out.push(&JsValue::from_f64(s.front_basis));
    out.push(&JsValue::from_f64(s.average_basis));
    out.push(&JsValue::from_f64(s.slope));
    out.push(&JsValue::from_f64(if s.is_contango { 1.0 } else { 0.0 }));
    out
}

/// Roll yield.
#[wasm_bindgen]
pub fn roll_yield(front_price: f64, next_price: f64, time_to_expiry: f64) -> f64 {
    ferro_ta_core::futures::roll::roll_yield(front_price, next_price, time_to_expiry)
}

/// Weighted continuous contract.
#[wasm_bindgen]
pub fn weighted_continuous(front: &Float64Array, next: &Float64Array, next_weights: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::futures::roll::weighted_continuous(&to_vec(front), &to_vec(next), &to_vec(next_weights)))
}

/// Back-adjusted continuous contract.
#[wasm_bindgen]
pub fn back_adjusted_continuous(front: &Float64Array, next: &Float64Array, next_weights: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::futures::roll::back_adjusted_continuous(&to_vec(front), &to_vec(next), &to_vec(next_weights)))
}

/// Ratio-adjusted continuous contract.
#[wasm_bindgen]
pub fn ratio_adjusted_continuous(front: &Float64Array, next: &Float64Array, next_weights: &Float64Array) -> Float64Array {
    from_vec(ferro_ta_core::futures::roll::ratio_adjusted_continuous(&to_vec(front), &to_vec(next), &to_vec(next_weights)))
}

/// Synthetic forward price from put-call parity.
#[wasm_bindgen]
pub fn synthetic_forward(call_price: f64, put_price: f64, strike: f64, rate: f64, time_to_expiry: f64) -> f64 {
    ferro_ta_core::futures::synthetic::synthetic_forward(call_price, put_price, strike, rate, time_to_expiry)
}

/// Synthetic spot implied by put-call parity.
#[wasm_bindgen]
pub fn synthetic_spot(call_price: f64, put_price: f64, strike: f64, rate: f64, carry: f64, time_to_expiry: f64) -> f64 {
    ferro_ta_core::futures::synthetic::synthetic_spot(call_price, put_price, strike, rate, carry, time_to_expiry)
}

/// Put-call parity residual.
#[wasm_bindgen]
pub fn parity_gap(call_price: f64, put_price: f64, spot: f64, strike: f64, rate: f64, carry: f64, time_to_expiry: f64) -> f64 {
    ferro_ta_core::futures::synthetic::parity_gap(call_price, put_price, spot, strike, rate, carry, time_to_expiry)
}

// ===========================================================================
// Backtesting (signal generators + utilities)
// ===========================================================================

/// Backtest core: close-only vectorized backtest. Returns [positions, bar_returns, strategy_returns, equity].
#[wasm_bindgen]
pub fn backtest_core(
    close: &Float64Array, signals: &Float64Array,
    slippage_bps: f64, initial_capital: f64, commission_per_trade: f64,
) -> Array {
    match ferro_ta_core::backtest::backtest_core(
        &to_vec(close), &to_vec(signals), None, slippage_bps, initial_capital, commission_per_trade,
    ) {
        Ok(result) => {
            let out = Array::new();
            out.push(&from_vec(result.positions));
            out.push(&from_vec(result.bar_returns));
            out.push(&from_vec(result.strategy_returns));
            out.push(&from_vec(result.equity));
            out
        }
        Err(_) => Array::new(),
    }
}

/// Simple single-asset backtest. Returns [positions, strategy_returns, equity].
#[wasm_bindgen]
pub fn single_asset_backtest(
    close: &Float64Array, signals: &Float64Array,
    commission_per_trade: f64, slippage_bps: f64,
) -> Array {
    let (pos, strat_ret, eq) = ferro_ta_core::backtest::single_asset_backtest(
        &to_vec(close), &to_vec(signals), commission_per_trade, slippage_bps,
    );
    let out = Array::new();
    out.push(&from_vec(pos));
    out.push(&from_vec(strat_ret));
    out.push(&from_vec(eq));
    out
}

/// Walk-forward train/test indices. Returns flat array [train_start, train_end, test_start, test_end, ...].
#[wasm_bindgen]
pub fn walk_forward_indices(
    n_bars: usize, train_bars: usize, test_bars: usize, anchored: bool, step_bars: usize,
) -> Float64Array {
    match ferro_ta_core::backtest::walk_forward_indices(n_bars, train_bars, test_bars, anchored, step_bars) {
        Ok(indices) => {
            let flat: Vec<f64> = indices.iter()
                .flat_map(|fold| vec![fold[0] as f64, fold[1] as f64, fold[2] as f64, fold[3] as f64])
                .collect();
            from_vec(flat)
        }
        Err(_) => from_vec(vec![]),
    }
}

/// Monte Carlo bootstrap of strategy returns. Returns Array of Float64Array (one per simulation).
#[wasm_bindgen]
pub fn monte_carlo_bootstrap(
    strategy_returns: &Float64Array, n_sims: usize, seed: f64, block_size: usize,
) -> Array {
    match ferro_ta_core::backtest::monte_carlo_bootstrap(
        &to_vec(strategy_returns), n_sims, seed as u64, block_size,
    ) {
        Ok(sims) => vecs_to_array_of_f64arr(sims),
        Err(_) => Array::new(),
    }
}

/// Kelly fraction.
#[wasm_bindgen]
pub fn kelly_fraction(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
    ferro_ta_core::backtest::kelly_fraction(win_rate, avg_win, avg_loss).unwrap_or(f64::NAN)
}

/// Half-Kelly fraction.
#[wasm_bindgen]
pub fn half_kelly_fraction(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
    ferro_ta_core::backtest::half_kelly_fraction(win_rate, avg_win, avg_loss).unwrap_or(f64::NAN)
}

/// Compute performance metrics from strategy returns and equity.
/// Returns Float64Array with 22 metrics in order:
/// [total_return, cagr, annualized_vol, sharpe, sortino, calmar, max_drawdown,
///  avg_drawdown, max_dd_duration, avg_dd_duration, ulcer_index, omega_ratio,
///  win_rate, profit_factor, r_expectancy, avg_win, avg_loss, tail_ratio,
///  skewness, kurtosis, best_bar, worst_bar]
#[wasm_bindgen]
pub fn compute_performance_metrics(
    strategy_returns: &Float64Array, equity: &Float64Array,
    periods_per_year: f64, risk_free_rate: f64,
) -> Float64Array {
    match ferro_ta_core::backtest::compute_performance_metrics(
        &to_vec(strategy_returns), &to_vec(equity), periods_per_year, risk_free_rate, None,
    ) {
        Ok(m) => from_vec(vec![
            m.total_return, m.cagr, m.annualized_vol, m.sharpe, m.sortino, m.calmar,
            m.max_drawdown, m.avg_drawdown, m.max_drawdown_duration_bars as f64,
            m.avg_drawdown_duration_bars, m.ulcer_index, m.omega_ratio,
            m.win_rate, m.profit_factor, m.r_expectancy, m.avg_win, m.avg_loss,
            m.tail_ratio, m.skewness, m.kurtosis, m.best_bar, m.worst_bar,
        ]),
        Err(_) => from_vec(vec![]),
    }
}

/// OHLCV-aware backtest. Returns [positions, fill_prices, bar_returns, strategy_returns, equity].
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn backtest_ohlcv(
    open: &Float64Array, high: &Float64Array, low: &Float64Array, close: &Float64Array,
    signals: &Float64Array, slippage_bps: f64, initial_capital: f64, commission_per_trade: f64,
    stop_loss_pct: f64, take_profit_pct: f64, trailing_stop_pct: f64, max_hold_bars: usize,
) -> Array {
    let mut config = ferro_ta_core::backtest::BacktestConfig::default();
    config.slippage_bps = slippage_bps;
    config.initial_capital = initial_capital;
    config.commission_per_trade = commission_per_trade;
    config.stop_loss_pct = stop_loss_pct;
    config.take_profit_pct = take_profit_pct;
    config.trailing_stop_pct = trailing_stop_pct;
    config.max_hold_bars = max_hold_bars;
    match ferro_ta_core::backtest::backtest_ohlcv_core(
        &to_vec(open), &to_vec(high), &to_vec(low), &to_vec(close),
        &to_vec(signals), &config, None,
    ) {
        Ok(r) => {
            let out = Array::new();
            out.push(&from_vec(r.positions));
            out.push(&from_vec(r.fill_prices));
            out.push(&from_vec(r.bar_returns));
            out.push(&from_vec(r.strategy_returns));
            out.push(&from_vec(r.equity));
            out
        }
        Err(_) => Array::new(),
    }
}

/// RSI threshold signals.
#[wasm_bindgen]
pub fn rsi_threshold_signals(close: &Float64Array, timeperiod: usize, oversold: f64, overbought: f64) -> Float64Array {
    from_vec(ferro_ta_core::backtest::rsi_threshold_signals(&to_vec(close), timeperiod, oversold, overbought))
}

/// SMA crossover signals.
#[wasm_bindgen]
pub fn sma_crossover_signals(close: &Float64Array, fast: usize, slow: usize) -> Float64Array {
    match ferro_ta_core::backtest::sma_crossover_signals(&to_vec(close), fast, slow) {
        Ok(v) => from_vec(v),
        Err(_) => from_vec(vec![f64::NAN; close.length() as usize]),
    }
}

/// MACD crossover signals.
#[wasm_bindgen]
pub fn macd_crossover_signals(close: &Float64Array, fastperiod: usize, slowperiod: usize, signalperiod: usize) -> Float64Array {
    match ferro_ta_core::backtest::macd_crossover_signals(&to_vec(close), fastperiod, slowperiod, signalperiod) {
        Ok(v) => from_vec(v),
        Err(_) => from_vec(vec![f64::NAN; close.length() as usize]),
    }
}

// ===========================================================================
// New Options Features (extended Greeks, digital, American, vol estimators,
//   vol cone, expected move, put-call parity, strategy payoff/value/Greeks)
// ===========================================================================

// ---------------------------------------------------------------------------
// Helpers shared by the new features
// ---------------------------------------------------------------------------

fn parse_digital_kind(digital_type: &str) -> ferro_ta_core::options::digital::DigitalKind {
    match digital_type.to_ascii_lowercase().as_str() {
        "asset_or_nothing" | "asset" => ferro_ta_core::options::digital::DigitalKind::AssetOrNothing,
        _ => ferro_ta_core::options::digital::DigitalKind::CashOrNothing,
    }
}

/// Convert a Float64Array to a Vec<i64> (for instrument/side/option_type codes).
fn to_i64_vec(arr: &Float64Array) -> Vec<i64> {
    to_vec(arr).into_iter().map(|x| x as i64).collect()
}

/// Convert a Float64Array to a Vec<usize> (for window sizes).
fn to_usize_vec(arr: &Float64Array) -> Vec<usize> {
    to_vec(arr).into_iter().map(|x| x as usize).collect()
}

// ---------------------------------------------------------------------------
// Put-call parity check
// ---------------------------------------------------------------------------

/// Put-call parity deviation: `C - P - (S·e^{-qT} - K·e^{-rT})`.
///
/// Returns 0 at no-arbitrage.
#[wasm_bindgen]
pub fn put_call_parity_deviation(
    call_price: f64,
    put_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
) -> f64 {
    ferro_ta_core::options::pricing::put_call_parity_deviation(
        call_price, put_price, spot, strike, rate, carry, time_to_expiry,
    )
}

// ---------------------------------------------------------------------------
// Extended (higher-order) Greeks
// ---------------------------------------------------------------------------

/// Extended BSM Greeks: vanna, volga, charm, speed, color.
///
/// # Returns
/// `js_sys::Array` of five f64 values: `[vanna, volga, charm, speed, color]`.
#[wasm_bindgen]
pub fn extended_greeks(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: &str,
) -> Array {
    use ferro_ta_core::options::{greeks::model_extended_greeks, OptionContract, OptionEvaluation, PricingModel};
    let k = parse_option_kind(kind);
    // In this codebase, `carry` = dividend yield q (same convention as all other WASM/PyO3 APIs).
    let eg = model_extended_greeks(OptionEvaluation {
        contract: OptionContract {
            model: PricingModel::BlackScholes,
            underlying: spot,
            strike,
            rate,
            carry,
            time_to_expiry,
            kind: k,
        },
        volatility,
    });
    let out = Array::new();
    out.push(&JsValue::from_f64(eg.vanna));
    out.push(&JsValue::from_f64(eg.volga));
    out.push(&JsValue::from_f64(eg.charm));
    out.push(&JsValue::from_f64(eg.speed));
    out.push(&JsValue::from_f64(eg.color));
    out
}

// ---------------------------------------------------------------------------
// Digital options
// ---------------------------------------------------------------------------

/// Price a digital (binary) option.
///
/// # Arguments
/// - `kind` – `"call"` or `"put"`
/// - `digital_type` – `"cash_or_nothing"` (default) or `"asset_or_nothing"`
#[wasm_bindgen]
pub fn digital_price(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: &str,
    digital_type: &str,
) -> f64 {
    ferro_ta_core::options::digital::digital_price(
        spot,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        parse_option_kind(kind),
        parse_digital_kind(digital_type),
    )
}

/// Greeks for a digital option (numerical central differences).
///
/// # Returns
/// `js_sys::Array` of three f64 values: `[delta, gamma, vega]`.
#[wasm_bindgen]
pub fn digital_greeks(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: &str,
    digital_type: &str,
) -> Array {
    let (delta, gamma, vega) = ferro_ta_core::options::digital::digital_greeks(
        spot,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        parse_option_kind(kind),
        parse_digital_kind(digital_type),
    );
    let out = Array::new();
    out.push(&JsValue::from_f64(delta));
    out.push(&JsValue::from_f64(gamma));
    out.push(&JsValue::from_f64(vega));
    out
}

// ---------------------------------------------------------------------------
// American options (Barone-Adesi-Whaley)
// ---------------------------------------------------------------------------

/// American option price using the Barone-Adesi-Whaley approximation.
#[wasm_bindgen]
pub fn american_price(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: &str,
) -> f64 {
    ferro_ta_core::options::american::american_price_baw(
        spot,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        parse_option_kind(kind),
    )
}

/// Early exercise premium: `american_price - european_price`.
#[wasm_bindgen]
pub fn early_exercise_premium(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: &str,
) -> f64 {
    ferro_ta_core::options::american::early_exercise_premium(
        spot,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        parse_option_kind(kind),
    )
}

// ---------------------------------------------------------------------------
// Historical volatility estimators
// ---------------------------------------------------------------------------

/// Close-to-close realised volatility (rolling).
///
/// First `window - 1` values are `NaN`.
#[wasm_bindgen]
pub fn close_to_close_vol(
    close: &Float64Array,
    window: usize,
    trading_days: f64,
) -> Float64Array {
    from_vec(ferro_ta_core::options::realized_vol::close_to_close_vol(&to_vec(close), window, trading_days))
}

/// Parkinson (high-low) volatility estimator (rolling).
#[wasm_bindgen]
pub fn parkinson_vol(
    high: &Float64Array,
    low: &Float64Array,
    window: usize,
    trading_days: f64,
) -> Float64Array {
    from_vec(ferro_ta_core::options::realized_vol::parkinson_vol(
        &to_vec(high),
        &to_vec(low),
        window,
        trading_days,
    ))
}

/// Garman-Klass OHLC volatility estimator (rolling).
#[wasm_bindgen]
pub fn garman_klass_vol(
    open: &Float64Array,
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    window: usize,
    trading_days: f64,
) -> Float64Array {
    from_vec(ferro_ta_core::options::realized_vol::garman_klass_vol(
        &to_vec(open),
        &to_vec(high),
        &to_vec(low),
        &to_vec(close),
        window,
        trading_days,
    ))
}

/// Rogers-Satchell OHLC volatility estimator (rolling).
#[wasm_bindgen]
pub fn rogers_satchell_vol(
    open: &Float64Array,
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    window: usize,
    trading_days: f64,
) -> Float64Array {
    from_vec(ferro_ta_core::options::realized_vol::rogers_satchell_vol(
        &to_vec(open),
        &to_vec(high),
        &to_vec(low),
        &to_vec(close),
        window,
        trading_days,
    ))
}

/// Yang-Zhang OHLC volatility estimator (rolling).
///
/// Most efficient estimator — handles overnight gaps.
#[wasm_bindgen]
pub fn yang_zhang_vol(
    open: &Float64Array,
    high: &Float64Array,
    low: &Float64Array,
    close: &Float64Array,
    window: usize,
    trading_days: f64,
) -> Float64Array {
    from_vec(ferro_ta_core::options::realized_vol::yang_zhang_vol(
        &to_vec(open),
        &to_vec(high),
        &to_vec(low),
        &to_vec(close),
        window,
        trading_days,
    ))
}

// ---------------------------------------------------------------------------
// Volatility cone
// ---------------------------------------------------------------------------

/// Volatility cone: percentile distribution of close-to-close vol across windows.
///
/// # Arguments
/// - `close` – `Float64Array` of close prices.
/// - `windows` – `Float64Array` of window sizes (e.g. `[21, 42, 63, 126, 252]`).
/// - `trading_days` – annualisation factor (default 252).
///
/// # Returns
/// `js_sys::Array` of length `n_windows`, each element an `Array`:
/// `[window, min, p25, median, p75, max]`.
#[wasm_bindgen]
pub fn vol_cone(
    close: &Float64Array,
    windows: &Float64Array,
    trading_days: f64,
) -> Array {
    let c = to_vec(close);
    let wins = to_usize_vec(windows);
    let slices = ferro_ta_core::options::realized_vol::vol_cone(&c, &wins, trading_days);
    let out = Array::new();
    for s in slices {
        let row = Array::new();
        row.push(&JsValue::from_f64(s.window as f64));
        row.push(&JsValue::from_f64(s.min));
        row.push(&JsValue::from_f64(s.p25));
        row.push(&JsValue::from_f64(s.median));
        row.push(&JsValue::from_f64(s.p75));
        row.push(&JsValue::from_f64(s.max));
        out.push(&row);
    }
    out
}

// ---------------------------------------------------------------------------
// Expected move
// ---------------------------------------------------------------------------

/// Expected move over `days_to_expiry` trading days.
///
/// Uses log-normal: `spot · e^{±σ√(days/trading_days)} − spot`.
///
/// # Returns
/// `js_sys::Array` of two f64 values: `[lower_move, upper_move]` (signed).
#[wasm_bindgen]
pub fn expected_move(
    spot: f64,
    iv: f64,
    days_to_expiry: f64,
    trading_days_per_year: f64,
) -> Array {
    let (lower, upper) = ferro_ta_core::options::surface::expected_move(spot, iv, days_to_expiry, trading_days_per_year);
    let out = Array::new();
    out.push(&JsValue::from_f64(lower));
    out.push(&JsValue::from_f64(upper));
    out
}

// ---------------------------------------------------------------------------
// Strategy payoff / value (Feature 8 — WASM exposure)
// ---------------------------------------------------------------------------

/// Aggregate strategy payoff over a spot grid at expiry.
///
/// Instrument codes: `0`=option, `1`=future, `2`=stock.
/// Side codes: `1`=long, `-1`=short.
/// Option type codes: `1`=call, `-1`=put.
///
/// # Returns
/// `Float64Array` of aggregate P&L per spot grid point.
#[wasm_bindgen]
pub fn strategy_payoff_dense(
    spot_grid: &Float64Array,
    instruments: &Float64Array,
    sides: &Float64Array,
    option_types: &Float64Array,
    strikes: &Float64Array,
    premiums: &Float64Array,
    entry_prices: &Float64Array,
    quantities: &Float64Array,
    multipliers: &Float64Array,
) -> Float64Array {
    from_vec(ferro_ta_core::options::payoff::strategy_payoff_dense(
        &to_vec(spot_grid),
        &to_i64_vec(instruments),
        &to_i64_vec(sides),
        &to_i64_vec(option_types),
        &to_vec(strikes),
        &to_vec(premiums),
        &to_vec(entry_prices),
        &to_vec(quantities),
        &to_vec(multipliers),
    ))
}

/// Aggregate BSM Greeks across option and futures/stock legs at a single spot.
///
/// # Returns
/// `js_sys::Array` of five f64 values: `[delta, gamma, vega, theta, rho]`.
#[wasm_bindgen]
pub fn aggregate_greeks_dense(
    spot: f64,
    instruments: &Float64Array,
    sides: &Float64Array,
    option_types: &Float64Array,
    strikes: &Float64Array,
    volatilities: &Float64Array,
    time_to_expiries: &Float64Array,
    rates: &Float64Array,
    carries: &Float64Array,
    quantities: &Float64Array,
    multipliers: &Float64Array,
) -> Array {
    let (delta, gamma, vega, theta, rho) = ferro_ta_core::options::payoff::aggregate_greeks_dense(
        spot,
        &to_i64_vec(instruments),
        &to_i64_vec(sides),
        &to_i64_vec(option_types),
        &to_vec(strikes),
        &to_vec(volatilities),
        &to_vec(time_to_expiries),
        &to_vec(rates),
        &to_vec(carries),
        &to_vec(quantities),
        &to_vec(multipliers),
    );
    let out = Array::new();
    out.push(&JsValue::from_f64(delta));
    out.push(&JsValue::from_f64(gamma));
    out.push(&JsValue::from_f64(vega));
    out.push(&JsValue::from_f64(theta));
    out.push(&JsValue::from_f64(rho));
    out
}

/// Current BSM mid-price value of a multi-leg strategy over a spot grid (pre-expiry).
///
/// Unlike `strategy_payoff_dense`, this uses live BSM pricing for option legs.
///
/// # Returns
/// `Float64Array` of strategy value (P&L vs premium paid) per spot grid point.
#[wasm_bindgen]
pub fn strategy_value_grid(
    spot_grid: &Float64Array,
    instruments: &Float64Array,
    sides: &Float64Array,
    option_types: &Float64Array,
    strikes: &Float64Array,
    premiums: &Float64Array,
    entry_prices: &Float64Array,
    quantities: &Float64Array,
    multipliers: &Float64Array,
    time_to_expiries: &Float64Array,
    volatilities: &Float64Array,
    rates: &Float64Array,
    carries: &Float64Array,
) -> Float64Array {
    from_vec(ferro_ta_core::options::payoff::strategy_value_grid(
        &to_vec(spot_grid),
        &to_i64_vec(instruments),
        &to_i64_vec(sides),
        &to_i64_vec(option_types),
        &to_vec(strikes),
        &to_vec(premiums),
        &to_vec(entry_prices),
        &to_vec(quantities),
        &to_vec(multipliers),
        &to_vec(time_to_expiries),
        &to_vec(volatilities),
        &to_vec(rates),
        &to_vec(carries),
    ))
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
