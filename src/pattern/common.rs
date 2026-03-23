//! Shared helpers for candlestick pattern detection.

use pyo3::prelude::PyResult;

/// Validate that open, high, low, close arrays have the same length (for use in CDL* functions).
pub fn validate_ohlc_length(
    open_len: usize,
    high_len: usize,
    low_len: usize,
    close_len: usize,
) -> PyResult<()> {
    crate::validation::validate_equal_length(&[
        (open_len, "open"),
        (high_len, "high"),
        (low_len, "low"),
        (close_len, "close"),
    ])
}

/// Epsilon for doji-like candles (body ≈ 0) to avoid division by zero.
pub const DOJI_BODY_EPSILON: f64 = 0.0001;

#[inline]
pub fn body_size(open: f64, close: f64) -> f64 {
    (close - open).abs()
}

#[inline]
pub fn upper_shadow(open: f64, high: f64, close: f64) -> f64 {
    high - open.max(close)
}

#[inline]
pub fn lower_shadow(open: f64, low: f64, close: f64) -> f64 {
    open.min(close) - low
}

#[inline]
pub fn candle_range(high: f64, low: f64) -> f64 {
    high - low
}

#[inline]
pub fn is_bullish(open: f64, close: f64) -> bool {
    close >= open
}

#[inline]
pub fn is_bearish(open: f64, close: f64) -> bool {
    close < open
}
