//! Tick / Trade Aggregation Pipeline — Rust implementations.
//!
//! Aggregates raw tick/trade data into OHLCV bars:
//! - **time bars**   — fixed duration buckets (label-based via Python timestamps)
//! - **volume bars** — fixed volume threshold per bar
//! - **tick bars**   — fixed number of ticks per bar
//!
//! The Python layer (ferro_ta.aggregation) provides the timestamp bucketing
//! for time bars; this module handles the compute-intensive OHLCV accumulation.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Return type for functions that return five OHLCV 1-D arrays.
type Ohlcv5<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Return type for time bars: five OHLCV arrays plus labels.
type Ohlcv5AndLabels<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
);

// ---------------------------------------------------------------------------
// aggregate_tick_bars
// ---------------------------------------------------------------------------

/// Aggregate tick/trade data into tick bars (every N ticks become one bar).
///
/// Parameters
/// ----------
/// price, size : 1-D float64 arrays (equal length, one entry per trade/tick)
/// ticks_per_bar : int — number of ticks per bar (must be >= 1)
///
/// Returns
/// -------
/// Tuple of five 1-D arrays: (open, high, low, close, volume)
/// where volume = sum of sizes in each bar.
#[pyfunction]
#[pyo3(signature = (price, size, ticks_per_bar))]
pub fn aggregate_tick_bars<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    size: PyReadonlyArray1<'py, f64>,
    ticks_per_bar: usize,
) -> PyResult<Ohlcv5<'py>> {
    if ticks_per_bar == 0 {
        return Err(PyValueError::new_err("ticks_per_bar must be >= 1"));
    }
    let p = price.as_slice()?;
    let s = size.as_slice()?;
    let n = p.len();
    if n == 0 || s.len() != n {
        return Err(PyValueError::new_err(
            "price and size must be non-empty and equal length",
        ));
    }

    let n_bars = n.div_ceil(ticks_per_bar);
    let mut out_open = Vec::with_capacity(n_bars);
    let mut out_high = Vec::with_capacity(n_bars);
    let mut out_low = Vec::with_capacity(n_bars);
    let mut out_close = Vec::with_capacity(n_bars);
    let mut out_vol = Vec::with_capacity(n_bars);

    let mut i = 0;
    while i < n {
        let end = (i + ticks_per_bar).min(n);
        let bar_p = &p[i..end];
        let bar_s = &s[i..end];
        let bar_open = bar_p[0];
        let bar_high = bar_p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bar_low = bar_p.iter().cloned().fold(f64::INFINITY, f64::min);
        let bar_close = *bar_p.last().unwrap();
        let bar_vol: f64 = bar_s.iter().sum();
        out_open.push(bar_open);
        out_high.push(bar_high);
        out_low.push(bar_low);
        out_close.push(bar_close);
        out_vol.push(bar_vol);
        i = end;
    }

    Ok((
        out_open.into_pyarray(py),
        out_high.into_pyarray(py),
        out_low.into_pyarray(py),
        out_close.into_pyarray(py),
        out_vol.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// aggregate_volume_bars_ticks
// ---------------------------------------------------------------------------

/// Aggregate tick data into volume bars (fixed volume threshold).
///
/// Accumulates ticks until cumulative size >= `volume_threshold`, then emits
/// a bar.
///
/// Parameters
/// ----------
/// price, size : 1-D float64 arrays (equal length)
/// volume_threshold : float — cumulative size threshold per bar (must be > 0)
///
/// Returns
/// -------
/// Tuple of five 1-D arrays: (open, high, low, close, volume)
#[pyfunction]
#[pyo3(signature = (price, size, volume_threshold))]
pub fn aggregate_volume_bars_ticks<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    size: PyReadonlyArray1<'py, f64>,
    volume_threshold: f64,
) -> PyResult<Ohlcv5<'py>> {
    if volume_threshold <= 0.0 {
        return Err(PyValueError::new_err("volume_threshold must be > 0"));
    }
    let p = price.as_slice()?;
    let s = size.as_slice()?;
    let n = p.len();
    if n == 0 || s.len() != n {
        return Err(PyValueError::new_err(
            "price and size must be non-empty and equal length",
        ));
    }

    let mut out_open: Vec<f64> = Vec::new();
    let mut out_high: Vec<f64> = Vec::new();
    let mut out_low: Vec<f64> = Vec::new();
    let mut out_close: Vec<f64> = Vec::new();
    let mut out_vol: Vec<f64> = Vec::new();

    let mut bar_open = p[0];
    let mut bar_high = p[0];
    let mut bar_low = p[0];
    let mut bar_close = p[0];
    let mut bar_vol = s[0];

    for i in 1..n {
        bar_high = bar_high.max(p[i]);
        bar_low = bar_low.min(p[i]);
        bar_close = p[i];
        bar_vol += s[i];

        if bar_vol >= volume_threshold {
            out_open.push(bar_open);
            out_high.push(bar_high);
            out_low.push(bar_low);
            out_close.push(bar_close);
            out_vol.push(bar_vol);
            if i + 1 < n {
                bar_open = p[i + 1];
                bar_high = p[i + 1];
                bar_low = p[i + 1];
                bar_close = p[i + 1];
                bar_vol = s[i + 1];
            } else {
                bar_vol = 0.0;
            }
        }
    }
    // Push remaining partial bar
    if bar_vol > 0.0 {
        out_open.push(bar_open);
        out_high.push(bar_high);
        out_low.push(bar_low);
        out_close.push(bar_close);
        out_vol.push(bar_vol);
    }

    Ok((
        out_open.into_pyarray(py),
        out_high.into_pyarray(py),
        out_low.into_pyarray(py),
        out_close.into_pyarray(py),
        out_vol.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// aggregate_time_bars
// ---------------------------------------------------------------------------

/// Aggregate tick data into time bars using pre-computed integer bucket labels.
///
/// Each tick is assigned a `label` (e.g. unix_ts // period_secs). Ticks with
/// the same label are accumulated into one bar.  Labels must be non-decreasing.
///
/// Parameters
/// ----------
/// price, size : 1-D float64 arrays
/// labels : 1-D int64 array — bucket label per tick (non-decreasing)
///
/// Returns
/// -------
/// Tuple of five 1-D arrays: (open, high, low, close, volume)
/// and a 1-D int64 array of unique labels (one per bar).
#[pyfunction]
#[pyo3(signature = (price, size, labels))]
pub fn aggregate_time_bars<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    size: PyReadonlyArray1<'py, f64>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Ohlcv5AndLabels<'py>> {
    let p = price.as_slice()?;
    let s = size.as_slice()?;
    let lbl = labels.as_slice()?;
    let n = p.len();
    if n == 0 || s.len() != n || lbl.len() != n {
        return Err(PyValueError::new_err(
            "price, size, and labels must be non-empty and equal length",
        ));
    }

    let mut out_open: Vec<f64> = Vec::new();
    let mut out_high: Vec<f64> = Vec::new();
    let mut out_low: Vec<f64> = Vec::new();
    let mut out_close: Vec<f64> = Vec::new();
    let mut out_vol: Vec<f64> = Vec::new();
    let mut out_labels: Vec<i64> = Vec::new();

    let mut cur_label = lbl[0];
    let mut bar_open = p[0];
    let mut bar_high = p[0];
    let mut bar_low = p[0];
    let mut bar_close = p[0];
    let mut bar_vol = s[0];

    for i in 1..n {
        if lbl[i] != cur_label {
            out_open.push(bar_open);
            out_high.push(bar_high);
            out_low.push(bar_low);
            out_close.push(bar_close);
            out_vol.push(bar_vol);
            out_labels.push(cur_label);
            cur_label = lbl[i];
            bar_open = p[i];
            bar_high = p[i];
            bar_low = p[i];
            bar_close = p[i];
            bar_vol = s[i];
        } else {
            bar_high = bar_high.max(p[i]);
            bar_low = bar_low.min(p[i]);
            bar_close = p[i];
            bar_vol += s[i];
        }
    }
    out_open.push(bar_open);
    out_high.push(bar_high);
    out_low.push(bar_low);
    out_close.push(bar_close);
    out_vol.push(bar_vol);
    out_labels.push(cur_label);

    Ok((
        out_open.into_pyarray(py),
        out_high.into_pyarray(py),
        out_low.into_pyarray(py),
        out_close.into_pyarray(py),
        out_vol.into_pyarray(py),
        out_labels.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_tick_bars, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_volume_bars_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_time_bars, m)?)?;
    Ok(())
}
