//! Resampling — OHLCV resampling and multi-timeframe helpers.
//!
//! Provides volume-bar resampling and OHLCV aggregation primitives.
//! Time-based resampling (pandas rule strings) is handled in the Python layer;
//! this module provides the compute-heavy parts that benefit from Rust.
//!
//! # Functions
//! - `volume_bars`   — Aggregate ticks/bars into bars of fixed volume size.
//! - `ohlcv_agg`     — Aggregate an array of OHLCV bars given bar-index labels.

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

// ---------------------------------------------------------------------------
// volume_bars
// ---------------------------------------------------------------------------

/// Aggregate OHLCV data into volume bars of a fixed volume threshold.
///
/// Each output bar accumulates input bars until `volume_threshold` units of
/// volume have been consumed.  The resulting bar has:
///   - open  = first open of the group
///   - high  = max high of the group
///   - low   = min low of the group
///   - close = last close of the group
///   - volume = sum of volumes (approximately `volume_threshold`)
///
/// Returns five 1-D arrays: (open, high, low, close, volume).
///
/// Parameters
/// ----------
/// open, high, low, close, volume : 1-D float64 arrays (equal length)
/// volume_threshold : float  — target volume per bar (must be > 0)
///
/// Returns
/// -------
/// Tuple of five 1-D float64 arrays (open, high, low, close, volume).
#[pyfunction]
#[pyo3(signature = (open, high, low, close, volume, volume_threshold))]
pub fn volume_bars<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    volume_threshold: f64,
) -> PyResult<Ohlcv5<'py>> {
    if volume_threshold <= 0.0 {
        return Err(PyValueError::new_err("volume_threshold must be > 0"));
    }
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    let n = o.len();
    if n == 0 || h.len() != n || l.len() != n || c.len() != n || v.len() != n {
        return Err(PyValueError::new_err(
            "All input arrays must be non-empty and have equal length",
        ));
    }

    let mut out_open: Vec<f64> = Vec::new();
    let mut out_high: Vec<f64> = Vec::new();
    let mut out_low: Vec<f64> = Vec::new();
    let mut out_close: Vec<f64> = Vec::new();
    let mut out_vol: Vec<f64> = Vec::new();

    let mut bar_open = o[0];
    let mut bar_high = h[0];
    let mut bar_low = l[0];
    let mut bar_close = c[0];
    let mut bar_vol = v[0];

    for i in 1..n {
        bar_high = bar_high.max(h[i]);
        bar_low = bar_low.min(l[i]);
        bar_close = c[i];
        bar_vol += v[i];

        if bar_vol >= volume_threshold {
            out_open.push(bar_open);
            out_high.push(bar_high);
            out_low.push(bar_low);
            out_close.push(bar_close);
            out_vol.push(bar_vol);
            // Start new bar
            if i + 1 < n {
                bar_open = o[i + 1];
                bar_high = h[i + 1];
                bar_low = l[i + 1];
                bar_close = c[i + 1];
                bar_vol = v[i + 1];
            }
        }
    }
    // Push any remaining partial bar
    if bar_vol > 0.0 && out_vol.last().is_none_or(|&last| last != bar_vol) {
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
// ohlcv_agg
// ---------------------------------------------------------------------------

/// Aggregate OHLCV bars by integer group labels.
///
/// Given OHLCV arrays and a `labels` array of non-negative integers (same
/// length), groups consecutive bars with the same label and computes:
///   - open  = first open of the group
///   - high  = max high of the group
///   - low   = min low of the group
///   - close = last close of the group
///   - volume = sum of volumes
///
/// `labels` must be non-decreasing (groups are contiguous).
///
/// Returns five 1-D arrays: (open, high, low, close, volume).
#[pyfunction]
#[pyo3(signature = (open, high, low, close, volume, labels))]
pub fn ohlcv_agg<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Ohlcv5<'py>> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    let lbl = labels.as_slice()?;
    let n = o.len();
    if n == 0 || h.len() != n || l.len() != n || c.len() != n || v.len() != n || lbl.len() != n {
        return Err(PyValueError::new_err(
            "All input arrays must be non-empty and have equal length",
        ));
    }

    let mut out_open: Vec<f64> = Vec::new();
    let mut out_high: Vec<f64> = Vec::new();
    let mut out_low: Vec<f64> = Vec::new();
    let mut out_close: Vec<f64> = Vec::new();
    let mut out_vol: Vec<f64> = Vec::new();

    let mut cur_label = lbl[0];
    let mut bar_open = o[0];
    let mut bar_high = h[0];
    let mut bar_low = l[0];
    let mut bar_close = c[0];
    let mut bar_vol = v[0];

    for i in 1..n {
        if lbl[i] != cur_label {
            out_open.push(bar_open);
            out_high.push(bar_high);
            out_low.push(bar_low);
            out_close.push(bar_close);
            out_vol.push(bar_vol);
            cur_label = lbl[i];
            bar_open = o[i];
            bar_high = h[i];
            bar_low = l[i];
            bar_close = c[i];
            bar_vol = v[i];
        } else {
            bar_high = bar_high.max(h[i]);
            bar_low = bar_low.min(l[i]);
            bar_close = c[i];
            bar_vol += v[i];
        }
    }
    out_open.push(bar_open);
    out_high.push(bar_high);
    out_low.push(bar_low);
    out_close.push(bar_close);
    out_vol.push(bar_vol);

    Ok((
        out_open.into_pyarray(py),
        out_high.into_pyarray(py),
        out_low.into_pyarray(py),
        out_close.into_pyarray(py),
        out_vol.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(volume_bars, m)?)?;
    m.add_function(wrap_pyfunction!(ohlcv_agg, m)?)?;
    Ok(())
}
