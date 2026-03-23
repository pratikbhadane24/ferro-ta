//! Alerts — condition evaluation helpers.
//!
//! These Rust functions evaluate conditions over price/indicator series and
//! return boolean or integer arrays indicating where conditions fire.  They
//! are designed to be called once per batch (backtest) or per bar (live) and
//! return the full history of firings.
//!
//! Functions
//! ---------
//! - `check_threshold` — fires when a series crosses above/below a level
//! - `check_cross`     — fires when *fast* crosses above or below *slow*
//! - `collect_alert_bars` — returns indices of bars where a bool mask is True

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// check_threshold
// ---------------------------------------------------------------------------

/// Fire an alert when *series* crosses a threshold level.
///
/// Parameters
/// ----------
/// series    : 1-D float64 array — indicator values (e.g. RSI)
/// level     : float — threshold value
/// direction : int
///     ``1``  → fire when series crosses **above** *level* (value goes from
///               ≤ level to > level).
///     ``-1`` → fire when series crosses **below** *level* (value goes from
///               ≥ level to < level).
///
/// Returns
/// -------
/// 1-D int8 array — 1 at the bar where the crossing occurs, 0 elsewhere.
/// Element 0 is always 0 (no crossing possible without a prior bar).
#[pyfunction]
pub fn check_threshold<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    level: f64,
    direction: i32,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    if direction != 1 && direction != -1 {
        return Err(PyValueError::new_err(
            "direction must be 1 (cross above) or -1 (cross below)",
        ));
    }
    let s = series.as_slice()?;
    let n = s.len();
    let mut out = vec![0i8; n];
    if n < 2 {
        return Ok(out.into_pyarray(py));
    }
    for i in 1..n {
        let prev = s[i - 1];
        let curr = s[i];
        if prev.is_nan() || curr.is_nan() {
            continue;
        }
        if direction == 1 {
            // cross above: was at or below level, now above
            if prev <= level && curr > level {
                out[i] = 1;
            }
        } else {
            // cross below: was at or above level, now below
            if prev >= level && curr < level {
                out[i] = 1;
            }
        }
    }
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// check_cross
// ---------------------------------------------------------------------------

/// Detect cross-over / cross-under events between two series.
///
/// Parameters
/// ----------
/// fast : 1-D float64 array — the "fast" series (e.g. short SMA)
/// slow : 1-D float64 array — the "slow" series (e.g. long SMA)
///
/// Returns
/// -------
/// 1-D int8 array:
///   ``1``  at bars where *fast* crosses **above** *slow* (bullish cross)
///   ``-1`` at bars where *fast* crosses **below** *slow* (bearish cross)
///   ``0``  elsewhere
/// Element 0 is always 0.
#[pyfunction]
pub fn check_cross<'py>(
    py: Python<'py>,
    fast: PyReadonlyArray1<'py, f64>,
    slow: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let f = fast.as_slice()?;
    let s = slow.as_slice()?;
    let n = f.len();
    if n != s.len() {
        return Err(PyValueError::new_err(
            "fast and slow must have the same length",
        ));
    }
    let mut out = vec![0i8; n];
    if n < 2 {
        return Ok(out.into_pyarray(py));
    }
    for i in 1..n {
        let fp = f[i - 1];
        let fc = f[i];
        let sp = s[i - 1];
        let sc = s[i];
        if fp.is_nan() || fc.is_nan() || sp.is_nan() || sc.is_nan() {
            continue;
        }
        // Bullish: fast was below slow, now above
        if fp <= sp && fc > sc {
            out[i] = 1;
        }
        // Bearish: fast was above slow, now below
        else if fp >= sp && fc < sc {
            out[i] = -1;
        }
    }
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// collect_alert_bars
// ---------------------------------------------------------------------------

/// Collect bar indices where *mask* is non-zero (i.e. condition fired).
///
/// Parameters
/// ----------
/// mask : 1-D int8 array (output of ``check_threshold`` or ``check_cross``)
///
/// Returns
/// -------
/// 1-D int64 array — indices of fired bars (ascending order)
#[pyfunction]
pub fn collect_alert_bars<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray1<'py, i8>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let m = mask.as_slice()?;
    let indices: Vec<i64> = m
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0)
        .map(|(i, _)| i as i64)
        .collect();
    Ok(indices.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(check_cross, m)?)?;
    m.add_function(wrap_pyfunction!(collect_alert_bars, m)?)?;
    Ok(())
}
