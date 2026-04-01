//! Alerts — condition evaluation helpers (thin PyO3 wrapper over ferro_ta_core::alerts).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Fire an alert when *series* crosses a threshold level.
///
/// Parameters
/// ----------
/// series    : 1-D float64 array
/// level     : float — threshold value
/// direction : int — ``1`` (cross above) or ``-1`` (cross below)
///
/// Returns
/// -------
/// 1-D int8 array — 1 at crossing bars, 0 elsewhere.
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
    let result = ferro_ta_core::alerts::check_threshold(s, level, direction);
    Ok(result.into_pyarray(py))
}

/// Detect cross-over / cross-under events between two series.
///
/// Returns
/// -------
/// 1-D int8 array: ``1`` = bullish, ``-1`` = bearish, ``0`` = none.
#[pyfunction]
pub fn check_cross<'py>(
    py: Python<'py>,
    fast: PyReadonlyArray1<'py, f64>,
    slow: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let f = fast.as_slice()?;
    let s = slow.as_slice()?;
    if f.len() != s.len() {
        return Err(PyValueError::new_err(
            "fast and slow must have the same length",
        ));
    }
    let result = ferro_ta_core::alerts::check_cross(f, s);
    Ok(result.into_pyarray(py))
}

/// Collect bar indices where *mask* is non-zero.
#[pyfunction]
pub fn collect_alert_bars<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray1<'py, i8>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let m = mask.as_slice()?;
    let result = ferro_ta_core::alerts::collect_alert_bars(m);
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(check_cross, m)?)?;
    m.add_function(wrap_pyfunction!(collect_alert_bars, m)?)?;
    Ok(())
}
