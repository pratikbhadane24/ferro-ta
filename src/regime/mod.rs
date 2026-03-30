//! Regime detection and structural breaks (thin PyO3 wrapper over ferro_ta_core::regime).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation;

/// Label each bar as **trend** (1) or **range** (0) based on ADX level.
#[pyfunction]
pub fn regime_adx<'py>(
    py: Python<'py>,
    adx: PyReadonlyArray1<'py, f64>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let a = adx.as_slice()?;
    let result = ferro_ta_core::regime::regime_adx(a, threshold);
    Ok(result.into_pyarray(py))
}

/// Label each bar as trend (1) or range (0) using ADX + ATR-ratio rule.
#[pyfunction]
pub fn regime_combined<'py>(
    py: Python<'py>,
    adx: PyReadonlyArray1<'py, f64>,
    atr: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    adx_threshold: f64,
    atr_pct_threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let a = adx.as_slice()?;
    let r = atr.as_slice()?;
    let c = close.as_slice()?;
    let n = a.len();
    validation::validate_equal_length(&[(n, "adx"), (r.len(), "atr"), (c.len(), "close")])?;
    let result = ferro_ta_core::regime::regime_combined(a, r, c, adx_threshold, atr_pct_threshold);
    Ok(result.into_pyarray(py))
}

/// Detect structural breaks using a CUSUM approach.
#[pyfunction]
pub fn detect_breaks_cusum<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    window: usize,
    threshold: f64,
    slack: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    validation::validate_timeperiod(window, "window", 2)?;
    let s = series.as_slice()?;
    let result = ferro_ta_core::regime::detect_breaks_cusum(s, window, threshold, slack);
    Ok(result.into_pyarray(py))
}

/// Detect volatility regime breaks using rolling variance ratio.
#[pyfunction]
pub fn rolling_variance_break<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    short_window: usize,
    long_window: usize,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    validation::validate_timeperiod(short_window, "short_window", 2)?;
    if long_window <= short_window {
        return Err(PyValueError::new_err(
            "long_window must be > short_window",
        ));
    }
    let s = series.as_slice()?;
    let result = ferro_ta_core::regime::rolling_variance_break(s, short_window, long_window, threshold);
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regime_adx, m)?)?;
    m.add_function(wrap_pyfunction!(regime_combined, m)?)?;
    m.add_function(wrap_pyfunction!(detect_breaks_cusum, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_variance_break, m)?)?;
    Ok(())
}
