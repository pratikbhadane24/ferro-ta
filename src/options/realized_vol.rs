use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use ferro_ta_core::options::realized_vol as core;

#[pyfunction]
#[pyo3(signature = (close, window, trading_days = 252.0))]
pub fn close_to_close_vol<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    window: usize,
    trading_days: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(core::close_to_close_vol(close.as_slice()?, window, trading_days).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (high, low, window, trading_days = 252.0))]
pub fn parkinson_vol<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    window: usize,
    trading_days: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(
        core::parkinson_vol(high.as_slice()?, low.as_slice()?, window, trading_days)
            .into_pyarray(py),
    )
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, window, trading_days = 252.0))]
#[allow(clippy::too_many_arguments)]
pub fn garman_klass_vol<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    window: usize,
    trading_days: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(core::garman_klass_vol(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        window,
        trading_days,
    )
    .into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, window, trading_days = 252.0))]
#[allow(clippy::too_many_arguments)]
pub fn rogers_satchell_vol<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    window: usize,
    trading_days: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(core::rogers_satchell_vol(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        window,
        trading_days,
    )
    .into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, window, trading_days = 252.0))]
#[allow(clippy::too_many_arguments)]
pub fn yang_zhang_vol<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    window: usize,
    trading_days: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(core::yang_zhang_vol(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        window,
        trading_days,
    )
    .into_pyarray(py))
}

/// Returns a list of (window, min, p25, median, p75, max) tuples.
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (close, windows, trading_days = 252.0))]
pub fn vol_cone(
    close: PyReadonlyArray1<'_, f64>,
    windows: Vec<usize>,
    trading_days: f64,
) -> PyResult<Vec<(usize, f64, f64, f64, f64, f64)>> {
    let slices = core::vol_cone(close.as_slice()?, &windows, trading_days);
    Ok(slices
        .into_iter()
        .map(|s| (s.window, s.min, s.p25, s.median, s.p75, s.max))
        .collect())
}
