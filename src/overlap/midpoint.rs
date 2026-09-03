use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Midpoint: (max(close) + min(close)) / 2 over the rolling window.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn midpoint<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::overlap::midpoint(prices, timeperiod);
    Ok(result.into_pyarray(py))
}
