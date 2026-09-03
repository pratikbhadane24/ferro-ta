use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Standard deviation over a rolling window; scaled by nbdev (default 1.0).
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, nbdev = 1.0))]
pub fn stddev<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    nbdev: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::statistic::stddev(prices, timeperiod, nbdev);
    Ok(result.into_pyarray(py))
}
