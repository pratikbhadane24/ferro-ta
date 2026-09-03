use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// TRIX: 1-period rate of change of triple-smoothed EMA.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn trix<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::momentum::trix(prices, timeperiod));
    Ok(result.into_pyarray(py))
}
