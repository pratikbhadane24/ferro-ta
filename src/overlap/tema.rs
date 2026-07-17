use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Triple Exponential Moving Average. Converges after ~3*(timeperiod-1) bars.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn tema<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::overlap::tema(prices, timeperiod));
    Ok(result.into_pyarray(py))
}
