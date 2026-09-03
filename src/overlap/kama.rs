use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Kaufman Adaptive Moving Average. First output at index `timeperiod` (TA-Lib).
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn kama<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::overlap::kama(prices, timeperiod);
    Ok(result.into_pyarray(py))
}
