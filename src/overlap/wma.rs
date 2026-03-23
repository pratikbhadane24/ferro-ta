use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Weighted Moving Average (linear weights). Leading timeperiod-1 values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn wma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    log::debug!("WMA: timeperiod={timeperiod}, n={n}");
    let result = ferro_ta_core::overlap::wma(prices, timeperiod);
    Ok(result.into_pyarray(py))
}
