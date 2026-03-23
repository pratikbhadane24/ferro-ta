use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Exponential Moving Average. Leading timeperiod-1 values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn ema<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    log::debug!("EMA: timeperiod={timeperiod}, n={n}");
    let result = ferro_ta_core::overlap::ema(prices, timeperiod);
    Ok(result.into_pyarray(py))
}
