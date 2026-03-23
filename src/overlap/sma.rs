use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Inner SMA implementation (timeperiod already validated as usize).
/// Used by the PyO3 sma() and by ma() when matype=0.
pub fn sma_inner<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = close.as_slice()?;
    let n = prices.len();
    log::debug!("SMA: timeperiod={timeperiod}, n={n}");
    let result = ferro_ta_core::overlap::sma(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Simple Moving Average. Leading timeperiod-1 values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn sma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let timeperiod = validation::parse_timeperiod(timeperiod, "timeperiod", 1)?;
    sma_inner(py, close, timeperiod)
}
