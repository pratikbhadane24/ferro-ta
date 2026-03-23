use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Relative Strength Index. Uses TA-Lib–compatible Wilder smoothing seed:
/// seed = SMA of first `timeperiod` gains (or losses), then Wilder EMA.
/// Returns NaN for the first `timeperiod` bars.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn rsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::momentum::rsi(prices, timeperiod);
    Ok(result.into_pyarray(py))
}
