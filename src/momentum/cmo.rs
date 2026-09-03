use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Chande Momentum Oscillator. Uses TA-Lib–compatible Wilder smoothing
/// (same gain/loss seed as RSI). Returns NaN for the first `timeperiod` bars.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn cmo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::momentum::cmo(prices, timeperiod);
    Ok(result.into_pyarray(py))
}
