use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Chande Momentum Oscillator: 100 * (gains - losses) / (gains + losses),
/// using TA-Lib's Wilder smoothing.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn cmo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::momentum::cmo(prices, timeperiod));
    Ok(result.into_pyarray(py))
}
