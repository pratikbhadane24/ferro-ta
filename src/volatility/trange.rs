use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// True Range: max(high - low, |high - prev_close|, |low - prev_close|).
/// Bar 0 is NaN (TA-Lib: no previous close).
#[pyfunction]
pub fn trange<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    validation::validate_equal_length(&[
        (highs.len(), "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result = ferro_ta_core::volatility::trange(highs, lows, closes);
    Ok(result.into_pyarray(py))
}
