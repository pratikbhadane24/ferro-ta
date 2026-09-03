use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Balance Of Power: (close - open) / (high - low). Zero when range is zero.
#[pyfunction]
pub fn bop<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let opens = open.as_slice()?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    validation::validate_equal_length(&[
        (opens.len(), "open"),
        (highs.len(), "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result = ferro_ta_core::momentum::bop(opens, highs, lows, closes);
    Ok(result.into_pyarray(py))
}
