use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Parabolic SAR. Same shape as TA-Lib; reversal history may differ slightly.
#[pyfunction]
#[pyo3(signature = (high, low, acceleration = 0.02, maximum = 0.2))]
pub fn sar<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    acceleration: f64,
    maximum: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    validation::validate_equal_length(&[(highs.len(), "high"), (lows.len(), "low")])?;
    let result = ferro_ta_core::overlap::sar(highs, lows, acceleration, maximum);
    Ok(result.into_pyarray(py))
}
