use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// On Balance Volume: cumulates volume * sign(close - prev_close).
#[pyfunction]
pub fn obv<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let closes = close.as_slice()?;
    let vols = volume.as_slice()?;
    let n = closes.len();
    validation::validate_equal_length(&[(n, "close"), (vols.len(), "volume")])?;
    let result = ferro_ta_core::volume::obv(closes, vols);
    Ok(result.into_pyarray(py))
}
