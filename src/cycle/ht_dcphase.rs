use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform Dominant Cycle Phase in degrees.
#[pyfunction]
pub fn ht_dcphase<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = ferro_ta_core::cycle::ht_dcphase(close.as_slice()?);
    Ok(result.into_pyarray(py))
}
