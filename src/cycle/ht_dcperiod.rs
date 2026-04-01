use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform Dominant Cycle Period in bars.
#[pyfunction]
pub fn ht_dcperiod<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = ferro_ta_core::cycle::ht_dcperiod(close.as_slice()?);
    Ok(result.into_pyarray(py))
}
