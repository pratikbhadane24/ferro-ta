use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform Trend vs Cycle Mode: 1 = trending, 0 = cycling.
#[pyfunction]
pub fn ht_trendmode<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let result = ferro_ta_core::cycle::ht_trendmode(close.as_slice()?);
    Ok(result.into_pyarray(py))
}
