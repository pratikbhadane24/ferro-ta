use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform Instantaneous Trendline (Ehlers). Smooths price over the dominant cycle period.
#[pyfunction]
pub fn ht_trendline<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = ferro_ta_core::cycle::ht_trendline(close.as_slice()?);
    Ok(result.into_pyarray(py))
}
