use super::common::compute_ht_core;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform Dominant Cycle Phase in degrees.
#[pyfunction]
pub fn ht_dcphase<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = close.as_slice()?;
    let core = compute_ht_core(prices);
    Ok(core.dc_phase.into_pyarray(py))
}
