use super::common::compute_ht_core;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform Trend vs Cycle Mode: 1 = trending, 0 = cycling.
#[pyfunction]
pub fn ht_trendmode<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let prices = close.as_slice()?;
    let core = compute_ht_core(prices);
    Ok(core.trend_mode.into_pyarray(py))
}
