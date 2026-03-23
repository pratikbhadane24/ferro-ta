use super::common::compute_ht_core;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform Phasor components. Returns (inphase, quadrature) tuple.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn ht_phasor<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let prices = close.as_slice()?;
    let core = compute_ht_core(prices);
    Ok((
        core.inphase.into_pyarray(py),
        core.quadrature.into_pyarray(py),
    ))
}
