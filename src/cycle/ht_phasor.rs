use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform Phasor components. Returns (inphase, quadrature) tuple.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn ht_phasor<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (inphase, quadrature) = ferro_ta_core::cycle::ht_phasor(close.as_slice()?);
    Ok((inphase.into_pyarray(py), quadrature.into_pyarray(py)))
}
