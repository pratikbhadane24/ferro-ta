use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Hilbert Transform SineWave. Returns (sine, leadsine) where leadsine leads sine by 45°.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn ht_sine<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (sine, lead_sine) = ferro_ta_core::cycle::ht_sine(close.as_slice()?);
    Ok((sine.into_pyarray(py), lead_sine.into_pyarray(py)))
}
