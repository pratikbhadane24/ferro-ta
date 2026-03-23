use super::common::{compute_ht_core, HT_LOOKBACK};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64::consts::PI;

/// Hilbert Transform SineWave. Returns (sine, leadsine) where leadsine leads sine by 45°.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn ht_sine<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let prices = close.as_slice()?;
    let n = prices.len();
    let core = compute_ht_core(prices);

    let mut sine = vec![f64::NAN; n];
    let mut lead_sine = vec![f64::NAN; n];

    for i in HT_LOOKBACK..n {
        if !core.dc_phase[i].is_nan() {
            let phase_rad = core.dc_phase[i] * PI / 180.0;
            sine[i] = phase_rad.sin();
            lead_sine[i] = (phase_rad + PI / 4.0).sin(); // 45-degree lead
        }
    }

    Ok((sine.into_pyarray(py), lead_sine.into_pyarray(py)))
}
