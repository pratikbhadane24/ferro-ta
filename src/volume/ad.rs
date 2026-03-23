use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Chaikin Accumulation/Distribution Line. Cumulates (close - low - (high - close)) / (high - low) * volume.
#[pyfunction]
pub fn ad<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let vols = volume.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
        (vols.len(), "volume"),
    ])?;
    let mut result = vec![0.0_f64; n];
    let mut ad_val = 0.0_f64;
    for i in 0..n {
        let hl = highs[i] - lows[i];
        let clv = if hl != 0.0 {
            ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
        } else {
            0.0
        };
        ad_val += clv * vols[i];
        result[i] = ad_val;
    }
    Ok(result.into_pyarray(py))
}
