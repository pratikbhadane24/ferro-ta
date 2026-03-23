use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// On Balance Volume: cumulates volume * sign(close - prev_close); bar 0 uses volume.
#[pyfunction]
pub fn obv<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let closes = close.as_slice()?;
    let vols = volume.as_slice()?;
    let n = closes.len();
    validation::validate_equal_length(&[(n, "close"), (vols.len(), "volume")])?;
    let mut result = vec![0.0_f64; n];
    let mut obv_val = 0.0_f64;
    for i in 1..n {
        if closes[i] > closes[i - 1] {
            obv_val += vols[i];
        } else if closes[i] < closes[i - 1] {
            obv_val -= vols[i];
        }
        result[i] = obv_val;
    }
    Ok(result.into_pyarray(py))
}
