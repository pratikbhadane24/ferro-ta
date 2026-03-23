use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Balance Of Power: (close - open) / (high - low). Zero when range is zero.
#[pyfunction]
pub fn bop<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let opens = open.as_slice()?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = opens.len();
    validation::validate_equal_length(&[
        (n, "open"),
        (highs.len(), "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        let range = highs[i] - lows[i];
        if range != 0.0 {
            result[i] = (closes[i] - opens[i]) / range;
        } else {
            result[i] = 0.0;
        }
    }
    Ok(result.into_pyarray(py))
}
