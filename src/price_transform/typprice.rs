use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Typical Price: (high + low + close) / 3.
#[pyfunction]
pub fn typprice<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .zip(closes.iter())
        .map(|((&h, &l), &c)| (h + l + c) / 3.0)
        .collect();
    Ok(result.into_pyarray(py))
}
