use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Median Price: (high + low) / 2.
#[pyfunction]
pub fn medprice<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[(n, "high"), (lows.len(), "low")])?;
    let result: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .map(|(&h, &l)| (h + l) / 2.0)
        .collect();
    Ok(result.into_pyarray(py))
}
