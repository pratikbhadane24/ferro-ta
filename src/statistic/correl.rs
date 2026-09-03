use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Pearson correlation coefficient between two series over the rolling window.
#[pyfunction]
#[pyo3(signature = (real0, real1, timeperiod = 30))]
pub fn correl<'py>(
    py: Python<'py>,
    real0: PyReadonlyArray1<'py, f64>,
    real1: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let x = real0.as_slice()?;
    let y = real1.as_slice()?;
    validation::validate_equal_length(&[(x.len(), "real0"), (y.len(), "real1")])?;
    let result = ferro_ta_core::statistic::correl(x, y, timeperiod);
    Ok(result.into_pyarray(py))
}
