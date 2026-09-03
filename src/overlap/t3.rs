use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Tillson T3 (triple smoothed EMA). Converges after ~6*(timeperiod-1) bars.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, vfactor = 0.7))]
pub fn t3<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    vfactor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::overlap::t3(prices, timeperiod, vfactor));
    Ok(result.into_pyarray(py))
}
