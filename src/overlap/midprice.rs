use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// MidPrice: (highest high + lowest low) / 2 over the rolling window.
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 14))]
pub fn midprice<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[(n, "high"), (lows.len(), "low")])?;
    let result = ferro_ta_core::overlap::midprice(highs, lows, timeperiod);
    Ok(result.into_pyarray(py))
}
