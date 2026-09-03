use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Parabolic SAR Extended: SAR with configurable start value and long/short acceleration.
#[pyfunction]
#[pyo3(signature = (high, low, startvalue = 0.0, offsetonreverse = 0.0, accelerationinitlong = 0.02, accelerationlong = 0.02, accelerationmaxlong = 0.2, accelerationinitshort = 0.02, accelerationshort = 0.02, accelerationmaxshort = 0.2))]
#[allow(clippy::too_many_arguments)]
pub fn sarext<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    startvalue: f64,
    offsetonreverse: f64,
    accelerationinitlong: f64,
    accelerationlong: f64,
    accelerationmaxlong: f64,
    accelerationinitshort: f64,
    accelerationshort: f64,
    accelerationmaxshort: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    validation::validate_equal_length(&[(highs.len(), "high"), (lows.len(), "low")])?;
    let result = ferro_ta_core::overlap::sarext(
        highs,
        lows,
        startvalue,
        offsetonreverse,
        accelerationinitlong,
        accelerationlong,
        accelerationmaxlong,
        accelerationinitshort,
        accelerationshort,
        accelerationmaxshort,
    );
    Ok(result.into_pyarray(py))
}
