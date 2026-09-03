use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Aroon. Returns (aroon_down, aroon_up) tuple. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 14))]
#[allow(clippy::type_complexity)]
pub fn aroon<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    validation::validate_equal_length(&[(highs.len(), "high"), (lows.len(), "low")])?;
    let (aroon_down, aroon_up) = ferro_ta_core::momentum::aroon(highs, lows, timeperiod);
    Ok((aroon_down.into_pyarray(py), aroon_up.into_pyarray(py)))
}

/// Aroon Oscillator: aroon_up - aroon_down. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 14))]
pub fn aroonosc<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    validation::validate_equal_length(&[(highs.len(), "high"), (lows.len(), "low")])?;
    let result = ferro_ta_core::momentum::aroonosc(highs, lows, timeperiod);
    Ok(result.into_pyarray(py))
}
