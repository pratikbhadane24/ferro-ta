use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Ultimate Oscillator: weighted sum of buying pressure over three periods (7, 14, 28).
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod1 = 7, timeperiod2 = 14, timeperiod3 = 28))]
pub fn ultosc<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod1, "timeperiod1", 1)?;
    validation::validate_timeperiod(timeperiod2, "timeperiod2", 1)?;
    validation::validate_timeperiod(timeperiod3, "timeperiod3", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    validation::validate_equal_length(&[
        (highs.len(), "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result =
        ferro_ta_core::momentum::ultosc(highs, lows, closes, timeperiod1, timeperiod2, timeperiod3);
    Ok(result.into_pyarray(py))
}
