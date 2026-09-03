use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Normalized ATR: (ATR / close) * 100. Same warmup as ATR.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn natr<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    validation::validate_equal_length(&[
        (highs.len(), "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result = ferro_ta_core::volatility::natr(highs, lows, closes, timeperiod);
    Ok(result.into_pyarray(py))
}
