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
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    // Reuse the ATR core; divide by close to get NATR (saves duplicate TR computation).
    let atr_vals = ferro_ta_core::volatility::atr(highs, lows, closes, timeperiod);
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        if !atr_vals[i].is_nan() && closes[i] != 0.0 {
            result[i] = (atr_vals[i] / closes[i]) * 100.0;
        }
    }
    Ok(result.into_pyarray(py))
}
