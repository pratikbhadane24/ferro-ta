use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Average True Range using TA-Lib–compatible Wilder smoothing.
///
/// Seeding: ATR[period] = SMA of TR[1..=period] (ignoring bar-0 TR which TA-Lib also skips).
/// Subsequent values: ATR[i] = (ATR[i-1] * (period-1) + TR[i]) / period.
/// Returns NaN for indices 0 through `timeperiod - 1`.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn atr<'py>(
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
    let result = ferro_ta_core::volatility::atr(highs, lows, closes, timeperiod);
    Ok(result.into_pyarray(py))
}
