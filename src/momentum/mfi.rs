use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Money Flow Index: volume-weighted RSI (typical price * volume). Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, timeperiod = 14))]
pub fn mfi<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let vols = volume.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
        (vols.len(), "volume"),
    ])?;
    log::debug!("MFI: timeperiod={timeperiod}, n={n}");
    let result = ferro_ta_core::volume::mfi(highs, lows, closes, vols, timeperiod);
    Ok(result.into_pyarray(py))
}
