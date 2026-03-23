use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::{Maximum, Minimum};
use ta::Next;

/// Williams' %R: -100 * (highest high - close) / (highest high - lowest low) over the window.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn willr<'py>(
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
    let mut max_ind = Maximum::new(timeperiod).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut min_ind = Minimum::new(timeperiod).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut result = vec![f64::NAN; n];
    for (i, ((&h, &l), &c)) in highs.iter().zip(lows.iter()).zip(closes.iter()).enumerate() {
        let highest = max_ind.next(h);
        let lowest = min_ind.next(l);
        if i + 1 >= timeperiod {
            let range = highest - lowest;
            if range != 0.0 {
                result[i] = -100.0 * (highest - c) / range;
            } else {
                result[i] = -50.0;
            }
        }
    }
    Ok(result.into_pyarray(py))
}
