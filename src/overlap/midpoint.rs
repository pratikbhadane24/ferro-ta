use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::{Maximum, Minimum};
use ta::Next;

/// Midpoint: (max(close) + min(close)) / 2 over the rolling window.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn midpoint<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut max_ind = Maximum::new(timeperiod).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut min_ind = Minimum::new(timeperiod).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut result = vec![f64::NAN; n];
    for (i, &price) in prices.iter().enumerate() {
        let mx = max_ind.next(price);
        let mn = min_ind.next(price);
        if i + 1 >= timeperiod {
            result[i] = (mx + mn) / 2.0;
        }
    }
    Ok(result.into_pyarray(py))
}
