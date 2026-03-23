use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::StandardDeviation;
use ta::Next;

/// Standard deviation over a rolling window; scaled by nbdev (default 1.0).
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, nbdev = 1.0))]
pub fn stddev<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    nbdev: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut indicator =
        StandardDeviation::new(timeperiod).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut result = vec![f64::NAN; n];
    for (i, &price) in prices.iter().enumerate() {
        let val = indicator.next(price);
        if i + 1 >= timeperiod {
            result[i] = val * nbdev;
        }
    }
    Ok(result.into_pyarray(py))
}
