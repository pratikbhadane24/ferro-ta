use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::ExponentialMovingAverage;
use ta::Next;

/// Absolute Price Oscillator: fast EMA - slow EMA.
#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, slowperiod = 26))]
pub fn apo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut fast_ema = ExponentialMovingAverage::new(fastperiod)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut slow_ema = ExponentialMovingAverage::new(slowperiod)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let warmup = slowperiod - 1;
    let mut result = vec![f64::NAN; n];
    for (i, &price) in prices.iter().enumerate() {
        let fast = fast_ema.next(price);
        let slow = slow_ema.next(price);
        if i >= warmup {
            result[i] = fast - slow;
        }
    }
    Ok(result.into_pyarray(py))
}
