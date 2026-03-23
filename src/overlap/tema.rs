use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::ExponentialMovingAverage;
use ta::Next;

/// Triple Exponential Moving Average. Converges after ~3*(timeperiod-1) bars.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn tema<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();

    let mut ema1 = ExponentialMovingAverage::new(timeperiod)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut ema2 = ExponentialMovingAverage::new(timeperiod)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut ema3 = ExponentialMovingAverage::new(timeperiod)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let warmup1 = timeperiod - 1;
    let warmup2 = 2 * (timeperiod - 1);
    let warmup3 = 3 * (timeperiod - 1);
    let mut result = vec![f64::NAN; n];

    for (i, &price) in prices.iter().enumerate() {
        let v1 = ema1.next(price);
        if i >= warmup1 {
            let v2 = ema2.next(v1);
            if i >= warmup2 {
                let v3 = ema3.next(v2);
                if i >= warmup3 {
                    result[i] = 3.0 * v1 - 3.0 * v2 + v3;
                }
            }
        }
    }
    Ok(result.into_pyarray(py))
}
