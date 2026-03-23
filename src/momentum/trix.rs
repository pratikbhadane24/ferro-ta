use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::ExponentialMovingAverage;
use ta::Next;

/// TRIX: 1-period rate of change of triple-smoothed EMA.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn trix<'py>(
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

    let warmup = 3 * (timeperiod - 1);
    let mut ema3_vals = vec![f64::NAN; n];
    let mut result = vec![f64::NAN; n];

    for (i, &price) in prices.iter().enumerate() {
        let v1 = ema1.next(price);
        if i >= timeperiod - 1 {
            let v2 = ema2.next(v1);
            if i >= 2 * (timeperiod - 1) {
                let v3 = ema3.next(v2);
                if i >= warmup {
                    ema3_vals[i] = v3;
                }
            }
        }
    }

    for i in (warmup + 1)..n {
        let prev = ema3_vals[i - 1];
        if !ema3_vals[i].is_nan() && !prev.is_nan() && prev != 0.0 {
            result[i] = (ema3_vals[i] - prev) / prev * 100.0;
        }
    }
    Ok(result.into_pyarray(py))
}
