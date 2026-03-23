use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Kaufman Adaptive Moving Average. First value at index timeperiod-1.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn kama<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    if n < timeperiod {
        return Ok(vec![f64::NAN; n].into_pyarray(py));
    }

    let fast_sc = 2.0 / (2.0 + 1.0_f64);
    let slow_sc = 2.0 / (30.0 + 1.0_f64);

    let mut result = vec![f64::NAN; n];
    let mut kama_val = prices[timeperiod - 1];
    result[timeperiod - 1] = kama_val;

    for i in timeperiod..n {
        let direction = (prices[i] - prices[i - timeperiod]).abs();
        let mut volatility = 0.0_f64;
        for j in 1..=timeperiod {
            volatility += (prices[i - j + 1] - prices[i - j]).abs();
        }
        let er = if volatility > 0.0 {
            direction / volatility
        } else {
            0.0
        };
        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);
        kama_val += sc * (prices[i] - kama_val);
        result[i] = kama_val;
    }
    Ok(result.into_pyarray(py))
}
