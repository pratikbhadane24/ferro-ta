use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Triangular Moving Average (triangle-weighted). Leading timeperiod-1 values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30))]
pub fn trima<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();

    let mut weights = Vec::with_capacity(timeperiod);
    let half = timeperiod.div_ceil(2);
    for i in 1..=timeperiod {
        let w = if i <= half { i } else { timeperiod + 1 - i };
        weights.push(w as f64);
    }
    let weight_sum: f64 = weights.iter().sum();

    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let mut val = 0.0_f64;
        for (j, &w) in weights.iter().enumerate() {
            val += prices[i - (timeperiod - 1 - j)] * w;
        }
        result[i] = val / weight_sum;
    }
    Ok(result.into_pyarray(py))
}
