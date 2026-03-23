use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Momentum: close[i] - close[i - timeperiod]. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 10))]
pub fn mom<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        result[i] = prices[i] - prices[i - timeperiod];
    }
    Ok(result.into_pyarray(py))
}
