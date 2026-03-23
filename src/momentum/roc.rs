use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::RateOfChange;
use ta::Next;

/// Rate of Change: (price - prev) / prev * 100. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 10))]
pub fn roc<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut indicator =
        RateOfChange::new(timeperiod).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut result = vec![f64::NAN; n];
    for (i, &price) in prices.iter().enumerate() {
        let val = indicator.next(price);
        if i >= timeperiod {
            result[i] = val;
        }
    }
    Ok(result.into_pyarray(py))
}

/// Rate of Change Percentage: (price - prev) / prev. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 10))]
pub fn rocp<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        let prev = prices[i - timeperiod];
        if prev != 0.0 {
            result[i] = (prices[i] - prev) / prev;
        }
    }
    Ok(result.into_pyarray(py))
}

/// Rate of Change Ratio: price / prev. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 10))]
pub fn rocr<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        let prev = prices[i - timeperiod];
        if prev != 0.0 {
            result[i] = prices[i] / prev;
        }
    }
    Ok(result.into_pyarray(py))
}

/// Rate of Change Ratio × 100: (price / prev) * 100. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 10))]
pub fn rocr100<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        let prev = prices[i - timeperiod];
        if prev != 0.0 {
            result[i] = (prices[i] / prev) * 100.0;
        }
    }
    Ok(result.into_pyarray(py))
}
