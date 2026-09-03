use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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
    let result = ferro_ta_core::momentum::roc(prices, timeperiod);
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
    let result = ferro_ta_core::momentum::rocp(prices, timeperiod);
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
    let result = ferro_ta_core::momentum::rocr(prices, timeperiod);
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
    let result = ferro_ta_core::momentum::rocr100(prices, timeperiod);
    Ok(result.into_pyarray(py))
}
