use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Linear regression fitted value at the last point of the window.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn linearreg<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::statistic::linearreg(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Slope of the rolling linear regression line.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn linearreg_slope<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::statistic::linearreg_slope(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Intercept of the rolling linear regression line.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn linearreg_intercept<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::statistic::linearreg_intercept(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Angle of the regression line in degrees (atan(slope) * 180/π).
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn linearreg_angle<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::statistic::linearreg_angle(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Time series forecast: linear regression extrapolated one period ahead.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn tsf<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = ferro_ta_core::statistic::tsf(prices, timeperiod);
    Ok(result.into_pyarray(py))
}
