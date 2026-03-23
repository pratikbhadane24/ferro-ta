use super::common::rolling_linreg_apply;
use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64::consts::PI;

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
    let last_x = (timeperiod - 1) as f64;
    let result = rolling_linreg_apply(prices, timeperiod, |slope, intercept| {
        intercept + slope * last_x
    });
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
    let result = rolling_linreg_apply(prices, timeperiod, |slope, _| slope);
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
    let result = rolling_linreg_apply(prices, timeperiod, |_, intercept| intercept);
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
    let result = rolling_linreg_apply(prices, timeperiod, |slope, _| slope.atan() * 180.0 / PI);
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
    let forecast_x = timeperiod as f64;
    let result = rolling_linreg_apply(prices, timeperiod, |slope, intercept| {
        intercept + slope * forecast_x
    });
    Ok(result.into_pyarray(py))
}
