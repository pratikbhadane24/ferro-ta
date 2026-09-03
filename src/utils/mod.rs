//! Signal utilities — thin PyO3 wrappers over `ferro_ta_core::utils`.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// `1.0` where `real0` crosses strictly above `real1`.
#[pyfunction]
pub fn crossover<'py>(
    py: Python<'py>,
    real0: PyReadonlyArray1<'py, f64>,
    real1: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = real0.as_slice()?;
    let b = real1.as_slice()?;
    validation::validate_equal_length(&[(a.len(), "real0"), (b.len(), "real1")])?;
    let result = py.allow_threads(|| ferro_ta_core::utils::crossover(a, b));
    Ok(result.into_pyarray(py))
}

/// `1.0` where `real0` crosses strictly below `real1`.
#[pyfunction]
pub fn crossunder<'py>(
    py: Python<'py>,
    real0: PyReadonlyArray1<'py, f64>,
    real1: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = real0.as_slice()?;
    let b = real1.as_slice()?;
    validation::validate_equal_length(&[(a.len(), "real0"), (b.len(), "real1")])?;
    let result = py.allow_threads(|| ferro_ta_core::utils::crossunder(a, b));
    Ok(result.into_pyarray(py))
}

/// `1.0` where `real0` crosses `real1` in either direction.
#[pyfunction]
pub fn cross<'py>(
    py: Python<'py>,
    real0: PyReadonlyArray1<'py, f64>,
    real1: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = real0.as_slice()?;
    let b = real1.as_slice()?;
    validation::validate_equal_length(&[(a.len(), "real0"), (b.len(), "real1")])?;
    let result = py.allow_threads(|| ferro_ta_core::utils::cross(a, b));
    Ok(result.into_pyarray(py))
}

/// Rolling highest value over `timeperiod` bars (same math as `MAX`).
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn highest<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::utils::highest(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

/// Rolling lowest value over `timeperiod` bars (same math as `MIN`).
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn lowest<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::utils::lowest(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

/// Lookback difference: `real[i] - real[i - timeperiod]`.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 1))]
pub fn change<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::utils::change(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

/// `1.0` when `real[i]` is strictly greater than `real[i - timeperiod]`.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 1))]
pub fn rising<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::utils::rising(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

/// `1.0` when `real[i]` is strictly less than `real[i - timeperiod]`.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 1))]
pub fn falling<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::utils::falling(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

/// Keep the first `primary` signal until a `secondary` signal resets it.
#[pyfunction]
pub fn exrem<'py>(
    py: Python<'py>,
    primary: PyReadonlyArray1<'py, f64>,
    secondary: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = primary.as_slice()?;
    let b = secondary.as_slice()?;
    validation::validate_equal_length(&[(a.len(), "primary"), (b.len(), "secondary")])?;
    let result = py.allow_threads(|| ferro_ta_core::utils::exrem(a, b));
    Ok(result.into_pyarray(py))
}

/// Hold `1.0` from a `primary` signal until a `secondary` signal clears it.
#[pyfunction]
pub fn flip<'py>(
    py: Python<'py>,
    primary: PyReadonlyArray1<'py, f64>,
    secondary: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = primary.as_slice()?;
    let b = secondary.as_slice()?;
    validation::validate_equal_length(&[(a.len(), "primary"), (b.len(), "secondary")])?;
    let result = py.allow_threads(|| ferro_ta_core::utils::flip(a, b));
    Ok(result.into_pyarray(py))
}

/// Value of `real` at the `occurrence`-th most recent true `condition`.
#[pyfunction]
#[pyo3(signature = (condition, real, occurrence = 1))]
pub fn valuewhen<'py>(
    py: Python<'py>,
    condition: PyReadonlyArray1<'py, f64>,
    real: PyReadonlyArray1<'py, f64>,
    occurrence: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(occurrence, "occurrence", 1)?;
    let cond = condition.as_slice()?;
    let prices = real.as_slice()?;
    validation::validate_equal_length(&[(cond.len(), "condition"), (prices.len(), "real")])?;
    let result = py.allow_threads(|| ferro_ta_core::utils::valuewhen(cond, prices, occurrence));
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(crossover, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crossunder, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cross, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(highest, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lowest, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(change, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rising, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(falling, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(exrem, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(flip, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(valuewhen, m)?)?;
    Ok(())
}
