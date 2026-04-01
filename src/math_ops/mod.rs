//! Rolling math operators (thin PyO3 wrapper over ferro_ta_core::math_ops).

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Rolling sum over `timeperiod` bars.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_sum<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = ferro_ta_core::math_ops::rolling_sum(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Rolling maximum over `timeperiod` bars (O(n) monotonic deque).
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_max<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = ferro_ta_core::math_ops::rolling_max(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Rolling minimum over `timeperiod` bars (O(n) monotonic deque).
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_min<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = ferro_ta_core::math_ops::rolling_min(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Index of rolling maximum over `timeperiod` bars.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_maxindex<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = ferro_ta_core::math_ops::rolling_maxindex(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Index of rolling minimum over `timeperiod` bars.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_minindex<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let result = ferro_ta_core::math_ops::rolling_minindex(prices, timeperiod);
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(rolling_sum, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rolling_max, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rolling_min, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rolling_maxindex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rolling_minindex, m)?)?;
    Ok(())
}
