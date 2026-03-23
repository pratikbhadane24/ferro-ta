//! Rust rolling math operators — O(n) sliding window using monotonic deques.
//!
//! Functions exposed to Python:
//!   rolling_sum      — Rolling sum over `timeperiod` bars
//!   rolling_max      — Rolling maximum (O(n) via monotonic deque)
//!   rolling_min      — Rolling minimum (O(n) via monotonic deque)
//!   rolling_maxindex — Index of rolling maximum
//!   rolling_minindex — Index of rolling minimum

use std::collections::VecDeque;

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// rolling_sum
// ---------------------------------------------------------------------------

/// Rolling sum over `timeperiod` bars.
///
/// Uses a prefix-sum array for O(n) computation.
/// Leading `timeperiod - 1` values are NaN.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_sum<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    if n < timeperiod {
        return Ok(result.into_pyarray(py));
    }
    // Prefix sum
    let mut cs = vec![0.0f64; n + 1];
    for i in 0..n {
        cs[i + 1] = cs[i] + prices[i];
    }
    for i in (timeperiod - 1)..n {
        result[i] = cs[i + 1] - cs[i + 1 - timeperiod];
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// rolling_max
// ---------------------------------------------------------------------------

/// Rolling maximum over `timeperiod` bars (O(n) monotonic deque).
///
/// Leading `timeperiod - 1` values are NaN.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_max<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    let mut dq: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        // Remove indices out of the window
        while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
            dq.pop_front();
        }
        // Maintain decreasing deque
        while dq.back().map(|&j| prices[j] <= prices[i]).unwrap_or(false) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i + 1 >= timeperiod {
            result[i] = prices[*dq.front().unwrap()];
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// rolling_min
// ---------------------------------------------------------------------------

/// Rolling minimum over `timeperiod` bars (O(n) monotonic deque).
///
/// Leading `timeperiod - 1` values are NaN.
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_min<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    let mut dq: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
            dq.pop_front();
        }
        // Maintain increasing deque
        while dq.back().map(|&j| prices[j] >= prices[i]).unwrap_or(false) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i + 1 >= timeperiod {
            result[i] = prices[*dq.front().unwrap()];
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// rolling_maxindex
// ---------------------------------------------------------------------------

/// Index of rolling maximum over `timeperiod` bars (O(n) monotonic deque).
///
/// Returns the 0-based index into the input array.  During the warmup window
/// the value is `-1` (not valid — mask with warmup period if needed).
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_maxindex<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let n = prices.len();
    let mut result = vec![-1i64; n];
    let mut dq: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
            dq.pop_front();
        }
        while dq.back().map(|&j| prices[j] <= prices[i]).unwrap_or(false) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i + 1 >= timeperiod {
            result[i] = *dq.front().unwrap() as i64;
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// rolling_minindex
// ---------------------------------------------------------------------------

/// Index of rolling minimum over `timeperiod` bars (O(n) monotonic deque).
///
/// Returns the 0-based index into the input array.  During the warmup window
/// the value is `-1` (not valid — mask with warmup period if needed).
#[pyfunction]
#[pyo3(signature = (real, timeperiod = 30))]
pub fn rolling_minindex<'py>(
    py: Python<'py>,
    real: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = real.as_slice()?;
    let n = prices.len();
    let mut result = vec![-1i64; n];
    let mut dq: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
            dq.pop_front();
        }
        while dq.back().map(|&j| prices[j] >= prices[i]).unwrap_or(false) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i + 1 >= timeperiod {
            result[i] = *dq.front().unwrap() as i64;
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(rolling_sum, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rolling_max, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rolling_min, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rolling_maxindex, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rolling_minindex, m)?)?;
    Ok(())
}
