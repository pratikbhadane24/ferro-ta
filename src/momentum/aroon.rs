use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Aroon. Returns (aroon_down, aroon_up) tuple. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 14))]
#[allow(clippy::type_complexity)]
pub fn aroon<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[(n, "high"), (lows.len(), "low")])?;
    let mut aroon_down = vec![f64::NAN; n];
    let mut aroon_up = vec![f64::NAN; n];
    let period_f = timeperiod as f64;

    for i in timeperiod..n {
        let window_size = timeperiod + 1;
        let start = i + 1 - window_size;
        let mut max_val = highs[start];
        let mut min_val = lows[start];
        let mut max_idx = 0usize;
        let mut min_idx = 0usize;
        for j in 0..window_size {
            if highs[start + j] >= max_val {
                max_val = highs[start + j];
                max_idx = j;
            }
            if lows[start + j] <= min_val {
                min_val = lows[start + j];
                min_idx = j;
            }
        }
        aroon_up[i] = 100.0 * (max_idx as f64) / period_f;
        aroon_down[i] = 100.0 * (min_idx as f64) / period_f;
    }
    Ok((aroon_down.into_pyarray(py), aroon_up.into_pyarray(py)))
}

/// Aroon Oscillator: aroon_up - aroon_down. Leading timeperiod values are NaN.
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 14))]
pub fn aroonosc<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[(n, "high"), (lows.len(), "low")])?;
    let mut result = vec![f64::NAN; n];
    let period_f = timeperiod as f64;

    #[allow(clippy::needless_range_loop)]
    for i in timeperiod..n {
        let window_size = timeperiod + 1;
        let start = i + 1 - window_size;
        let mut max_val = highs[start];
        let mut min_val = lows[start];
        let mut max_idx = 0usize;
        let mut min_idx = 0usize;
        for j in 0..window_size {
            if highs[start + j] >= max_val {
                max_val = highs[start + j];
                max_idx = j;
            }
            if lows[start + j] <= min_val {
                min_val = lows[start + j];
                min_idx = j;
            }
        }
        let up = 100.0 * (max_idx as f64) / period_f;
        let down = 100.0 * (min_idx as f64) / period_f;
        result[i] = up - down;
    }
    Ok(result.into_pyarray(py))
}
