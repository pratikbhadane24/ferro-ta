//! PyO3 wrappers for momentum extended indicators.
//!
//! Thin GIL-releasing bindings. All math lives in `ferro_ta_core::extended`.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 13))]
pub fn elder_ray<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (bull, bear) =
        py.allow_threads(|| ferro_ta_core::extended::elder_ray(h, lo, c, timeperiod));
    Ok((bull.into_pyarray(py), bear.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 9))]
pub fn fisher<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let (fish, signal) = py.allow_threads(|| ferro_ta_core::extended::fisher(h, lo, timeperiod));
    Ok((fish.into_pyarray(py), signal.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 3, streakperiod = 2, rankperiod = 100))]
pub fn crsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    streakperiod: usize,
    rankperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(streakperiod, "streakperiod", 1)?;
    validation::validate_timeperiod(rankperiod, "rankperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| {
        ferro_ta_core::extended::crsi(prices, timeperiod, streakperiod, rankperiod)
    });
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(elder_ray, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(fisher, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crsi, m)?)?;
    Ok(())
}
