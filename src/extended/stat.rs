//! Statistic extended indicators — thin PyO3 wrappers.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 3))]
pub fn median<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::median(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 3, atr_period = 14, multiplier = 2.0))]
pub fn median_bands<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    atr_period: usize,
    multiplier: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(atr_period, "atr_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (mid, upper, lower, mid_ema) = py.allow_threads(|| {
        ferro_ta_core::extended::median_bands(h, lo, c, timeperiod, atr_period, multiplier)
    });
    Ok((
        mid.into_pyarray(py),
        upper.into_pyarray(py),
        lower.into_pyarray(py),
        mid_ema.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 20, bins = 10))]
pub fn mode<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    bins: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(bins, "bins", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::mode(prices, timeperiod, bins));
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(median, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(median_bands, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(mode, m)?)?;
    Ok(())
}
