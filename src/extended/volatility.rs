//! PyO3 wrappers for volatility extended indicators.
//!
//! Thin GIL-releasing bindings. All math lives in `ferro_ta_core::extended`.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 10, rocperiod = 10))]
pub fn chaikin_vol<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    rocperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(rocperiod, "rocperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let result =
        py.allow_threads(|| ferro_ta_core::extended::chaikin_vol(h, lo, timeperiod, rocperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 9, sumperiod = 25))]
pub fn mass<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    sumperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(sumperiod, "sumperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let result = py.allow_threads(|| ferro_ta_core::extended::mass(h, lo, timeperiod, sumperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, nbdevup = 2.0, nbdevdn = 2.0))]
pub fn bbpercent<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py
        .allow_threads(|| ferro_ta_core::extended::bbpercent(prices, timeperiod, nbdevup, nbdevdn));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, nbdevup = 2.0, nbdevdn = 2.0))]
pub fn bbwidth<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result =
        py.allow_threads(|| ferro_ta_core::extended::bbwidth(prices, timeperiod, nbdevup, nbdevdn));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 20, annual = 252.0))]
pub fn historical_volatility<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    annual: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| {
        ferro_ta_core::extended::historical_volatility(prices, timeperiod, annual)
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn ulcer_index<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::ulcer_index(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 15, atr_period = 15, multiplier = 2.0))]
pub fn starc<'py>(
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
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(atr_period, "atr_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (upper, middle, lower) = py.allow_threads(|| {
        ferro_ta_core::extended::starc(h, lo, c, timeperiod, atr_period, multiplier)
    });
    Ok((
        upper.into_pyarray(py),
        middle.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(chaikin_vol, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(mass, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bbpercent, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bbwidth, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(historical_volatility, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ulcer_index, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(starc, m)?)?;
    Ok(())
}
