//! Hybrid extended indicators — thin PyO3 wrappers.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn dmi<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (plus_di, minus_di, adx) =
        py.allow_threads(|| ferro_ta_core::extended::dmi(h, lo, c, timeperiod));
    Ok((
        plus_di.into_pyarray(py),
        minus_di.into_pyarray(py),
        adx.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 2))]
pub fn williams_fractals<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let (up, down) =
        py.allow_threads(|| ferro_ta_core::extended::williams_fractals(h, lo, timeperiod));
    Ok((up.into_pyarray(py), down.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn rwi<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 2)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (rwi_high, rwi_low) =
        py.allow_threads(|| ferro_ta_core::extended::rwi(h, lo, c, timeperiod));
    Ok((rwi_high.into_pyarray(py), rwi_low.into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(dmi, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(williams_fractals, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(rwi, m)?)?;
    Ok(())
}
