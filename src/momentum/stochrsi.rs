use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Stochastic RSI (TA-Lib–compatible): stochastic applied to RSI. Returns (fastk, fastd).
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14, fastk_period = 5, fastd_period = 3))]
#[allow(clippy::type_complexity)]
pub fn stochrsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    fastk_period: usize,
    fastd_period: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(fastk_period, "fastk_period", 1)?;
    validation::validate_timeperiod(fastd_period, "fastd_period", 1)?;
    let prices = close.as_slice()?;

    let (fastk, fastd) = py.allow_threads(|| {
        ferro_ta_core::momentum::stochrsi(prices, timeperiod, fastk_period, fastd_period)
    });
    Ok((fastk.into_pyarray(py), fastd.into_pyarray(py)))
}
