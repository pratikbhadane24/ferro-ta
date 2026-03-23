use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Bollinger Bands. Returns (upper, middle, lower). Middle is SMA; bands are ± nbdev * stddev.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, nbdevup = 2.0, nbdevdn = 2.0))]
#[allow(clippy::type_complexity)]
pub fn bbands<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    log::debug!("BBANDS: timeperiod={timeperiod}, n={}", prices.len());
    let (upper, middle, lower) =
        ferro_ta_core::overlap::bbands(prices, timeperiod, nbdevup, nbdevdn);
    Ok((
        upper.into_pyarray(py),
        middle.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}
