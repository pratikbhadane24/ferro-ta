use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// MACD (EMA-based). Returns (macd_line, signal_line, histogram).
#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, slowperiod = 26, signalperiod = 9))]
#[allow(clippy::type_complexity)]
pub fn macd<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    validation::validate_timeperiod(signalperiod, "signalperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let prices = close.as_slice()?;
    log::debug!(
        "MACD: fast={fastperiod}, slow={slowperiod}, signal={signalperiod}, n={}",
        prices.len()
    );
    let (macd_line, signal_line, histogram) =
        ferro_ta_core::overlap::macd(prices, fastperiod, slowperiod, signalperiod);
    Ok((
        macd_line.into_pyarray(py),
        signal_line.into_pyarray(py),
        histogram.into_pyarray(py),
    ))
}

/// MACD with fixed 12/26 periods. Returns (macd_line, signal_line, histogram).
#[pyfunction]
#[pyo3(signature = (close, signalperiod = 9))]
#[allow(clippy::type_complexity)]
pub fn macdfix<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    signalperiod: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    macd(py, close, 12, 26, signalperiod)
}
