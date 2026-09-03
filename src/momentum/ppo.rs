use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Percentage Price Oscillator. Returns (ppo_line, signal_line, histogram).
#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, slowperiod = 26, signalperiod = 9))]
#[allow(clippy::type_complexity)]
pub fn ppo<'py>(
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
    let (ppo_line, signal_line, hist) =
        ferro_ta_core::momentum::ppo(prices, fastperiod, slowperiod, signalperiod);
    Ok((
        ppo_line.into_pyarray(py),
        signal_line.into_pyarray(py),
        hist.into_pyarray(py),
    ))
}
