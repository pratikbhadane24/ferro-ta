use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Percentage Price Oscillator. Returns (ppo_line, signal_line, histogram).
///
/// `matype` defaults to `1` (EMA), not TA-Lib's `0` (SMA): this wrapper has
/// always computed the EMA form, so `1` keeps every existing call's output.
#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, slowperiod = 26, signalperiod = 9, matype = 1))]
#[allow(clippy::type_complexity)]
pub fn ppo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
    matype: u8,
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
    if matype > 8 {
        return Err(PyValueError::new_err(
            "matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
        ));
    }
    let prices = close.as_slice()?;
    let (ppo_line, signal_line, hist) = py.allow_threads(|| {
        ferro_ta_core::momentum::ppo(prices, fastperiod, slowperiod, signalperiod, matype)
    });
    Ok((
        ppo_line.into_pyarray(py),
        signal_line.into_pyarray(py),
        hist.into_pyarray(py),
    ))
}
