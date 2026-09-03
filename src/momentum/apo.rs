use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Absolute Price Oscillator: fast MA - slow MA.
///
/// `matype` defaults to `1` (EMA), not TA-Lib's `0` (SMA): this wrapper has
/// always computed the EMA form, so `1` keeps every existing call's output.
#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, slowperiod = 26, matype = 1))]
pub fn apo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
    matype: u8,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
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
    let result =
        py.allow_threads(|| ferro_ta_core::momentum::apo(prices, fastperiod, slowperiod, matype));
    Ok(result.into_pyarray(py))
}
