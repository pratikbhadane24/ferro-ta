//! PyO3 wrappers for trend / overlap extended indicators.
//!
//! Thin GIL-releasing bindings. All math lives in `ferro_ta_core::extended`.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 21, offset = 0.85, sigma = 6.0))]
pub fn alma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    offset: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "sigma must be > 0, got {sigma}"
        )));
    }
    let prices = close.as_slice()?;
    let result =
        py.allow_threads(|| ferro_ta_core::extended::alma(prices, timeperiod, offset, sigma));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn zlema<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::zlema(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 16))]
pub fn frama<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 2)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::frama(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn mcginley<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::mcginley(prices, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14, cmo_period = 9))]
pub fn vidya<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    cmo_period: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(cmo_period, "cmo_period", 1)?;
    let prices = close.as_slice()?;
    let result =
        py.allow_threads(|| ferro_ta_core::extended::vidya(prices, timeperiod, cmo_period));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    high,
    low,
    jaw_period = 13,
    jaw_shift = 8,
    teeth_period = 8,
    teeth_shift = 5,
    lips_period = 5,
    lips_shift = 3
))]
pub fn alligator<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    jaw_period: usize,
    jaw_shift: usize,
    teeth_period: usize,
    teeth_shift: usize,
    lips_period: usize,
    lips_shift: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(jaw_period, "jaw_period", 1)?;
    validation::validate_timeperiod(teeth_period, "teeth_period", 1)?;
    validation::validate_timeperiod(lips_period, "lips_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let (jaw, teeth, lips) = py.allow_threads(|| {
        ferro_ta_core::extended::alligator(
            h,
            lo,
            jaw_period,
            jaw_shift,
            teeth_period,
            teeth_shift,
            lips_period,
            lips_shift,
        )
    });
    Ok((
        jaw.into_pyarray(py),
        teeth.into_pyarray(py),
        lips.into_pyarray(py),
    ))
}

/// Moving Average Envelopes: an MA of type `matype` with bands at
/// ±`percent`%. Returns (upper, middle, lower).
///
/// `0`–`6` and `8` match TA-Lib's numbering; `7` is T3 here where TA-Lib's `7`
/// is MAMA, and MAMA is not reachable through any `matype` (use `ferro_ta.MAMA`).
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 20, percent = 2.5, matype = 0))]
pub fn ma_envelopes<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    percent: f64,
    matype: u8,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    if matype > 8 {
        return Err(PyValueError::new_err(
            "matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
        ));
    }
    let prices = close.as_slice()?;
    let (upper, middle, lower) = py.allow_threads(|| {
        ferro_ta_core::extended::ma_envelopes(prices, timeperiod, percent, matype)
    });
    Ok((
        upper.into_pyarray(py),
        middle.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 10, multiplier = 1.0, stop_period = 9))]
pub fn chande_kroll_stop<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    multiplier: f64,
    stop_period: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(stop_period, "stop_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (long_stop, short_stop) = py.allow_threads(|| {
        ferro_ta_core::extended::chande_kroll_stop(h, lo, c, timeperiod, multiplier, stop_period)
    });
    Ok((long_stop.into_pyarray(py), short_stop.into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(alma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(zlema, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(frama, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(mcginley, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(vidya, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(alligator, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ma_envelopes, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(chande_kroll_stop, m)?)?;
    Ok(())
}
