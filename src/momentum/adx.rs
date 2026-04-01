//! ADX family: PLUS_DM, MINUS_DM, +DI, -DI, DX, ADX, ADXR.
//! Thin wrappers that delegate to ferro_ta_core::momentum.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Six-tuple of bound PyArray1 vectors (PLUS_DM, MINUS_DM, +DI, -DI, DX, ADX).
type AdxAllResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Plus Directional Movement (Wilder smoothing).
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 14))]
pub fn plus_dm<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    validation::validate_equal_length(&[(highs.len(), "high"), (lows.len(), "low")])?;
    let result = ferro_ta_core::momentum::plus_dm(highs, lows, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Minus Directional Movement (Wilder smoothing).
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 14))]
pub fn minus_dm<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    validation::validate_equal_length(&[(highs.len(), "high"), (lows.len(), "low")])?;
    let result = ferro_ta_core::momentum::minus_dm(highs, lows, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Plus Directional Indicator (Wilder smoothing).
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn plus_di<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result = ferro_ta_core::momentum::plus_di(highs, lows, closes, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Minus Directional Indicator (Wilder smoothing).
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn minus_di<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result = ferro_ta_core::momentum::minus_di(highs, lows, closes, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Directional Movement Index: 100 * |+DI - -DI| / (+DI + -DI).
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn dx<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result = ferro_ta_core::momentum::dx(highs, lows, closes, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Average Directional Movement Index (Wilder smoothing of DX).
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn adx<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result = ferro_ta_core::momentum::adx(highs, lows, closes, timeperiod);
    Ok(result.into_pyarray(py))
}

/// ADX Rating: (ADX[i] + ADX[i - timeperiod]) / 2.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn adxr<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result = ferro_ta_core::momentum::adxr(highs, lows, closes, timeperiod);
    Ok(result.into_pyarray(py))
}

/// Compute all six ADX-family outputs in a single TR/PDM/MDM pass.
///
/// Returns (plus_dm, minus_dm, plus_di, minus_di, dx, adx) — six arrays of
/// the same length as the inputs.  Use this when you need more than one ADX
/// family output to avoid redundant computation.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn adx_all<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<AdxAllResult<'py>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    validation::validate_equal_length(&[
        (highs.len(), "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let (pdm, mdm, pdi, mdi, dx, adx) =
        ferro_ta_core::momentum::adx_all(highs, lows, closes, timeperiod);
    Ok((
        pdm.into_pyarray(py),
        mdm.into_pyarray(py),
        pdi.into_pyarray(py),
        mdi.into_pyarray(py),
        dx.into_pyarray(py),
        adx.into_pyarray(py),
    ))
}
