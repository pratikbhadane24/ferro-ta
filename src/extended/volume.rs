//! Volume extended indicators — thin PyO3 wrappers.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_matype(matype: u8) -> PyResult<()> {
    if matype > 8 {
        return Err(PyValueError::new_err(
            "matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
        ));
    }
    Ok(())
}

/// On Balance Volume smoothed by an MA of type `matype`.
///
/// `0`–`6` and `8` match TA-Lib's numbering; `7` is T3 here where TA-Lib's `7`
/// is MAMA, and MAMA is not reachable through any `matype` (use `ferro_ta.MAMA`).
#[pyfunction]
#[pyo3(signature = (close, volume, timeperiod = 20, matype = 1))]
pub fn obv_smoothed<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    matype: u8,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validate_matype(matype)?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (v.len(), "volume")])?;
    let result =
        py.allow_threads(|| ferro_ta_core::extended::obv_smoothed(c, v, timeperiod, matype));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, timeperiod = 20))]
pub fn cmf<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[
        (h.len(), "high"),
        (lo.len(), "low"),
        (c.len(), "close"),
        (v.len(), "volume"),
    ])?;
    let result = py.allow_threads(|| ferro_ta_core::extended::cmf(h, lo, c, v, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (high, low, volume, timeperiod = 14, divisor = 10000.0))]
pub fn emv<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    divisor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[
        (h.len(), "high"),
        (lo.len(), "low"),
        (v.len(), "volume"),
    ])?;
    let result = py.allow_threads(|| ferro_ta_core::extended::emv(h, lo, v, timeperiod, divisor));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, volume, timeperiod = 13))]
pub fn force_index<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (v.len(), "volume")])?;
    let result = py.allow_threads(|| ferro_ta_core::extended::force_index(c, v, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, volume))]
pub fn nvi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (v.len(), "volume")])?;
    let result = py.allow_threads(|| ferro_ta_core::extended::nvi(c, v));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, volume, timeperiod = 255))]
pub fn nvi_with_ema<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (v.len(), "volume")])?;
    let (nvi, signal) =
        py.allow_threads(|| ferro_ta_core::extended::nvi_with_ema(c, v, timeperiod));
    Ok((nvi.into_pyarray(py), signal.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (close, volume))]
pub fn pvi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (v.len(), "volume")])?;
    let result = py.allow_threads(|| ferro_ta_core::extended::pvi(c, v));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, volume, timeperiod = 255, matype = 1))]
pub fn pvi_with_signal<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    matype: u8,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validate_matype(matype)?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (v.len(), "volume")])?;
    let (pvi, signal) =
        py.allow_threads(|| ferro_ta_core::extended::pvi_with_signal(c, v, timeperiod, matype));
    Ok((pvi.into_pyarray(py), signal.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (volume, fastperiod = 5, slowperiod = 10))]
pub fn volosc<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let v = volume.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::volosc(v, fastperiod, slowperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (volume, timeperiod = 25))]
pub fn vroc<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let v = volume.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::vroc(v, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, fastperiod = 34, slowperiod = 55, signalperiod = 13))]
pub fn kvo<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    validation::validate_timeperiod(signalperiod, "signalperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[
        (h.len(), "high"),
        (lo.len(), "low"),
        (c.len(), "close"),
        (v.len(), "volume"),
    ])?;
    let (line, signal) = py.allow_threads(|| {
        ferro_ta_core::extended::kvo(h, lo, c, v, fastperiod, slowperiod, signalperiod)
    });
    Ok((line.into_pyarray(py), signal.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (close, volume))]
pub fn pvt<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (v.len(), "volume")])?;
    let result = py.allow_threads(|| ferro_ta_core::extended::pvt(c, v));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (volume, timeperiod = 20))]
pub fn rvol<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let v = volume.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::rvol(v, timeperiod));
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(obv_smoothed, m)?)?;
    m.add_function(wrap_pyfunction!(cmf, m)?)?;
    m.add_function(wrap_pyfunction!(emv, m)?)?;
    m.add_function(wrap_pyfunction!(force_index, m)?)?;
    m.add_function(wrap_pyfunction!(nvi, m)?)?;
    m.add_function(wrap_pyfunction!(nvi_with_ema, m)?)?;
    m.add_function(wrap_pyfunction!(pvi, m)?)?;
    m.add_function(wrap_pyfunction!(pvi_with_signal, m)?)?;
    m.add_function(wrap_pyfunction!(volosc, m)?)?;
    m.add_function(wrap_pyfunction!(vroc, m)?)?;
    m.add_function(wrap_pyfunction!(kvo, m)?)?;
    m.add_function(wrap_pyfunction!(pvt, m)?)?;
    m.add_function(wrap_pyfunction!(rvol, m)?)?;
    Ok(())
}
