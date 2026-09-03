//! Oscillator extended indicators — thin PyO3 wrappers.

#![allow(clippy::too_many_arguments)]

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (high, low, fastperiod = 5, slowperiod = 34))]
pub fn ao<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
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
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let result = py.allow_threads(|| ferro_ta_core::extended::ao(h, lo, fastperiod, slowperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (high, low, fastperiod = 5, slowperiod = 34, smoothperiod = 5))]
pub fn ac<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
    smoothperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    validation::validate_timeperiod(smoothperiod, "smoothperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let result = py
        .allow_threads(|| ferro_ta_core::extended::ac(h, lo, fastperiod, slowperiod, smoothperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, fastperiod = 10, slowperiod = 21))]
pub fn po<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
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
    let c = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::po(c, fastperiod, slowperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 20))]
pub fn dpo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let c = close.as_slice()?;
    let result = py.allow_threads(|| ferro_ta_core::extended::dpo(c, timeperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, timeperiod = 10))]
pub fn rvi<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[
        (o.len(), "open"),
        (h.len(), "high"),
        (lo.len(), "low"),
        (c.len(), "close"),
    ])?;
    let (rvi, signal) = py.allow_threads(|| ferro_ta_core::extended::rvi(o, h, lo, c, timeperiod));
    Ok((rvi.into_pyarray(py), signal.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, fastperiod = 3, slowperiod = 10))]
pub fn cho<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
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
    let result =
        py.allow_threads(|| ferro_ta_core::extended::cho(h, lo, c, v, fastperiod, slowperiod));
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    close,
    roc1 = 10,
    roc2 = 15,
    roc3 = 20,
    roc4 = 30,
    sma1 = 10,
    sma2 = 10,
    sma3 = 10,
    sma4 = 15,
    signalperiod = 9
))]
pub fn kst<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    roc1: usize,
    roc2: usize,
    roc3: usize,
    roc4: usize,
    sma1: usize,
    sma2: usize,
    sma3: usize,
    sma4: usize,
    signalperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(roc1, "roc1", 1)?;
    validation::validate_timeperiod(roc2, "roc2", 1)?;
    validation::validate_timeperiod(roc3, "roc3", 1)?;
    validation::validate_timeperiod(roc4, "roc4", 1)?;
    validation::validate_timeperiod(sma1, "sma1", 1)?;
    validation::validate_timeperiod(sma2, "sma2", 1)?;
    validation::validate_timeperiod(sma3, "sma3", 1)?;
    validation::validate_timeperiod(sma4, "sma4", 1)?;
    validation::validate_timeperiod(signalperiod, "signalperiod", 1)?;
    let c = close.as_slice()?;
    let (kst, signal) = py.allow_threads(|| {
        ferro_ta_core::extended::kst(
            c,
            roc1,
            roc2,
            roc3,
            roc4,
            sma1,
            sma2,
            sma3,
            sma4,
            signalperiod,
        )
    });
    Ok((kst.into_pyarray(py), signal.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (close, longperiod = 25, shortperiod = 13, signalperiod = 13))]
pub fn tsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    longperiod: usize,
    shortperiod: usize,
    signalperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(longperiod, "longperiod", 1)?;
    validation::validate_timeperiod(shortperiod, "shortperiod", 1)?;
    validation::validate_timeperiod(signalperiod, "signalperiod", 1)?;
    let c = close.as_slice()?;
    let (tsi, signal) =
        py.allow_threads(|| ferro_ta_core::extended::tsi(c, longperiod, shortperiod, signalperiod));
    Ok((tsi.into_pyarray(py), signal.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn vortex<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (plus_vi, minus_vi) =
        py.allow_threads(|| ferro_ta_core::extended::vortex(h, lo, c, timeperiod));
    Ok((plus_vi.into_pyarray(py), minus_vi.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (close, fastperiod = 23, slowperiod = 50, cycleperiod = 10, d1 = 3, d2 = 3))]
pub fn stc<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
    cycleperiod: usize,
    d1: usize,
    d2: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    validation::validate_timeperiod(cycleperiod, "cycleperiod", 1)?;
    validation::validate_timeperiod(d1, "d1", 1)?;
    validation::validate_timeperiod(d2, "d2", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let c = close.as_slice()?;
    let result = py.allow_threads(|| {
        ferro_ta_core::extended::stc(c, fastperiod, slowperiod, cycleperiod, d1, d2)
    });
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
pub fn gator<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    jaw_period: usize,
    jaw_shift: usize,
    teeth_period: usize,
    teeth_shift: usize,
    lips_period: usize,
    lips_shift: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(jaw_period, "jaw_period", 1)?;
    validation::validate_timeperiod(teeth_period, "teeth_period", 1)?;
    validation::validate_timeperiod(lips_period, "lips_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let (upper, lower) = py.allow_threads(|| {
        ferro_ta_core::extended::gator(
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
    Ok((upper.into_pyarray(py), lower.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (close, wma_period = 10, roc1_period = 14, roc2_period = 11))]
pub fn coppock<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    wma_period: usize,
    roc1_period: usize,
    roc2_period: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(wma_period, "wma_period", 1)?;
    validation::validate_timeperiod(roc1_period, "roc1_period", 1)?;
    validation::validate_timeperiod(roc2_period, "roc2_period", 1)?;
    let c = close.as_slice()?;
    let result = py.allow_threads(|| {
        ferro_ta_core::extended::coppock(c, wma_period, roc1_period, roc2_period)
    });
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ao, m)?)?;
    m.add_function(wrap_pyfunction!(ac, m)?)?;
    m.add_function(wrap_pyfunction!(po, m)?)?;
    m.add_function(wrap_pyfunction!(dpo, m)?)?;
    m.add_function(wrap_pyfunction!(rvi, m)?)?;
    m.add_function(wrap_pyfunction!(cho, m)?)?;
    m.add_function(wrap_pyfunction!(kst, m)?)?;
    m.add_function(wrap_pyfunction!(tsi, m)?)?;
    m.add_function(wrap_pyfunction!(vortex, m)?)?;
    m.add_function(wrap_pyfunction!(stc, m)?)?;
    m.add_function(wrap_pyfunction!(gator, m)?)?;
    m.add_function(wrap_pyfunction!(coppock, m)?)?;
    Ok(())
}
