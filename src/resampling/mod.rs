//! Resampling — OHLCV resampling (thin PyO3 wrapper over ferro_ta_core::resampling).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Ohlcv5<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Aggregate OHLCV data into volume bars of a fixed volume threshold.
#[pyfunction]
#[pyo3(signature = (open, high, low, close, volume, volume_threshold))]
pub fn volume_bars<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    volume_threshold: f64,
) -> PyResult<Ohlcv5<'py>> {
    if volume_threshold <= 0.0 {
        return Err(PyValueError::new_err("volume_threshold must be > 0"));
    }
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    let n = o.len();
    if n == 0 || h.len() != n || l.len() != n || c.len() != n || v.len() != n {
        return Err(PyValueError::new_err(
            "All input arrays must be non-empty and have equal length",
        ));
    }
    let (ro, rh, rl, rc, rv) = ferro_ta_core::resampling::volume_bars(o, h, l, c, v, volume_threshold);
    Ok((
        ro.into_pyarray(py),
        rh.into_pyarray(py),
        rl.into_pyarray(py),
        rc.into_pyarray(py),
        rv.into_pyarray(py),
    ))
}

/// Aggregate OHLCV bars by integer group labels.
#[pyfunction]
#[pyo3(signature = (open, high, low, close, volume, labels))]
pub fn ohlcv_agg<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Ohlcv5<'py>> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    let lbl = labels.as_slice()?;
    let n = o.len();
    if n == 0 || h.len() != n || l.len() != n || c.len() != n || v.len() != n || lbl.len() != n {
        return Err(PyValueError::new_err(
            "All input arrays must be non-empty and have equal length",
        ));
    }
    let (ro, rh, rl, rc, rv) = ferro_ta_core::resampling::ohlcv_agg(o, h, l, c, v, lbl);
    Ok((
        ro.into_pyarray(py),
        rh.into_pyarray(py),
        rl.into_pyarray(py),
        rc.into_pyarray(py),
        rv.into_pyarray(py),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(volume_bars, m)?)?;
    m.add_function(wrap_pyfunction!(ohlcv_agg, m)?)?;
    Ok(())
}
