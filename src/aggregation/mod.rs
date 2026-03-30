//! Tick/trade aggregation (thin PyO3 wrapper over ferro_ta_core::aggregation).

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

type Ohlcv5AndLabels<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
);

/// Aggregate tick/trade data into tick bars (every N ticks become one bar).
#[pyfunction]
#[pyo3(signature = (price, size, ticks_per_bar))]
pub fn aggregate_tick_bars<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    size: PyReadonlyArray1<'py, f64>,
    ticks_per_bar: usize,
) -> PyResult<Ohlcv5<'py>> {
    if ticks_per_bar == 0 {
        return Err(PyValueError::new_err("ticks_per_bar must be >= 1"));
    }
    let p = price.as_slice()?;
    let s = size.as_slice()?;
    let n = p.len();
    if n == 0 || s.len() != n {
        return Err(PyValueError::new_err(
            "price and size must be non-empty and equal length",
        ));
    }
    let (ro, rh, rl, rc, rv) = ferro_ta_core::aggregation::aggregate_tick_bars(p, s, ticks_per_bar);
    Ok((
        ro.into_pyarray(py),
        rh.into_pyarray(py),
        rl.into_pyarray(py),
        rc.into_pyarray(py),
        rv.into_pyarray(py),
    ))
}

/// Aggregate tick data into volume bars (fixed volume threshold).
#[pyfunction]
#[pyo3(signature = (price, size, volume_threshold))]
pub fn aggregate_volume_bars_ticks<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    size: PyReadonlyArray1<'py, f64>,
    volume_threshold: f64,
) -> PyResult<Ohlcv5<'py>> {
    if volume_threshold <= 0.0 {
        return Err(PyValueError::new_err("volume_threshold must be > 0"));
    }
    let p = price.as_slice()?;
    let s = size.as_slice()?;
    let n = p.len();
    if n == 0 || s.len() != n {
        return Err(PyValueError::new_err(
            "price and size must be non-empty and equal length",
        ));
    }
    let (ro, rh, rl, rc, rv) = ferro_ta_core::aggregation::aggregate_volume_bars_ticks(p, s, volume_threshold);
    Ok((
        ro.into_pyarray(py),
        rh.into_pyarray(py),
        rl.into_pyarray(py),
        rc.into_pyarray(py),
        rv.into_pyarray(py),
    ))
}

/// Aggregate tick data into time bars using pre-computed integer bucket labels.
#[pyfunction]
#[pyo3(signature = (price, size, labels))]
pub fn aggregate_time_bars<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    size: PyReadonlyArray1<'py, f64>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Ohlcv5AndLabels<'py>> {
    let p = price.as_slice()?;
    let s = size.as_slice()?;
    let lbl = labels.as_slice()?;
    let n = p.len();
    if n == 0 || s.len() != n || lbl.len() != n {
        return Err(PyValueError::new_err(
            "price, size, and labels must be non-empty and equal length",
        ));
    }
    let (ro, rh, rl, rc, rv, rlbl) = ferro_ta_core::aggregation::aggregate_time_bars(p, s, lbl);
    Ok((
        ro.into_pyarray(py),
        rh.into_pyarray(py),
        rl.into_pyarray(py),
        rc.into_pyarray(py),
        rv.into_pyarray(py),
        rlbl.into_pyarray(py),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_tick_bars, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_volume_bars_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_time_bars, m)?)?;
    Ok(())
}
