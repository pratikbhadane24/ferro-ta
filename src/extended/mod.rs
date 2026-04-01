//! Extended Indicators — thin PyO3 wrappers delegating to `ferro_ta_core::extended`.
//!
//! All compute-heavy work lives in the core crate.  These functions convert
//! numpy arrays to slices, call the core, and convert the results back.

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// VWAP
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, timeperiod = 0))]
pub fn vwap<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
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
    let result = ferro_ta_core::extended::vwap(h, lo, c, v, timeperiod);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// VWMA
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (close, volume, timeperiod = 20))]
pub fn vwma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (v.len(), "volume")])?;
    let result = ferro_ta_core::extended::vwma(c, v, timeperiod);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// SUPERTREND
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 7, multiplier = 3.0))]
pub fn supertrend<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    multiplier: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i8>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (st, dir) = ferro_ta_core::extended::supertrend(h, lo, c, timeperiod, multiplier);
    Ok((st.into_pyarray(py), dir.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// DONCHIAN
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 20))]
pub fn donchian<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low")])?;
    let (upper, middle, lower) = ferro_ta_core::extended::donchian(h, lo, timeperiod);
    Ok((
        upper.into_pyarray(py),
        middle.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// CHOPPINESS_INDEX
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn choppiness_index<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let result = ferro_ta_core::extended::choppiness_index(h, lo, c, timeperiod);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// KELTNER_CHANNELS
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 20, atr_period = 10, multiplier = 2.0))]
pub fn keltner_channels<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    atr_period: usize,
    multiplier: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(atr_period, "atr_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (upper, middle, lower) =
        ferro_ta_core::extended::keltner_channels(h, lo, c, timeperiod, atr_period, multiplier);
    Ok((
        upper.into_pyarray(py),
        middle.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// HULL_MA
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 16))]
pub fn hull_ma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let c = close.as_slice()?;
    let result = ferro_ta_core::extended::hull_ma(c, timeperiod);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// CHANDELIER_EXIT
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 22, multiplier = 3.0))]
pub fn chandelier_exit<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    multiplier: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (long_exit, short_exit) =
        ferro_ta_core::extended::chandelier_exit(h, lo, c, timeperiod, multiplier);
    Ok((long_exit.into_pyarray(py), short_exit.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// ICHIMOKU
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, tenkan_period = 9, kijun_period = 26, senkou_b_period = 52, displacement = 26))]
pub fn ichimoku<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
    displacement: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(tenkan_period, "tenkan_period", 1)?;
    validation::validate_timeperiod(kijun_period, "kijun_period", 1)?;
    validation::validate_timeperiod(senkou_b_period, "senkou_b_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;
    let (tenkan, kijun, senkou_a, senkou_b, chikou) = ferro_ta_core::extended::ichimoku(
        h,
        lo,
        c,
        tenkan_period,
        kijun_period,
        senkou_b_period,
        displacement,
    );
    Ok((
        tenkan.into_pyarray(py),
        kijun.into_pyarray(py),
        senkou_a.into_pyarray(py),
        senkou_b.into_pyarray(py),
        chikou.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// PIVOT_POINTS
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, method = "classic"))]
pub fn pivot_points<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    validation::validate_equal_length(&[(h.len(), "high"), (lo.len(), "low"), (c.len(), "close")])?;

    let method_lower = method.to_lowercase();
    if !matches!(method_lower.as_str(), "classic" | "fibonacci" | "camarilla") {
        return Err(PyValueError::new_err(format!(
            "Unknown pivot method '{}'. Use 'classic', 'fibonacci', or 'camarilla'.",
            method
        )));
    }

    let (pivot, r1, s1, r2, s2) = ferro_ta_core::extended::pivot_points(h, lo, c, method);
    Ok((
        pivot.into_pyarray(py),
        r1.into_pyarray(py),
        s1.into_pyarray(py),
        r2.into_pyarray(py),
        s2.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(vwap, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(vwma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(supertrend, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(donchian, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(choppiness_index, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(keltner_channels, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(hull_ma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(chandelier_exit, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ichimoku, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(pivot_points, m)?)?;
    Ok(())
}
