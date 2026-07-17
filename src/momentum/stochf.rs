use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Fast Stochastic. Returns (fastk, fastd). %K from high-low range; %D is the
/// SMA of %K (TA-Lib's `fastd_matype=0` default).
#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period = 5, fastd_period = 3))]
#[allow(clippy::type_complexity)]
pub fn stochf<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fastk_period: usize,
    fastd_period: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(fastk_period, "fastk_period", 1)?;
    validation::validate_timeperiod(fastd_period, "fastd_period", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;

    // STOCHF(fastk, fastd) == STOCH(fastk, slowk_period = 1, slowd_period = fastd),
    // which reuses core's TA-Lib-compatible SMA smoothing and NaN padding.
    let (fastk, fastd) = py.allow_threads(|| {
        ferro_ta_core::momentum::stoch(highs, lows, closes, fastk_period, 1, fastd_period)
    });
    Ok((fastk.into_pyarray(py), fastd.into_pyarray(py)))
}
