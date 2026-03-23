use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Slow Stochastic. Returns (slowk, slowd). Matches TA-Lib: Fast %K raw, Slow %K = SMA(fast %K, slowk_period), Slow %D = SMA(slow %K, slowd_period).
/// Uses O(n) sliding max/min via monotonic deques.
#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period = 5, slowk_period = 3, slowd_period = 3))]
#[allow(clippy::type_complexity)]
pub fn stoch<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(fastk_period, "fastk_period", 1)?;
    validation::validate_timeperiod(slowk_period, "slowk_period", 1)?;
    validation::validate_timeperiod(slowd_period, "slowd_period", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let (slowk, slowd) = ferro_ta_core::momentum::stoch(
        highs,
        lows,
        closes,
        fastk_period,
        slowk_period,
        slowd_period,
    );
    Ok((slowk.into_pyarray(py), slowd.into_pyarray(py)))
}
