use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Slow Stochastic. Returns (slowk, slowd). Matches TA-Lib: Fast %K raw, Slow %K = MA(fast %K, slowk_period, slowk_matype), Slow %D = MA(slow %K, slowd_period, slowd_matype).
/// Uses O(n) sliding max/min via monotonic deques.
///
/// The argument order is TA-Lib's *interleaved* one — each matype immediately
/// after the period it types — so a mis-ordered positional call is a type
/// error rather than a silently wrong answer.
#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period = 5, slowk_period = 3, slowk_matype = 0, slowd_period = 3, slowd_matype = 0))]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn stoch<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fastk_period: usize,
    slowk_period: usize,
    slowk_matype: u8,
    slowd_period: usize,
    slowd_matype: u8,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(fastk_period, "fastk_period", 1)?;
    validation::validate_timeperiod(slowk_period, "slowk_period", 1)?;
    validation::validate_timeperiod(slowd_period, "slowd_period", 1)?;
    if slowk_matype > 8 {
        return Err(PyValueError::new_err(
            "slowk_matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
        ));
    }
    if slowd_matype > 8 {
        return Err(PyValueError::new_err(
            "slowd_matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
        ));
    }
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let (slowk, slowd) = py.allow_threads(|| {
        ferro_ta_core::momentum::stoch(
            highs,
            lows,
            closes,
            fastk_period,
            slowk_period,
            slowk_matype,
            slowd_period,
            slowd_matype,
        )
    });
    Ok((slowk.into_pyarray(py), slowd.into_pyarray(py)))
}
