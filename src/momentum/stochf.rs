use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Fast Stochastic. Returns (fastk, fastd). %K from high-low range; %D is the
/// MA of %K typed by `fastd_matype` (TA-Lib's `0` = SMA default).
#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period = 5, fastd_period = 3, fastd_matype = 0))]
#[allow(clippy::type_complexity)]
pub fn stochf<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: u8,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(fastk_period, "fastk_period", 1)?;
    validation::validate_timeperiod(fastd_period, "fastd_period", 1)?;
    if fastd_matype > 8 {
        return Err(PyValueError::new_err(
            "fastd_matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
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
    let (fastk, fastd) = py.allow_threads(|| {
        ferro_ta_core::momentum::stochf(
            highs,
            lows,
            closes,
            fastk_period,
            fastd_period,
            fastd_matype,
        )
    });
    Ok((fastk.into_pyarray(py), fastd.into_pyarray(py)))
}
