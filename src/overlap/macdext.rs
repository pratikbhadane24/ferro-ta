use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Reject MA types outside the documented 0–8 range instead of silently
/// falling back to SMA.
fn validate_matype(matype: u8, name: &str) -> PyResult<()> {
    if matype > 8 {
        return Err(PyValueError::new_err(format!(
            "{name} must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)"
        )));
    }
    Ok(())
}

/// MACD with configurable MA types for fast/slow/signal (matype 0–8). Returns (macd_line, signal_line, histogram).
///
/// `fastmatype`/`slowmatype`/`signalmatype` default to `1` (EMA), not TA-Lib's
/// `0` (SMA), so the defaults reproduce plain `MACD`.
///
/// `0`–`6` and `8` match TA-Lib's numbering; `7` is T3 here where TA-Lib's `7`
/// is MAMA, and MAMA is not reachable through any `matype` (use `ferro_ta.MAMA`).
#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, fastmatype = 1, slowperiod = 26, slowmatype = 1, signalperiod = 9, signalmatype = 1))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn macdext<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    fastmatype: u8,
    slowperiod: usize,
    slowmatype: u8,
    signalperiod: usize,
    signalmatype: u8,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    validation::validate_timeperiod(signalperiod, "signalperiod", 1)?;
    validate_matype(fastmatype, "fastmatype")?;
    validate_matype(slowmatype, "slowmatype")?;
    validate_matype(signalmatype, "signalmatype")?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let prices = close.as_slice()?;
    let (macd_line, signal_line, histogram) = py.allow_threads(|| {
        ferro_ta_core::overlap::macdext(
            prices,
            fastperiod,
            fastmatype,
            slowperiod,
            slowmatype,
            signalperiod,
            signalmatype,
        )
    });

    Ok((
        macd_line.into_pyarray(py),
        signal_line.into_pyarray(py),
        histogram.into_pyarray(py),
    ))
}
