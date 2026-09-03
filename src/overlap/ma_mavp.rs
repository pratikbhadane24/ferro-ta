use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Generic Moving Average. matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=T3, 8=T3.
///
/// `0`–`6` and `8` match TA-Lib's numbering; `7` is T3 here where TA-Lib's `7`
/// is MAMA, and MAMA is not reachable through any `matype` (use `ferro_ta.MAMA`).
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30, matype = 0))]
pub fn ma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    matype: u8,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    if matype > 8 {
        return Err(PyValueError::new_err(
            "matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
        ));
    }
    let prices = close.as_slice()?;
    let result = ferro_ta_core::overlap::ma(prices, timeperiod, matype);
    Ok(result.into_pyarray(py))
}

/// Moving Average with variable period per bar (MA of type `matype` over the
/// per-bar period from the `periods` array).
#[pyfunction]
#[pyo3(signature = (close, periods, minperiod = 2, maxperiod = 30, matype = 0))]
pub fn mavp<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    periods: PyReadonlyArray1<'py, f64>,
    minperiod: usize,
    maxperiod: usize,
    matype: u8,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = close.as_slice()?;
    let per = periods.as_slice()?;
    validation::validate_equal_length(&[(prices.len(), "close"), (per.len(), "periods")])?;
    validation::validate_timeperiod(minperiod, "minperiod", 1)?;
    validation::validate_timeperiod(maxperiod, "maxperiod", minperiod)?;
    if matype > 8 {
        return Err(PyValueError::new_err(
            "matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
        ));
    }
    let result = py
        .allow_threads(|| ferro_ta_core::overlap::mavp(prices, per, minperiod, maxperiod, matype));
    Ok(result.into_pyarray(py))
}
