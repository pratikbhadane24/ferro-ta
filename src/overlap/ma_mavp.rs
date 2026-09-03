use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Generic Moving Average. matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=T3.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 30, matype = 0))]
pub fn ma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    matype: u8,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    if matype > 7 {
        return Err(PyValueError::new_err(
            "matype must be 0–7 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3)",
        ));
    }
    let prices = close.as_slice()?;
    let result = ferro_ta_core::overlap::ma(prices, timeperiod, matype);
    Ok(result.into_pyarray(py))
}

/// Moving Average with variable period per bar (SMA over period from periods array).
#[pyfunction]
#[pyo3(signature = (close, periods, minperiod = 2, maxperiod = 30))]
pub fn mavp<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    periods: PyReadonlyArray1<'py, f64>,
    minperiod: usize,
    maxperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = close.as_slice()?;
    let per = periods.as_slice()?;
    validation::validate_equal_length(&[(prices.len(), "close"), (per.len(), "periods")])?;
    validation::validate_timeperiod(minperiod, "minperiod", 1)?;
    validation::validate_timeperiod(maxperiod, "maxperiod", minperiod)?;
    let result =
        py.allow_threads(|| ferro_ta_core::overlap::mavp(prices, per, minperiod, maxperiod));
    Ok(result.into_pyarray(py))
}
