use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{dema, ema, kama, sma, t3, tema, trima, wma};

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
    match matype {
        0 => sma::sma_inner(py, close, timeperiod),
        1 => ema::ema(py, close, timeperiod),
        2 => wma::wma(py, close, timeperiod),
        3 => dema::dema(py, close, timeperiod),
        4 => tema::tema(py, close, timeperiod),
        5 => trima::trima(py, close, timeperiod),
        6 => kama::kama(py, close, timeperiod),
        7 => t3::t3(py, close, timeperiod, 0.7),
        _ => Err(PyValueError::new_err(
            "matype must be 0–7 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3)",
        )),
    }
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
    let n = prices.len();
    validation::validate_equal_length(&[(n, "close"), (per.len(), "periods")])?;
    validation::validate_timeperiod(minperiod, "minperiod", 1)?;
    validation::validate_timeperiod(maxperiod, "maxperiod", minperiod)?;
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        let p = (per[i].round() as usize).clamp(minperiod, maxperiod);
        if i + 1 >= p {
            let sum: f64 = prices[(i + 1 - p)..=i].iter().sum();
            result[i] = sum / p as f64;
        }
    }
    Ok(result.into_pyarray(py))
}
