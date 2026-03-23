use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlhangingman<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let opens = open.as_slice()?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = opens.len();
    super::common::validate_ohlc_length(n, highs.len(), lows.len(), closes.len())?;
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (opens[i], highs[i], lows[i], closes[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        if range > 0.0 && body > 0.0 && ls >= body * 2.0 && us <= body && body / range <= 0.4 {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
