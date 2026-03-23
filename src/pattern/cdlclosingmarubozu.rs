use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlclosingmarubozu<'py>(
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
        if body < range * 0.4 {
            continue;
        }
        if is_bullish(o, c) && (h - c).abs() <= range * 0.01 {
            result[i] = 100;
        } else if is_bearish(o, c) && (c - l).abs() <= range * 0.01 {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
