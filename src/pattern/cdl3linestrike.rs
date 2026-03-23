use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdl3linestrike<'py>(
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
    for i in 3..n {
        let (o0, c0) = (opens[i - 3], closes[i - 3]);
        let (o1, c1) = (opens[i - 2], closes[i - 2]);
        let (o2, c2) = (opens[i - 1], closes[i - 1]);
        let (o3, c3) = (opens[i], closes[i]);
        if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && is_bearish(o2, c2)
            && c1 < c0
            && c2 < c1
            && is_bullish(o3, c3)
            && o3 < c2
            && c3 > o0
        {
            result[i] = 100;
        } else if is_bullish(o0, c0)
            && is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && c1 > c0
            && c2 > c1
            && is_bearish(o3, c3)
            && o3 > c2
            && c3 < o0
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
