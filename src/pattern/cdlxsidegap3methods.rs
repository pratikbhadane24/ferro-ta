use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlxsidegap3methods<'py>(
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
    for i in 2..n {
        let (o0, _h0, _l0, c0) = (opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]);
        let (o1, _h1, _l1, c1) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o2, _h2, _l2, c2) = (opens[i], highs[i], lows[i], closes[i]);
        if is_bullish(o0, c0)
            && is_bullish(o1, c1)
            && o1 > c0
            && is_bearish(o2, c2)
            && o2 <= c1
            && o2 >= o1
            && c2 >= c0
            && c2 <= o1
        {
            result[i] = 100;
        } else if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && o1 < c0
            && is_bullish(o2, c2)
            && o2 >= c1
            && o2 <= o1
            && c2 <= c0
            && c2 >= o1
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
