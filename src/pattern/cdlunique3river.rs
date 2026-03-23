use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlunique3river<'py>(
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
        let (o0, h0, l0, c0) = (opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]);
        let (o1, _h1, l1, c1) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o2, h2, l2, c2) = (opens[i], highs[i], lows[i], closes[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        let body2 = body_size(o2, c2);
        let range2 = candle_range(h2, l2);
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && is_bearish(o1, c1)
            && l1 < l0
            && lower_shadow(o1, l1, c1) > 0.0
            && is_bullish(o2, c2)
            && range2 > 0.0
            && body2 <= range2 * 0.5
            && c2 < c1
            && c2 > l1
        {
            result[i] = 100;
        }
    }
    Ok(result.into_pyarray(py))
}
