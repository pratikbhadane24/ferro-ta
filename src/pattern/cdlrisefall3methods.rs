use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlrisefall3methods<'py>(
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
    for i in 4..n {
        let (o0, h0, l0, c0) = (opens[i - 4], highs[i - 4], lows[i - 4], closes[i - 4]);
        let (o1, h1, l1, c1) = (opens[i - 3], highs[i - 3], lows[i - 3], closes[i - 3]);
        let (o2, h2, l2, c2) = (opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]);
        let (o3, h3, l3, c3) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o4, h4, l4, c4) = (opens[i], highs[i], lows[i], closes[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        let range4 = candle_range(h4, l4);
        let body4 = body_size(o4, c4);
        if is_bullish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.5
            && is_bearish(o1, c1)
            && is_bearish(o2, c2)
            && is_bearish(o3, c3)
            && h1 <= h0
            && l1 >= l0
            && h2 <= h0
            && l2 >= l0
            && h3 <= h0
            && l3 >= l0
            && is_bullish(o4, c4)
            && range4 > 0.0
            && body4 >= range4 * 0.5
            && c4 > c0
            && o4 > c3
        {
            result[i] = 100;
        } else if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.5
            && is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && is_bullish(o3, c3)
            && h1 <= h0
            && l1 >= l0
            && h2 <= h0
            && l2 >= l0
            && h3 <= h0
            && l3 >= l0
            && is_bearish(o4, c4)
            && range4 > 0.0
            && body4 >= range4 * 0.5
            && c4 < c0
            && o4 < c3
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
