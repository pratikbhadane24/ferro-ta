use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlbreakaway<'py>(
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
        let c1 = closes[i - 3];
        let c2 = closes[i - 2];
        let c3 = closes[i - 1];
        let _l3 = lows[i - 1];
        let _h3 = highs[i - 1];
        let (o4, c4) = (opens[i], closes[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && c1 < l0
            && c2 < c1
            && c3 < c2
            && is_bullish(o4, c4)
            && c4 > c1
            && c4 < c0
        {
            result[i] = 100;
        } else if is_bullish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && c1 > h0
            && c2 > c1
            && c3 > c2
            && is_bearish(o4, c4)
            && c4 < c1
            && c4 > c0
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
