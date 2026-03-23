use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdleveningstar<'py>(
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
        let (o1, h1, l1, c1) = (opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]);
        let (o2, _h2, _l2, c2) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o3, h3, l3, c3) = (opens[i], highs[i], lows[i], closes[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let body3 = body_size(o3, c3);
        let range1 = candle_range(h1, l1);
        let range3 = candle_range(h3, l3);

        // Evening star conditions:
        // 1. First candle is a large bullish candle
        // 2. Second candle is a star (small body) gapping above first
        // 3. Third candle is a large bearish candle
        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let small_body2 = range1 > 0.0 && body2 < body1 * 0.3;
        let large_body3 = range3 > 0.0 && body3 >= range3 * 0.6;

        if is_bullish(o1, c1)
            && large_body1
            && small_body2
            && is_bearish(o3, c3)
            && large_body3
            && c3 < (o1 + c1) / 2.0
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
