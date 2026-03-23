use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlharami<'py>(
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
    for i in 1..n {
        let (o1, h1, l1, c1) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o2, _h2, _l2, c2) = (opens[i], highs[i], lows[i], closes[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let range1 = candle_range(h1, l1);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.5;

        // Current candle body is inside prior candle body
        let body1_high = o1.max(c1);
        let body1_low = o1.min(c1);
        let body2_high = o2.max(c2);
        let body2_low = o2.min(c2);

        let inside = body2_high <= body1_high && body2_low >= body1_low && body2 < body1 * 0.6;

        // Bullish Harami: prior bearish, current bullish inside
        if is_bearish(o1, c1) && large_body1 && inside && is_bullish(o2, c2) {
            result[i] = 100;
        }
        // Bearish Harami: prior bullish, current bearish inside
        else if is_bullish(o1, c1) && large_body1 && inside && is_bearish(o2, c2) {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
