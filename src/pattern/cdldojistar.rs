use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdldojistar<'py>(
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
        let (o2, h2, l2, c2) = (opens[i], highs[i], lows[i], closes[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(h2, l2);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        // Doji: body <= 10% of range
        let is_doji2 = range2 > 0.0 && body2 / range2 <= 0.1;

        // Bullish Doji Star: prior bearish large candle, doji opens/closes below prior low
        let gap_down = o2.max(c2) < l1;
        if is_bearish(o1, c1) && large_body1 && is_doji2 && gap_down {
            result[i] = 100;
        }
        // Bearish Doji Star: prior bullish large candle, doji opens/closes above prior high
        let gap_up = o2.min(c2) > h1;
        if is_bullish(o1, c1) && large_body1 && is_doji2 && gap_up {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
