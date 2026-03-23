use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdl3outside<'py>(
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
        let (o1, _h1, _l1, c1) = (opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]);
        let (o2, _h2, _l2, c2) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (_o3, _h3, _l3, c3) = (opens[i], highs[i], lows[i], closes[i]);

        let body1_high = o1.max(c1);
        let body1_low = o1.min(c1);
        let body2_high = o2.max(c2);
        let body2_low = o2.min(c2);

        // Engulfing: candle 2 body completely covers candle 1 body
        let engulfs = body2_high > body1_high && body2_low < body1_low;

        // Three Outside Up: C1 bearish, C2 bullish engulfing, C3 closes above C2
        if is_bearish(o1, c1) && is_bullish(o2, c2) && engulfs && c3 > c2 {
            result[i] = 100;
        }
        // Three Outside Down: C1 bullish, C2 bearish engulfing, C3 closes below C2
        else if is_bullish(o1, c1) && is_bearish(o2, c2) && engulfs && c3 < c2 {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
