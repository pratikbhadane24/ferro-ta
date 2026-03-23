use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdl2crows<'py>(
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
        let (o3, _h3, _l3, c3) = (opens[i], highs[i], lows[i], closes[i]);

        // Two Crows:
        // 1. First candle is a long white (bullish) candle
        // 2. Second candle gaps up (opens above first close) and closes lower but still above first close
        // 3. Third candle opens within second body and closes within first body
        if is_bullish(o1, c1)
            && is_bearish(o2, c2)
            && o2 > c1   // gap up
            && c2 > c1   // second still closes above first close
            && is_bearish(o3, c3)
            && o3 < o2 && o3 > c2   // opens within second body
            && c3 > o1 && c3 < c1
        // closes within first body
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
