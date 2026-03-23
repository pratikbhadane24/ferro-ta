use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlgapsidesidewhite<'py>(
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
        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let both_bullish = is_bullish(o1, c1) && is_bullish(o2, c2);
        let similar_size = body1 > 0.0 && (body2 - body1).abs() / body1 <= 0.3;
        let similar_open = body1 > 0.0 && (o2 - o1).abs() / body1 <= 0.3;
        if is_bullish(o0, c0) && both_bullish && similar_size && similar_open && o1 > c0 {
            result[i] = 100;
        } else if is_bearish(o0, c0) && both_bullish && similar_size && similar_open && c1 < o0 {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
