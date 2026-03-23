use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlupsidegap2crows<'py>(
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
        let (o1, _h1, _l1, c1) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o2, _h2, _l2, c2) = (opens[i], highs[i], lows[i], closes[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        if is_bullish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && is_bearish(o1, c1)
            && o1 > c0
            && is_bearish(o2, c2)
            && o2 > o1
            && c2 < o1
            && c2 > c0
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
