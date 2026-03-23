use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlcounterattack<'py>(
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
        let (o0, h0, l0, c0) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o1, h1, l1, c1) = (opens[i], highs[i], lows[i], closes[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let body0 = body_size(o0, c0);
        let body1 = body_size(o1, c1);
        let long0 = range0 > 0.0 && body0 >= range0 * 0.5;
        let long1 = range1 > 0.0 && body1 >= range1 * 0.5;
        let same_close = (c1 - c0).abs() <= range0 * 0.02;
        if is_bearish(o0, c0) && long0 && is_bullish(o1, c1) && long1 && same_close {
            result[i] = 100;
        } else if is_bullish(o0, c0) && long0 && is_bearish(o1, c1) && long1 && same_close {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
