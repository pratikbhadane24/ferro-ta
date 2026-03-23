use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlengulfing<'py>(
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
        let prev_o = opens[i - 1];
        let prev_c = closes[i - 1];
        let curr_o = opens[i];
        let curr_c = closes[i];

        let prev_body_high = prev_o.max(prev_c);
        let prev_body_low = prev_o.min(prev_c);
        let curr_body_high = curr_o.max(curr_c);
        let curr_body_low = curr_o.min(curr_c);

        // Bullish engulfing: prev is bearish, current is bullish and engulfs
        if is_bearish(prev_o, prev_c)
            && is_bullish(curr_o, curr_c)
            && curr_body_high > prev_body_high
            && curr_body_low < prev_body_low
        {
            result[i] = 100;
        }
        // Bearish engulfing: prev is bullish, current is bearish and engulfs
        else if is_bullish(prev_o, prev_c)
            && is_bearish(curr_o, curr_c)
            && curr_body_high > prev_body_high
            && curr_body_low < prev_body_low
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
