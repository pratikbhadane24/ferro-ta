use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlspinningtop<'py>(
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
    for i in 0..n {
        let body = body_size(opens[i], closes[i]);
        let range = candle_range(highs[i], lows[i]);
        let lower = lower_shadow(opens[i], lows[i], closes[i]);
        let upper = upper_shadow(opens[i], highs[i], closes[i]);

        // Spinning top: small body (< 1/3 range), both shadows longer than body
        if range > 0.0 && body > 0.0 && body <= range / 3.0 && upper > body && lower > body {
            if is_bullish(opens[i], closes[i]) {
                result[i] = 100;
            } else {
                result[i] = -100;
            }
        }
    }
    Ok(result.into_pyarray(py))
}
