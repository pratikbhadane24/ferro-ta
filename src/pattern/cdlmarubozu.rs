use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlmarubozu<'py>(
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

        // Marubozu: body is >= 95% of range, tiny or no shadows
        if range > 0.0 && body >= range * 0.95 && upper <= range * 0.025 && lower <= range * 0.025 {
            if is_bullish(opens[i], closes[i]) {
                result[i] = 100;
            } else {
                result[i] = -100;
            }
        }
    }
    Ok(result.into_pyarray(py))
}
