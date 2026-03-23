use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlhikkake<'py>(
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
        let h1 = highs[i - 1];
        let l1 = lows[i - 1];
        let h2 = highs[i];
        let l2 = lows[i];
        let inside = h1 <= h0 && l1 >= l0;
        if !inside {
            continue;
        }
        if is_bearish(o0, c0) && h2 > h1 && l2 > l1 {
            result[i] = 100;
        } else if is_bullish(o0, c0) && l2 < l1 && h2 < h1 {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
