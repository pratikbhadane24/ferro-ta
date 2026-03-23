use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlrickshawman<'py>(
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
        let (o, h, l, c) = (opens[i], highs[i], lows[i], closes[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        let body_mid = (o + c) / 2.0;
        let range_mid = (h + l) / 2.0;
        let is_doji = body / range <= 0.1;
        let long_shadows = us >= range * 0.3 && ls >= range * 0.3;
        let near_center = (body_mid - range_mid).abs() <= range * 0.15;
        if is_doji && long_shadows && near_center {
            result[i] = 100;
        }
    }
    Ok(result.into_pyarray(py))
}
