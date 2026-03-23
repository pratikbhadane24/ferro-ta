use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlidentical3crows<'py>(
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
        let (o1, h1, l1, c1) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o2, h2, l2, c2) = (opens[i], highs[i], lows[i], closes[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let tol0 = range0 * 0.03;
        let tol1 = range1 * 0.03;
        if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && is_bearish(o2, c2)
            && c1 < c0
            && c2 < c1
            && (o1 - c0).abs() <= tol0
            && (o2 - c1).abs() <= tol1
            && range0 > 0.0
            && range1 > 0.0
            && candle_range(h2, l2) > 0.0
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
