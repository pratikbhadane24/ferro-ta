use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlhomingpigeon<'py>(
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
        let (o0, _h0, _l0, c0) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o1, _h1, _l1, c1) = (opens[i], highs[i], lows[i], closes[i]);
        let body0_high = o0.max(c0);
        let body0_low = o0.min(c0);
        let body1_high = o1.max(c1);
        let body1_low = o1.min(c1);
        if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && body1_high <= body0_high
            && body1_low >= body0_low
        {
            result[i] = 100;
        }
    }
    Ok(result.into_pyarray(py))
}
