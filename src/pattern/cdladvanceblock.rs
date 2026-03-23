use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdladvanceblock<'py>(
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
        let (o0, h0, _l0, c0) = (opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]);
        let (o1, h1, _l1, c1) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o2, h2, _l2, c2) = (opens[i], highs[i], lows[i], closes[i]);
        let body0 = body_size(o0, c0);
        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let us0 = upper_shadow(o0, h0, c0);
        let us1 = upper_shadow(o1, h1, c1);
        let us2 = upper_shadow(o2, h2, c2);
        if is_bullish(o0, c0)
            && is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && c1 > c0
            && c2 > c1
            && o1 >= o0
            && o1 <= c0
            && o2 >= o1
            && o2 <= c1
            && (body1 < body0 || body2 < body1 || us2 > us1 || us1 > us0)
        {
            result[i] = -100;
        }
    }
    Ok(result.into_pyarray(py))
}
