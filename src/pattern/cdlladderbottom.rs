use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdlladderbottom<'py>(
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
    for i in 4..n {
        let (o0, _h0, _l0, c0) = (opens[i - 4], highs[i - 4], lows[i - 4], closes[i - 4]);
        let (o1, _h1, _l1, c1) = (opens[i - 3], highs[i - 3], lows[i - 3], closes[i - 3]);
        let (o2, _h2, _l2, c2) = (opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]);
        let (o3, h3, _l3, c3) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o4, h4, l4, c4) = (opens[i], highs[i], lows[i], closes[i]);
        let three_bear = is_bearish(o0, c0) && is_bearish(o1, c1) && is_bearish(o2, c2);
        let descend = c1 < c0 && c2 < c1;
        let us3 = upper_shadow(o3, h3, c3);
        let body3 = body_size(o3, c3);
        let inv_hammer = us3 >= body3 * 1.5;
        let range4 = candle_range(h4, l4);
        let body4 = body_size(o4, c4);
        let large_bull = is_bullish(o4, c4) && range4 > 0.0 && body4 >= range4 * 0.5;
        if three_bear && descend && inv_hammer && large_bull && c4 > c2 {
            result[i] = 100;
        }
    }
    Ok(result.into_pyarray(py))
}
