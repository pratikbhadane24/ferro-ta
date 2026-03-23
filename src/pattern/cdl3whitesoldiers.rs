use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::common::*;

#[pyfunction]
pub fn cdl3whitesoldiers<'py>(
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
        let (o1, h1, l1, c1) = (opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]);
        let (o2, h2, l2, c2) = (opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]);
        let (o3, h3, l3, c3) = (opens[i], highs[i], lows[i], closes[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let body3 = body_size(o3, c3);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(h2, l2);
        let range3 = candle_range(h3, l3);

        let long_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let long_body2 = range2 > 0.0 && body2 >= range2 * 0.5;
        let long_body3 = range3 > 0.0 && body3 >= range3 * 0.5;

        let open2_in_body1 = o2 > o1 && o2 < c1;
        let open3_in_body2 = o3 > o2 && o3 < c2;

        let small_lower1 = lower_shadow(o1, l1, c1) <= body1 * 0.3;
        let small_lower2 = lower_shadow(o2, l2, c2) <= body2 * 0.3;
        let small_lower3 = lower_shadow(o3, l3, c3) <= body3 * 0.3;

        if is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && is_bullish(o3, c3)
            && long_body1
            && long_body2
            && long_body3
            && open2_in_body1
            && open3_in_body2
            && small_lower1
            && small_lower2
            && small_lower3
            && c2 > c1
            && c3 > c2
            && h3 > h2
            && h2 > h1
        {
            result[i] = 100;
        }
    }
    Ok(result.into_pyarray(py))
}
