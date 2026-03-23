use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// True Range: max(high - low, |high - prev_close|, |low - prev_close|). Bar 0 uses high - low.
#[pyfunction]
pub fn trange<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return Ok(result.into_pyarray(py));
    }
    result[0] = highs[0] - lows[0];
    for i in 1..n {
        let hl = highs[i] - lows[i];
        let hpc = (highs[i] - closes[i - 1]).abs();
        let lpc = (lows[i] - closes[i - 1]).abs();
        result[i] = hl.max(hpc).max(lpc);
    }
    Ok(result.into_pyarray(py))
}
