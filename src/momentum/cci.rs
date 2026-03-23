use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Commodity Channel Index (TA-Lib–compatible): (typical_price - SMA) / (0.015 * MAD).
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn cci<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let tp: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .zip(closes.iter())
        .map(|((&h, &l), &c)| (h + l + c) / 3.0)
        .collect();
    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let window = &tp[(i + 1 - timeperiod)..=i];
        let mean: f64 = window.iter().sum::<f64>() / timeperiod as f64;
        let mad: f64 = window.iter().map(|&x| (x - mean).abs()).sum::<f64>() / timeperiod as f64;
        result[i] = if mad != 0.0 {
            (tp[i] - mean) / (0.015 * mad)
        } else {
            0.0
        };
    }
    Ok(result.into_pyarray(py))
}
