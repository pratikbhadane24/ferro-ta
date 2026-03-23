use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::ExponentialMovingAverage;
use ta::Next;

/// Chaikin A/D Oscillator: fast EMA of AD minus slow EMA of AD.
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, fastperiod = 3, slowperiod = 10))]
pub fn adosc<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let vols = volume.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
        (vols.len(), "volume"),
    ])?;

    // Compute raw AD values
    let mut ad_vals = vec![0.0_f64; n];
    let mut ad_val = 0.0_f64;
    for i in 0..n {
        let hl = highs[i] - lows[i];
        let clv = if hl != 0.0 {
            ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
        } else {
            0.0
        };
        ad_val += clv * vols[i];
        ad_vals[i] = ad_val;
    }

    // Apply fast and slow EMA to AD
    let mut fast_ema = ExponentialMovingAverage::new(fastperiod)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut slow_ema = ExponentialMovingAverage::new(slowperiod)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let warmup = slowperiod - 1;
    let mut result = vec![f64::NAN; n];
    for (i, &v) in ad_vals.iter().enumerate() {
        let fast = fast_ema.next(v);
        let slow = slow_ema.next(v);
        if i >= warmup {
            result[i] = fast - slow;
        }
    }
    Ok(result.into_pyarray(py))
}
