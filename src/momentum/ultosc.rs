use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Ultimate Oscillator: weighted sum of buying pressure over three periods (7, 14, 28).
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod1 = 7, timeperiod2 = 14, timeperiod3 = 28))]
pub fn ultosc<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod1, "timeperiod1", 1)?;
    validation::validate_timeperiod(timeperiod2, "timeperiod2", 1)?;
    validation::validate_timeperiod(timeperiod3, "timeperiod3", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;

    let max_period = timeperiod1.max(timeperiod2).max(timeperiod3);
    let mut result = vec![f64::NAN; n];

    let mut bp = vec![0.0_f64; n];
    let mut tr = vec![0.0_f64; n];
    for i in 1..n {
        let true_low = lows[i].min(closes[i - 1]);
        let true_high = highs[i].max(closes[i - 1]);
        bp[i] = closes[i] - true_low;
        tr[i] = true_high - true_low;
    }

    for i in max_period..n {
        let raw1 = {
            let sum_bp: f64 = bp[(i + 1 - timeperiod1)..=i].iter().sum();
            let sum_tr: f64 = tr[(i + 1 - timeperiod1)..=i].iter().sum();
            if sum_tr != 0.0 {
                sum_bp / sum_tr
            } else {
                0.0
            }
        };
        let raw2 = {
            let sum_bp: f64 = bp[(i + 1 - timeperiod2)..=i].iter().sum();
            let sum_tr: f64 = tr[(i + 1 - timeperiod2)..=i].iter().sum();
            if sum_tr != 0.0 {
                sum_bp / sum_tr
            } else {
                0.0
            }
        };
        let raw3 = {
            let sum_bp: f64 = bp[(i + 1 - timeperiod3)..=i].iter().sum();
            let sum_tr: f64 = tr[(i + 1 - timeperiod3)..=i].iter().sum();
            if sum_tr != 0.0 {
                sum_bp / sum_tr
            } else {
                0.0
            }
        };
        result[i] = 100.0 * (4.0 * raw1 + 2.0 * raw2 + raw3) / 7.0;
    }
    Ok(result.into_pyarray(py))
}
