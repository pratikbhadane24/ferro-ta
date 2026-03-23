use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn compute_ma_slice(prices: &[f64], period: usize, matype: u8) -> Vec<f64> {
    let n = prices.len();
    match matype {
        1 => {
            if period == 0 {
                return vec![f64::NAN; n];
            }
            let k = 2.0 / (period as f64 + 1.0);
            let mut result = vec![f64::NAN; n];
            let mut ema_val = prices[period - 1];
            result[period - 1] = ema_val;
            for i in period..n {
                ema_val = prices[i] * k + ema_val * (1.0 - k);
                result[i] = ema_val;
            }
            result
        }
        2 => {
            if period == 0 {
                return vec![f64::NAN; n];
            }
            let weight_sum = (period * (period + 1) / 2) as f64;
            let mut result = vec![f64::NAN; n];
            for i in (period - 1)..n {
                let val: f64 = (0..period)
                    .map(|j| prices[i - j] * (period - j) as f64)
                    .sum();
                result[i] = val / weight_sum;
            }
            result
        }
        _ => {
            if period == 0 {
                return vec![f64::NAN; n];
            }
            let mut result = vec![f64::NAN; n];
            for i in (period - 1)..n {
                let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
                result[i] = sum / period as f64;
            }
            result
        }
    }
}

/// MACD with configurable MA types for fast/slow/signal (matype 0–7). Returns (macd_line, signal_line, histogram).
#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, fastmatype = 1, slowperiod = 26, slowmatype = 1, signalperiod = 9, signalmatype = 1))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn macdext<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    fastmatype: u8,
    slowperiod: usize,
    slowmatype: u8,
    signalperiod: usize,
    signalmatype: u8,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    validation::validate_timeperiod(signalperiod, "signalperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(
            "fastperiod must be less than slowperiod",
        ));
    }
    let prices = close.as_slice()?;
    let n = prices.len();

    let fast_ma = compute_ma_slice(prices, fastperiod, fastmatype);
    let slow_ma = compute_ma_slice(prices, slowperiod, slowmatype);

    let mut macd_line = vec![f64::NAN; n];
    let macd_start = slowperiod - 1;
    for i in macd_start..n {
        if !fast_ma[i].is_nan() && !slow_ma[i].is_nan() {
            macd_line[i] = fast_ma[i] - slow_ma[i];
        }
    }

    let macd_valid: Vec<f64> = macd_line[macd_start..].to_vec();
    let signal_slice = compute_ma_slice(&macd_valid, signalperiod, signalmatype);

    let mut signal_line = vec![f64::NAN; n];
    let warmup = macd_start + signalperiod - 1;
    #[allow(clippy::needless_range_loop)]
    for i in warmup..n {
        let j = i - macd_start;
        if j < signal_slice.len() && !signal_slice[j].is_nan() {
            signal_line[i] = signal_slice[j];
        }
    }

    let mut histogram = vec![f64::NAN; n];
    for i in 0..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    Ok((
        macd_line.into_pyarray(py),
        signal_line.into_pyarray(py),
        histogram.into_pyarray(py),
    ))
}
