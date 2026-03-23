use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn compute_rsi_talib(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    if n <= period || period == 0 {
        return result;
    }
    let mut avg_gain = 0.0_f64;
    let mut avg_loss = 0.0_f64;
    for i in 1..=period {
        let delta = prices[i] - prices[i - 1];
        if delta > 0.0 {
            avg_gain += delta;
        } else {
            avg_loss += -delta;
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;
    let rs = if avg_loss == 0.0 {
        f64::MAX
    } else {
        avg_gain / avg_loss
    };
    result[period] = 100.0 - 100.0 / (1.0 + rs);
    let period_f = period as f64;
    for i in (period + 1)..n {
        let delta = prices[i] - prices[i - 1];
        let (gain, loss) = if delta > 0.0 {
            (delta, 0.0)
        } else {
            (0.0, -delta)
        };
        avg_gain = (avg_gain * (period_f - 1.0) + gain) / period_f;
        avg_loss = (avg_loss * (period_f - 1.0) + loss) / period_f;
        let rs = if avg_loss == 0.0 {
            f64::MAX
        } else {
            avg_gain / avg_loss
        };
        result[i] = 100.0 - 100.0 / (1.0 + rs);
    }
    result
}

/// Stochastic RSI (TA-Lib–compatible): stochastic applied to RSI. Returns (fastk, fastd).
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14, fastk_period = 5, fastd_period = 3))]
#[allow(clippy::type_complexity)]
pub fn stochrsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    fastk_period: usize,
    fastd_period: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(fastk_period, "fastk_period", 1)?;
    validation::validate_timeperiod(fastd_period, "fastd_period", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();

    let rsi_vals = compute_rsi_talib(prices, timeperiod);

    let rsi_warmup = timeperiod;
    let k_warmup = rsi_warmup + fastk_period - 1;
    let d_warmup = k_warmup + fastd_period - 1;

    let mut fastk = vec![f64::NAN; n];
    let mut fastd = vec![f64::NAN; n];

    for i in k_warmup..n {
        if rsi_vals[i].is_nan() {
            continue;
        }
        let start = i + 1 - fastk_period;
        if (start..=i).any(|j| rsi_vals[j].is_nan()) {
            continue;
        }
        let mx = rsi_vals[start..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mn = rsi_vals[start..=i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        fastk[i] = if mx != mn {
            100.0 * (rsi_vals[i] - mn) / (mx - mn)
        } else {
            50.0
        };
    }

    for i in d_warmup..n {
        let start = i + 1 - fastd_period;
        let window = &fastk[start..=i];
        if window.iter().all(|v| !v.is_nan()) {
            fastd[i] = window.iter().sum::<f64>() / fastd_period as f64;
        }
    }
    Ok((fastk.into_pyarray(py), fastd.into_pyarray(py)))
}
