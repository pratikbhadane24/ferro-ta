use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn correl_fallback(x: &[f64], y: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = x.len();
    let mut result = vec![f64::NAN; n];
    for (end, slot) in result.iter_mut().enumerate().take(n).skip(timeperiod - 1) {
        let wx = &x[(end + 1 - timeperiod)..=end];
        let wy = &y[(end + 1 - timeperiod)..=end];
        let mean_x = wx.iter().sum::<f64>() / timeperiod as f64;
        let mean_y = wy.iter().sum::<f64>() / timeperiod as f64;
        let cov = wx
            .iter()
            .zip(wy.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>();
        let std_x = wx
            .iter()
            .map(|&xi| (xi - mean_x).powi(2))
            .sum::<f64>()
            .sqrt();
        let std_y = wy
            .iter()
            .map(|&yi| (yi - mean_y).powi(2))
            .sum::<f64>()
            .sqrt();
        let denom = std_x * std_y;
        *slot = if denom != 0.0 { cov / denom } else { f64::NAN };
    }
    result
}

/// Pearson correlation coefficient between two series over the rolling window.
#[pyfunction]
#[pyo3(signature = (real0, real1, timeperiod = 30))]
pub fn correl<'py>(
    py: Python<'py>,
    real0: PyReadonlyArray1<'py, f64>,
    real1: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let x = real0.as_slice()?;
    let y = real1.as_slice()?;
    let n = x.len();
    validation::validate_equal_length(&[(n, "real0"), (y.len(), "real1")])?;

    if x.iter().any(|value| !value.is_finite()) || y.iter().any(|value| !value.is_finite()) {
        return Ok(correl_fallback(x, y, timeperiod).into_pyarray(py));
    }

    let mut result = vec![f64::NAN; n];
    if n < timeperiod {
        return Ok(result.into_pyarray(py));
    }

    let period = timeperiod as f64;
    let mut sum_x = x[..timeperiod].iter().sum::<f64>();
    let mut sum_y = y[..timeperiod].iter().sum::<f64>();
    let mut sum_x2 = x[..timeperiod]
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let mut sum_y2 = y[..timeperiod]
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let mut sum_xy = x[..timeperiod]
        .iter()
        .zip(y[..timeperiod].iter())
        .map(|(&lhs, &rhs)| lhs * rhs)
        .sum::<f64>();

    for (end, slot) in result.iter_mut().enumerate().take(n).skip(timeperiod - 1) {
        let denom_x = period * sum_x2 - sum_x * sum_x;
        let denom_y = period * sum_y2 - sum_y * sum_y;
        *slot = if denom_x > 0.0 && denom_y > 0.0 {
            (period * sum_xy - sum_x * sum_y) / (denom_x * denom_y).sqrt()
        } else {
            f64::NAN
        };

        if end + 1 < n {
            let outgoing = end + 1 - timeperiod;
            let incoming = end + 1;

            let outgoing_x = x[outgoing];
            let outgoing_y = y[outgoing];
            let incoming_x = x[incoming];
            let incoming_y = y[incoming];

            sum_x += incoming_x - outgoing_x;
            sum_y += incoming_y - outgoing_y;
            sum_x2 += incoming_x * incoming_x - outgoing_x * outgoing_x;
            sum_y2 += incoming_y * incoming_y - outgoing_y * outgoing_y;
            sum_xy += incoming_x * incoming_y - outgoing_x * outgoing_y;
        }
    }
    Ok(result.into_pyarray(py))
}
