use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn price_return(curr: f64, prev: f64) -> f64 {
    if prev != 0.0 {
        curr / prev - 1.0
    } else {
        f64::NAN
    }
}

fn beta_fallback(x: &[f64], y: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = x.len();
    let mut result = vec![f64::NAN; n];
    for (end, slot) in result.iter_mut().enumerate().take(n).skip(timeperiod) {
        let start = end - timeperiod;
        let mut rx = vec![0.0_f64; timeperiod];
        let mut ry = vec![0.0_f64; timeperiod];
        for offset in 0..timeperiod {
            let prev = start + offset;
            let curr = prev + 1;
            rx[offset] = price_return(x[curr], x[prev]);
            ry[offset] = price_return(y[curr], y[prev]);
        }
        let mean_x = rx.iter().sum::<f64>() / timeperiod as f64;
        let mean_y = ry.iter().sum::<f64>() / timeperiod as f64;
        let cov = rx
            .iter()
            .zip(ry.iter())
            .map(|(&lhs, &rhs)| (lhs - mean_x) * (rhs - mean_y))
            .sum::<f64>()
            / timeperiod as f64;
        let var_x = rx
            .iter()
            .map(|&value| (value - mean_x).powi(2))
            .sum::<f64>()
            / timeperiod as f64;
        *slot = if var_x != 0.0 { cov / var_x } else { f64::NAN };
    }
    result
}

/// Beta: regression of *real1* daily returns on *real0* daily returns over a
/// rolling window of *timeperiod* return pairs.
///
/// Matches TA-Lib's algorithm:
///   - For bar *i* (output index *i*): use `timeperiod` pairs of consecutive
///     price returns from the window ending at bar *i*.
///   - Return for bar t: r_x[t] = x[t]/x[t-1] - 1 (similarly for y).
///   - beta = Cov(r_y, r_x) / Var(r_x)  (sample, divided by timeperiod).
///   - First valid output is at index `timeperiod` (needs `timeperiod+1` bars).
#[pyfunction]
#[pyo3(signature = (real0, real1, timeperiod = 5))]
pub fn beta<'py>(
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
        return Ok(beta_fallback(x, y, timeperiod).into_pyarray(py));
    }

    let mut result = vec![f64::NAN; n];
    if n <= timeperiod {
        return Ok(result.into_pyarray(py));
    }

    let rx: Vec<f64> = x
        .windows(2)
        .map(|window| price_return(window[1], window[0]))
        .collect();
    let ry: Vec<f64> = y
        .windows(2)
        .map(|window| price_return(window[1], window[0]))
        .collect();

    let period = timeperiod as f64;
    let mut invalid_pairs = 0_usize;
    let mut sum_rx = 0.0_f64;
    let mut sum_ry = 0.0_f64;
    let mut sum_rx2 = 0.0_f64;
    let mut sum_rxry = 0.0_f64;

    for idx in 0..timeperiod {
        let ret_x = rx[idx];
        let ret_y = ry[idx];
        if ret_x.is_finite() && ret_y.is_finite() {
            sum_rx += ret_x;
            sum_ry += ret_y;
            sum_rx2 += ret_x * ret_x;
            sum_rxry += ret_x * ret_y;
        } else {
            invalid_pairs += 1;
        }
    }

    for (end, slot) in result.iter_mut().enumerate().take(n).skip(timeperiod) {
        *slot = if invalid_pairs == 0 {
            let denom = period * sum_rx2 - sum_rx * sum_rx;
            if denom != 0.0 {
                (period * sum_rxry - sum_rx * sum_ry) / denom
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };

        if end + 1 < n {
            let outgoing = end - timeperiod;
            let incoming = end;

            let outgoing_x = rx[outgoing];
            let outgoing_y = ry[outgoing];
            if outgoing_x.is_finite() && outgoing_y.is_finite() {
                sum_rx -= outgoing_x;
                sum_ry -= outgoing_y;
                sum_rx2 -= outgoing_x * outgoing_x;
                sum_rxry -= outgoing_x * outgoing_y;
            } else {
                invalid_pairs -= 1;
            }

            let incoming_x = rx[incoming];
            let incoming_y = ry[incoming];
            if incoming_x.is_finite() && incoming_y.is_finite() {
                sum_rx += incoming_x;
                sum_ry += incoming_y;
                sum_rx2 += incoming_x * incoming_x;
                sum_rxry += incoming_x * incoming_y;
            } else {
                invalid_pairs += 1;
            }
        }
    }

    Ok(result.into_pyarray(py))
}
