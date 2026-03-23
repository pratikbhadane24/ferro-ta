use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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
    let mut result = vec![f64::NAN; n];
    // Need at least timeperiod+1 bars to compute timeperiod return pairs
    #[allow(clippy::needless_range_loop)]
    for i in timeperiod..n {
        // returns from bar (i - timeperiod) to bar i  => timeperiod pairs
        let start = i - timeperiod;
        let mut rx = vec![0.0_f64; timeperiod];
        let mut ry = vec![0.0_f64; timeperiod];
        for k in 0..timeperiod {
            let prev = start + k;
            let curr = start + k + 1;
            rx[k] = if x[prev] != 0.0 {
                x[curr] / x[prev] - 1.0
            } else {
                f64::NAN
            };
            ry[k] = if y[prev] != 0.0 {
                y[curr] / y[prev] - 1.0
            } else {
                f64::NAN
            };
        }
        let mean_x: f64 = rx.iter().sum::<f64>() / timeperiod as f64;
        let mean_y: f64 = ry.iter().sum::<f64>() / timeperiod as f64;
        let cov: f64 = rx
            .iter()
            .zip(ry.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / timeperiod as f64;
        let var_x: f64 =
            rx.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>() / timeperiod as f64;
        result[i] = if var_x != 0.0 { cov / var_x } else { f64::NAN };
    }
    Ok(result.into_pyarray(py))
}
