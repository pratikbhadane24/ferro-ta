use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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
    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let wx = &x[(i + 1 - timeperiod)..=i];
        let wy = &y[(i + 1 - timeperiod)..=i];
        let mean_x: f64 = wx.iter().sum::<f64>() / timeperiod as f64;
        let mean_y: f64 = wy.iter().sum::<f64>() / timeperiod as f64;
        let cov: f64 = wx
            .iter()
            .zip(wy.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>();
        let std_x: f64 = (wx.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>()).sqrt();
        let std_y: f64 = (wy.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>()).sqrt();
        let denom = std_x * std_y;
        result[i] = if denom != 0.0 { cov / denom } else { f64::NAN };
    }
    Ok(result.into_pyarray(py))
}
