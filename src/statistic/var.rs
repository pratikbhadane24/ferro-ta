use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Rolling variance; scaled by nbdev².
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, nbdev = 1.0))]
pub fn var<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    nbdev: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let window = &prices[(i + 1 - timeperiod)..=i];
        let mean: f64 = window.iter().sum::<f64>() / timeperiod as f64;
        let variance: f64 =
            window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / timeperiod as f64;
        result[i] = variance * nbdev * nbdev;
    }
    Ok(result.into_pyarray(py))
}
