use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Tillson T3 (triple smoothed EMA). Converges after ~6*(timeperiod-1) bars.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, vfactor = 0.7))]
pub fn t3<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    vfactor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();

    let mut e = [0.0_f64; 6];
    let k = 2.0 / (timeperiod as f64 + 1.0);

    let v = vfactor;
    let c1 = -(v * v * v);
    let c2 = 3.0 * v * v + 3.0 * v * v * v;
    let c3 = -6.0 * v * v - 3.0 * v - 3.0 * v * v * v;
    let c4 = 1.0 + 3.0 * v + v * v * v + 3.0 * v * v;

    let warmup = 6 * (timeperiod - 1);
    let mut result = vec![f64::NAN; n];

    for (i, &price) in prices.iter().enumerate() {
        if i == 0 {
            for ej in e.iter_mut() {
                *ej = price;
            }
        } else {
            e[0] += k * (price - e[0]);
            for j in 1..6 {
                e[j] += k * (e[j - 1] - e[j]);
            }
        }
        if i >= warmup {
            result[i] = c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2];
        }
    }
    Ok(result.into_pyarray(py))
}
