use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Chande Momentum Oscillator: 100 * (sum of gains - sum of losses) / (sum of gains + sum of losses) over window.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14))]
pub fn cmo<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if n < timeperiod + 1 {
        return Ok(result.into_pyarray(py));
    }

    let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

    #[allow(clippy::needless_range_loop)]
    for i in timeperiod..n {
        let mut ups = 0.0_f64;
        let mut downs = 0.0_f64;
        for ch in &changes[(i - timeperiod)..i] {
            if *ch > 0.0 {
                ups += ch;
            } else {
                downs -= ch;
            }
        }
        let denom = ups + downs;
        result[i] = if denom != 0.0 {
            100.0 * (ups - downs) / denom
        } else {
            0.0
        };
    }
    Ok(result.into_pyarray(py))
}
