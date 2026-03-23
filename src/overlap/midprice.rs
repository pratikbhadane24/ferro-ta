use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use ta::indicators::{Maximum, Minimum};
use ta::Next;

/// MidPrice: (highest high + lowest low) / 2 over the rolling window.
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 14))]
pub fn midprice<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[(n, "high"), (lows.len(), "low")])?;
    let mut max_ind = Maximum::new(timeperiod)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut min_ind = Minimum::new(timeperiod)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut result = vec![f64::NAN; n];
    for (i, (&h, &l)) in highs.iter().zip(lows.iter()).enumerate() {
        let mx = max_ind.next(h);
        let mn = min_ind.next(l);
        if i + 1 >= timeperiod {
            result[i] = (mx + mn) / 2.0;
        }
    }
    Ok(result.into_pyarray(py))
}
