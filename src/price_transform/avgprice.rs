use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Average Price: (open + high + low + close) / 4.
#[pyfunction]
pub fn avgprice<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let opens = open.as_slice()?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = opens.len();
    validation::validate_equal_length(&[
        (n, "open"),
        (highs.len(), "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;
    let result: Vec<f64> = opens
        .iter()
        .zip(highs.iter())
        .zip(lows.iter())
        .zip(closes.iter())
        .map(|(((&o, &h), &l), &c)| (o + h + l + c) / 4.0)
        .collect();
    Ok(result.into_pyarray(py))
}
