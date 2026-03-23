use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
pub fn calendar_spreads<'py>(
    py: Python<'py>,
    futures_prices: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(
        ferro_ta_core::futures::curve::calendar_spreads(futures_prices.as_slice()?)
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn curve_slope<'py>(
    tenors: PyReadonlyArray1<'py, f64>,
    futures_prices: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let tenors = tenors.as_slice()?;
    let futures_prices = futures_prices.as_slice()?;
    validation::validate_equal_length(&[
        (tenors.len(), "tenors"),
        (futures_prices.len(), "futures_prices"),
    ])?;
    Ok(ferro_ta_core::futures::curve::curve_slope(
        tenors,
        futures_prices,
    ))
}

#[pyfunction]
pub fn curve_summary<'py>(
    spot: f64,
    tenors: PyReadonlyArray1<'py, f64>,
    futures_prices: PyReadonlyArray1<'py, f64>,
) -> PyResult<(f64, f64, f64, bool)> {
    let tenors = tenors.as_slice()?;
    let futures_prices = futures_prices.as_slice()?;
    validation::validate_equal_length(&[
        (tenors.len(), "tenors"),
        (futures_prices.len(), "futures_prices"),
    ])?;
    let summary = ferro_ta_core::futures::curve::curve_summary(spot, tenors, futures_prices);
    Ok((
        summary.front_basis,
        summary.average_basis,
        summary.slope,
        summary.is_contango,
    ))
}
