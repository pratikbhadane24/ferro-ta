use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (strikes, reference_price, option_type = "call"))]
pub fn moneyness_labels<'py>(
    py: Python<'py>,
    strikes: PyReadonlyArray1<'py, f64>,
    reference_price: f64,
    option_type: &str,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let kind = super::parse_option_kind(option_type)?;
    let strikes = strikes.as_slice()?;
    let labels = ferro_ta_core::options::chain::label_moneyness(strikes, reference_price, kind);
    Ok(labels.into_pyarray(py))
}

#[pyfunction]
pub fn select_strike_offset<'py>(
    strikes: PyReadonlyArray1<'py, f64>,
    reference_price: f64,
    offset: isize,
) -> PyResult<Option<f64>> {
    Ok(ferro_ta_core::options::chain::select_strike_by_offset(
        strikes.as_slice()?,
        reference_price,
        offset,
    ))
}

#[pyfunction]
#[pyo3(signature = (strikes, vols, reference_price, time_to_expiry, target_delta, option_type = "call", model = "bsm", rate = 0.0, carry = 0.0))]
#[allow(clippy::too_many_arguments)]
pub fn select_strike_delta<'py>(
    strikes: PyReadonlyArray1<'py, f64>,
    vols: PyReadonlyArray1<'py, f64>,
    reference_price: f64,
    time_to_expiry: f64,
    target_delta: f64,
    option_type: &str,
    model: &str,
    rate: f64,
    carry: f64,
) -> PyResult<Option<f64>> {
    let kind = super::parse_option_kind(option_type)?;
    let model = super::parse_pricing_model(model)?;
    let strikes = strikes.as_slice()?;
    let vols = vols.as_slice()?;
    validation::validate_equal_length(&[(strikes.len(), "strikes"), (vols.len(), "vols")])?;
    Ok(ferro_ta_core::options::chain::select_strike_by_delta(
        strikes,
        vols,
        ferro_ta_core::options::ChainGreeksContext {
            model,
            reference_price,
            rate,
            carry,
            time_to_expiry,
            kind,
        },
        target_delta,
    ))
}
