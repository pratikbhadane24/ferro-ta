use crate::validation;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (strikes, vols, reference_price, time_to_expiry, model = "bsm", rate = 0.0, carry = 0.0))]
pub fn smile_metrics<'py>(
    strikes: PyReadonlyArray1<'py, f64>,
    vols: PyReadonlyArray1<'py, f64>,
    reference_price: f64,
    time_to_expiry: f64,
    model: &str,
    rate: f64,
    carry: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let strikes = strikes.as_slice()?;
    let vols = vols.as_slice()?;
    validation::validate_equal_length(&[(strikes.len(), "strikes"), (vols.len(), "vols")])?;
    let model = super::parse_pricing_model(model)?;
    let metrics = ferro_ta_core::options::surface::smile_metrics(
        strikes,
        vols,
        reference_price,
        rate,
        carry,
        time_to_expiry,
        model,
    );
    Ok((
        metrics.atm_iv,
        metrics.risk_reversal_25d,
        metrics.butterfly_25d,
        metrics.skew_slope,
        metrics.convexity,
    ))
}

#[pyfunction]
pub fn term_structure_slope<'py>(
    tenors: PyReadonlyArray1<'py, f64>,
    atm_ivs: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let tenors = tenors.as_slice()?;
    let atm_ivs = atm_ivs.as_slice()?;
    validation::validate_equal_length(&[(tenors.len(), "tenors"), (atm_ivs.len(), "atm_ivs")])?;
    Ok(ferro_ta_core::options::surface::term_structure_slope(
        tenors, atm_ivs,
    ))
}

#[pyfunction]
#[pyo3(signature = (spot, iv, days_to_expiry, trading_days_per_year = 252.0))]
pub fn expected_move(
    spot: f64,
    iv: f64,
    days_to_expiry: f64,
    trading_days_per_year: f64,
) -> PyResult<(f64, f64)> {
    Ok(ferro_ta_core::options::surface::expected_move(
        spot,
        iv,
        days_to_expiry,
        trading_days_per_year,
    ))
}
