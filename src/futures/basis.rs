use pyo3::prelude::*;

#[pyfunction]
pub fn futures_basis(spot: f64, future: f64) -> PyResult<f64> {
    Ok(ferro_ta_core::futures::basis::basis(spot, future))
}

#[pyfunction]
pub fn annualized_basis(spot: f64, future: f64, time_to_expiry: f64) -> PyResult<f64> {
    Ok(ferro_ta_core::futures::basis::annualized_basis(
        spot,
        future,
        time_to_expiry,
    ))
}

#[pyfunction]
pub fn implied_carry_rate(spot: f64, future: f64, time_to_expiry: f64) -> PyResult<f64> {
    Ok(ferro_ta_core::futures::basis::implied_carry_rate(
        spot,
        future,
        time_to_expiry,
    ))
}

#[pyfunction]
pub fn carry_spread(spot: f64, future: f64, rate: f64, time_to_expiry: f64) -> PyResult<f64> {
    Ok(ferro_ta_core::futures::basis::carry_spread(
        spot,
        future,
        rate,
        time_to_expiry,
    ))
}
