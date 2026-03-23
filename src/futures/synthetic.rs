use pyo3::prelude::*;

#[pyfunction]
pub fn synthetic_forward(
    call_price: f64,
    put_price: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
) -> PyResult<f64> {
    Ok(ferro_ta_core::futures::synthetic::synthetic_forward(
        call_price,
        put_price,
        strike,
        rate,
        time_to_expiry,
    ))
}

#[pyfunction]
#[pyo3(signature = (call_price, put_price, strike, rate, time_to_expiry, carry = 0.0))]
pub fn synthetic_spot(
    call_price: f64,
    put_price: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    carry: f64,
) -> PyResult<f64> {
    Ok(ferro_ta_core::futures::synthetic::synthetic_spot(
        call_price,
        put_price,
        strike,
        rate,
        carry,
        time_to_expiry,
    ))
}

#[pyfunction]
#[pyo3(signature = (call_price, put_price, spot, strike, rate, time_to_expiry, carry = 0.0))]
pub fn parity_gap(
    call_price: f64,
    put_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    carry: f64,
) -> PyResult<f64> {
    Ok(ferro_ta_core::futures::synthetic::parity_gap(
        call_price,
        put_price,
        spot,
        strike,
        rate,
        carry,
        time_to_expiry,
    ))
}
