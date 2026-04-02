use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (spot, strike, rate, time_to_expiry, volatility, option_type = "call", dividend_yield = 0.0))]
pub fn bsm_price(
    spot: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_type: &str,
    dividend_yield: f64,
) -> PyResult<f64> {
    let kind = super::parse_option_kind(option_type)?;
    Ok(ferro_ta_core::options::pricing::black_scholes_price(
        spot,
        strike,
        rate,
        dividend_yield,
        time_to_expiry,
        volatility,
        kind,
    ))
}

#[pyfunction]
#[pyo3(signature = (forward, strike, rate, time_to_expiry, volatility, option_type = "call"))]
pub fn black76_price(
    forward: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_type: &str,
) -> PyResult<f64> {
    let kind = super::parse_option_kind(option_type)?;
    Ok(ferro_ta_core::options::pricing::black_76_price(
        forward,
        strike,
        rate,
        time_to_expiry,
        volatility,
        kind,
    ))
}

#[pyfunction]
#[pyo3(signature = (spot, strike, rate, time_to_expiry, volatility, dividend_yield, option_type = "call"))]
#[allow(clippy::too_many_arguments)]
pub fn bsm_price_batch<'py>(
    py: Python<'py>,
    spot: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    time_to_expiry: PyReadonlyArray1<'py, f64>,
    volatility: PyReadonlyArray1<'py, f64>,
    dividend_yield: PyReadonlyArray1<'py, f64>,
    option_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = super::parse_option_kind(option_type)?;
    let spot = spot.as_slice()?;
    let strike = strike.as_slice()?;
    let rate = rate.as_slice()?;
    let time_to_expiry = time_to_expiry.as_slice()?;
    let volatility = volatility.as_slice()?;
    let dividend_yield = dividend_yield.as_slice()?;
    validation::validate_equal_length(&[
        (spot.len(), "spot"),
        (strike.len(), "strike"),
        (rate.len(), "rate"),
        (time_to_expiry.len(), "time_to_expiry"),
        (volatility.len(), "volatility"),
        (dividend_yield.len(), "dividend_yield"),
    ])?;

    let out: Vec<f64> = spot
        .iter()
        .zip(strike.iter())
        .zip(rate.iter())
        .zip(time_to_expiry.iter())
        .zip(volatility.iter())
        .zip(dividend_yield.iter())
        .map(|(((((&s, &k), &r), &t), &vol), &q)| {
            ferro_ta_core::options::pricing::black_scholes_price(s, k, r, q, t, vol, kind)
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (call_price, put_price, spot, strike, rate, time_to_expiry, carry = 0.0))]
#[allow(clippy::too_many_arguments)]
pub fn put_call_parity_deviation(
    call_price: f64,
    put_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    carry: f64,
) -> PyResult<f64> {
    Ok(ferro_ta_core::options::pricing::put_call_parity_deviation(
        call_price,
        put_price,
        spot,
        strike,
        rate,
        carry,
        time_to_expiry,
    ))
}

#[pyfunction]
#[pyo3(signature = (forward, strike, rate, time_to_expiry, volatility, option_type = "call"))]
pub fn black76_price_batch<'py>(
    py: Python<'py>,
    forward: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    time_to_expiry: PyReadonlyArray1<'py, f64>,
    volatility: PyReadonlyArray1<'py, f64>,
    option_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = super::parse_option_kind(option_type)?;
    let forward = forward.as_slice()?;
    let strike = strike.as_slice()?;
    let rate = rate.as_slice()?;
    let time_to_expiry = time_to_expiry.as_slice()?;
    let volatility = volatility.as_slice()?;
    validation::validate_equal_length(&[
        (forward.len(), "forward"),
        (strike.len(), "strike"),
        (rate.len(), "rate"),
        (time_to_expiry.len(), "time_to_expiry"),
        (volatility.len(), "volatility"),
    ])?;

    let out: Vec<f64> = forward
        .iter()
        .zip(strike.iter())
        .zip(rate.iter())
        .zip(time_to_expiry.iter())
        .zip(volatility.iter())
        .map(|((((&f, &k), &r), &t), &vol)| {
            ferro_ta_core::options::pricing::black_76_price(f, k, r, t, vol, kind)
        })
        .collect();
    Ok(out.into_pyarray(py))
}
