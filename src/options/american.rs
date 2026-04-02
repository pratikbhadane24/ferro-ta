use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use ferro_ta_core::options::american::{
    american_price_baw as core_american_price,
    early_exercise_premium as core_early_exercise_premium,
};

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", carry = 0.0))]
#[allow(clippy::too_many_arguments)]
pub fn american_price(
    underlying: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_type: &str,
    carry: f64,
) -> PyResult<f64> {
    let kind = super::parse_option_kind(option_type)?;
    Ok(core_american_price(
        underlying,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        kind,
    ))
}

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", carry = None))]
#[allow(clippy::too_many_arguments)]
pub fn american_price_batch<'py>(
    py: Python<'py>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    time_to_expiry: PyReadonlyArray1<'py, f64>,
    volatility: PyReadonlyArray1<'py, f64>,
    option_type: &str,
    carry: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = super::parse_option_kind(option_type)?;
    let underlying = underlying.as_slice()?;
    let strike = strike.as_slice()?;
    let rate = rate.as_slice()?;
    let tte = time_to_expiry.as_slice()?;
    let vol = volatility.as_slice()?;
    let n = underlying.len();
    let carry_vec = match carry {
        Some(arr) => arr.as_slice()?.to_vec(),
        None => vec![0.0; n],
    };
    let out: Vec<f64> = underlying
        .iter()
        .zip(strike.iter())
        .zip(rate.iter())
        .zip(tte.iter())
        .zip(vol.iter())
        .zip(carry_vec.iter())
        .map(|(((((&u, &k), &r), &t), &v), &c)| core_american_price(u, k, r, c, t, v, kind))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", carry = 0.0))]
#[allow(clippy::too_many_arguments)]
pub fn early_exercise_premium(
    underlying: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_type: &str,
    carry: f64,
) -> PyResult<f64> {
    let kind = super::parse_option_kind(option_type)?;
    Ok(core_early_exercise_premium(
        underlying,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        kind,
    ))
}

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", carry = None))]
#[allow(clippy::too_many_arguments)]
pub fn early_exercise_premium_batch<'py>(
    py: Python<'py>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    time_to_expiry: PyReadonlyArray1<'py, f64>,
    volatility: PyReadonlyArray1<'py, f64>,
    option_type: &str,
    carry: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = super::parse_option_kind(option_type)?;
    let underlying = underlying.as_slice()?;
    let strike = strike.as_slice()?;
    let rate = rate.as_slice()?;
    let tte = time_to_expiry.as_slice()?;
    let vol = volatility.as_slice()?;
    let n = underlying.len();
    let carry_vec = match carry {
        Some(arr) => arr.as_slice()?.to_vec(),
        None => vec![0.0; n],
    };
    let out: Vec<f64> = underlying
        .iter()
        .zip(strike.iter())
        .zip(rate.iter())
        .zip(tte.iter())
        .zip(vol.iter())
        .zip(carry_vec.iter())
        .map(|(((((&u, &k), &r), &t), &v), &c)| core_early_exercise_premium(u, k, r, c, t, v, kind))
        .collect();
    Ok(out.into_pyarray(py))
}
