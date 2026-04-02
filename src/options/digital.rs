use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ferro_ta_core::options::digital::{
    digital_greeks as core_digital_greeks, digital_price as core_digital_price, DigitalKind,
};

fn parse_digital_kind(s: &str) -> PyResult<DigitalKind> {
    match s.to_ascii_lowercase().replace('-', "_").as_str() {
        "cash_or_nothing" | "cash" => Ok(DigitalKind::CashOrNothing),
        "asset_or_nothing" | "asset" => Ok(DigitalKind::AssetOrNothing),
        _ => Err(PyValueError::new_err(
            "digital_type must be 'cash_or_nothing' or 'asset_or_nothing'",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", digital_type = "cash_or_nothing", carry = 0.0))]
#[allow(clippy::too_many_arguments)]
pub fn digital_price(
    underlying: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_type: &str,
    digital_type: &str,
    carry: f64,
) -> PyResult<f64> {
    let kind = super::parse_option_kind(option_type)?;
    let dkind = parse_digital_kind(digital_type)?;
    Ok(core_digital_price(
        underlying,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        kind,
        dkind,
    ))
}

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", digital_type = "cash_or_nothing", carry = None))]
#[allow(clippy::too_many_arguments)]
pub fn digital_price_batch<'py>(
    py: Python<'py>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    time_to_expiry: PyReadonlyArray1<'py, f64>,
    volatility: PyReadonlyArray1<'py, f64>,
    option_type: &str,
    digital_type: &str,
    carry: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = super::parse_option_kind(option_type)?;
    let dkind = parse_digital_kind(digital_type)?;
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
        .map(|(((((&u, &k), &r), &t), &v), &c)| core_digital_price(u, k, r, c, t, v, kind, dkind))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", digital_type = "cash_or_nothing", carry = 0.0))]
#[allow(clippy::too_many_arguments)]
pub fn digital_greeks(
    underlying: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_type: &str,
    digital_type: &str,
    carry: f64,
) -> PyResult<(f64, f64, f64)> {
    let kind = super::parse_option_kind(option_type)?;
    let dkind = parse_digital_kind(digital_type)?;
    Ok(core_digital_greeks(
        underlying,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        kind,
        dkind,
    ))
}

type GreekTriple<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", digital_type = "cash_or_nothing", carry = None))]
#[allow(clippy::too_many_arguments)]
pub fn digital_greeks_batch<'py>(
    py: Python<'py>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    time_to_expiry: PyReadonlyArray1<'py, f64>,
    volatility: PyReadonlyArray1<'py, f64>,
    option_type: &str,
    digital_type: &str,
    carry: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<GreekTriple<'py>> {
    let kind = super::parse_option_kind(option_type)?;
    let dkind = parse_digital_kind(digital_type)?;
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
    let mut delta = Vec::with_capacity(n);
    let mut gamma = Vec::with_capacity(n);
    let mut vega = Vec::with_capacity(n);
    for (((((&u, &k), &r), &t), &v), &c) in underlying
        .iter()
        .zip(strike.iter())
        .zip(rate.iter())
        .zip(tte.iter())
        .zip(vol.iter())
        .zip(carry_vec.iter())
    {
        let (d, g, ve) = core_digital_greeks(u, k, r, c, t, v, kind, dkind);
        delta.push(d);
        gamma.push(g);
        vega.push(ve);
    }
    Ok((
        delta.into_pyarray(py),
        gamma.into_pyarray(py),
        vega.into_pyarray(py),
    ))
}
