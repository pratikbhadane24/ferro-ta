use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

type GreekArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", model = "bsm", carry = 0.0))]
#[allow(clippy::too_many_arguments)]
pub fn option_greeks(
    underlying: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_type: &str,
    model: &str,
    carry: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let kind = super::parse_option_kind(option_type)?;
    let model = super::parse_pricing_model(model)?;
    let greeks =
        ferro_ta_core::options::greeks::model_greeks(ferro_ta_core::options::OptionEvaluation {
            contract: ferro_ta_core::options::OptionContract {
                model,
                underlying,
                strike,
                rate,
                carry,
                time_to_expiry,
                kind,
            },
            volatility,
        });
    Ok((
        greeks.delta,
        greeks.gamma,
        greeks.vega,
        greeks.theta,
        greeks.rho,
    ))
}

#[pyfunction]
#[pyo3(signature = (underlying, strike, rate, time_to_expiry, volatility, option_type = "call", model = "bsm", carry = None))]
#[allow(clippy::too_many_arguments)]
pub fn option_greeks_batch<'py>(
    py: Python<'py>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    time_to_expiry: PyReadonlyArray1<'py, f64>,
    volatility: PyReadonlyArray1<'py, f64>,
    option_type: &str,
    model: &str,
    carry: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<GreekArrays<'py>> {
    let kind = super::parse_option_kind(option_type)?;
    let model = super::parse_pricing_model(model)?;
    let underlying = underlying.as_slice()?;
    let strike = strike.as_slice()?;
    let rate = rate.as_slice()?;
    let time_to_expiry = time_to_expiry.as_slice()?;
    let volatility = volatility.as_slice()?;
    let carry_vec = match carry {
        Some(array) => array.as_slice()?.to_vec(),
        None => vec![0.0; underlying.len()],
    };
    validation::validate_equal_length(&[
        (underlying.len(), "underlying"),
        (strike.len(), "strike"),
        (rate.len(), "rate"),
        (time_to_expiry.len(), "time_to_expiry"),
        (volatility.len(), "volatility"),
        (carry_vec.len(), "carry"),
    ])?;

    let mut delta = Vec::with_capacity(underlying.len());
    let mut gamma = Vec::with_capacity(underlying.len());
    let mut vega = Vec::with_capacity(underlying.len());
    let mut theta = Vec::with_capacity(underlying.len());
    let mut rho = Vec::with_capacity(underlying.len());
    for (((((&u, &k), &r), &t), &vol), &c) in underlying
        .iter()
        .zip(strike.iter())
        .zip(rate.iter())
        .zip(time_to_expiry.iter())
        .zip(volatility.iter())
        .zip(carry_vec.iter())
    {
        let g = ferro_ta_core::options::greeks::model_greeks(
            ferro_ta_core::options::OptionEvaluation {
                contract: ferro_ta_core::options::OptionContract {
                    model,
                    underlying: u,
                    strike: k,
                    rate: r,
                    carry: c,
                    time_to_expiry: t,
                    kind,
                },
                volatility: vol,
            },
        );
        delta.push(g.delta);
        gamma.push(g.gamma);
        vega.push(g.vega);
        theta.push(g.theta);
        rho.push(g.rho);
    }

    Ok((
        delta.into_pyarray(py),
        gamma.into_pyarray(py),
        vega.into_pyarray(py),
        theta.into_pyarray(py),
        rho.into_pyarray(py),
    ))
}
