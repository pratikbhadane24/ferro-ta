use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (price, underlying, strike, rate, time_to_expiry, option_type = "call", model = "bsm", carry = 0.0, initial_guess = 0.2, tolerance = 1e-8, max_iterations = 100))]
#[allow(clippy::too_many_arguments)]
pub fn implied_volatility(
    price: f64,
    underlying: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    option_type: &str,
    model: &str,
    carry: f64,
    initial_guess: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<f64> {
    let kind = super::parse_option_kind(option_type)?;
    let model = super::parse_pricing_model(model)?;
    Ok(ferro_ta_core::options::iv::implied_volatility(
        ferro_ta_core::options::OptionContract {
            model,
            underlying,
            strike,
            rate,
            carry,
            time_to_expiry,
            kind,
        },
        price,
        ferro_ta_core::options::IvSolverConfig {
            initial_guess,
            tolerance,
            max_iterations,
        },
    ))
}

#[pyfunction]
#[pyo3(signature = (price, underlying, strike, rate, time_to_expiry, option_type = "call", model = "bsm", carry = None, initial_guess = None, tolerance = 1e-8, max_iterations = 100))]
#[allow(clippy::too_many_arguments)]
pub fn implied_volatility_batch<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    underlying: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    time_to_expiry: PyReadonlyArray1<'py, f64>,
    option_type: &str,
    model: &str,
    carry: Option<PyReadonlyArray1<'py, f64>>,
    initial_guess: Option<PyReadonlyArray1<'py, f64>>,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = super::parse_option_kind(option_type)?;
    let model = super::parse_pricing_model(model)?;
    let price = price.as_slice()?;
    let underlying = underlying.as_slice()?;
    let strike = strike.as_slice()?;
    let rate = rate.as_slice()?;
    let time_to_expiry = time_to_expiry.as_slice()?;
    let carry_vec = match carry {
        Some(array) => array.as_slice()?.to_vec(),
        None => vec![0.0; price.len()],
    };
    let guess_vec = match initial_guess {
        Some(array) => array.as_slice()?.to_vec(),
        None => vec![0.2; price.len()],
    };
    validation::validate_equal_length(&[
        (price.len(), "price"),
        (underlying.len(), "underlying"),
        (strike.len(), "strike"),
        (rate.len(), "rate"),
        (time_to_expiry.len(), "time_to_expiry"),
        (carry_vec.len(), "carry"),
        (guess_vec.len(), "initial_guess"),
    ])?;

    let out: Vec<f64> = price
        .iter()
        .zip(underlying.iter())
        .zip(strike.iter())
        .zip(rate.iter())
        .zip(time_to_expiry.iter())
        .zip(carry_vec.iter())
        .zip(guess_vec.iter())
        .map(|((((((&p, &u), &k), &r), &t), &c), &guess)| {
            ferro_ta_core::options::iv::implied_volatility(
                ferro_ta_core::options::OptionContract {
                    model,
                    underlying: u,
                    strike: k,
                    rate: r,
                    carry: c,
                    time_to_expiry: t,
                    kind,
                },
                p,
                ferro_ta_core::options::IvSolverConfig {
                    initial_guess: guess,
                    tolerance,
                    max_iterations,
                },
            )
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (iv_series, window = 252))]
pub fn iv_rank<'py>(
    py: Python<'py>,
    iv_series: PyReadonlyArray1<'py, f64>,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let window = validation::parse_timeperiod(window, "window", 1)?;
    let out = ferro_ta_core::options::iv::iv_rank(iv_series.as_slice()?, window);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (iv_series, window = 252))]
pub fn iv_percentile<'py>(
    py: Python<'py>,
    iv_series: PyReadonlyArray1<'py, f64>,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let window = validation::parse_timeperiod(window, "window", 1)?;
    let out = ferro_ta_core::options::iv::iv_percentile(iv_series.as_slice()?, window);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (iv_series, window = 252))]
pub fn iv_zscore<'py>(
    py: Python<'py>,
    iv_series: PyReadonlyArray1<'py, f64>,
    window: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let window = validation::parse_timeperiod(window, "window", 1)?;
    let out = ferro_ta_core::options::iv::iv_zscore(iv_series.as_slice()?, window);
    Ok(out.into_pyarray(py))
}
