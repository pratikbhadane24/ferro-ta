//! PyO3 wrappers for options analytics.

mod american;
mod chain;
mod digital;
mod greeks;
mod iv;
mod payoff;
mod pricing;
mod realized_vol;
mod surface;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn parse_option_kind(option_type: &str) -> PyResult<ferro_ta_core::options::OptionKind> {
    match option_type.to_ascii_lowercase().as_str() {
        "call" | "c" => Ok(ferro_ta_core::options::OptionKind::Call),
        "put" | "p" => Ok(ferro_ta_core::options::OptionKind::Put),
        _ => Err(PyValueError::new_err(format!(
            "option_type must be 'call' or 'put', got {option_type}"
        ))),
    }
}

pub(crate) fn parse_pricing_model(model: &str) -> PyResult<ferro_ta_core::options::PricingModel> {
    match model.to_ascii_lowercase().as_str() {
        "bsm" | "black_scholes" | "black-scholes" | "blackscholes" => {
            Ok(ferro_ta_core::options::PricingModel::BlackScholes)
        }
        "black76" | "black_76" | "black-76" => Ok(ferro_ta_core::options::PricingModel::Black76),
        _ => Err(PyValueError::new_err(format!(
            "model must be one of 'bsm'/'black_scholes' or 'black76', got {model}"
        ))),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::pricing::bsm_price, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::pricing::black76_price, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::pricing::bsm_price_batch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::pricing::black76_price_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::pricing::put_call_parity_deviation,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::greeks::option_greeks, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::greeks::option_greeks_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::greeks::extended_greeks, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::greeks::extended_greeks_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::iv::implied_volatility, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::iv::implied_volatility_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::iv::iv_rank, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::iv::iv_percentile, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::iv::iv_zscore, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::surface::smile_metrics, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::surface::term_structure_slope,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::surface::expected_move, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::chain::moneyness_labels, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::chain::select_strike_offset,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::chain::select_strike_delta, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::payoff::strategy_payoff_dense,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::payoff::strategy_payoff_legs,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::payoff::aggregate_greeks_dense,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::payoff::aggregate_greeks_legs,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::payoff::strategy_value_dense,
        m
    )?)?;
    // Digital options
    m.add_function(pyo3::wrap_pyfunction!(self::digital::digital_price, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::digital::digital_price_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::digital::digital_greeks, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::digital::digital_greeks_batch,
        m
    )?)?;
    // American options
    m.add_function(pyo3::wrap_pyfunction!(self::american::american_price, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::american::american_price_batch,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::american::early_exercise_premium,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::american::early_exercise_premium_batch,
        m
    )?)?;
    // Historical volatility estimators + vol cone
    m.add_function(pyo3::wrap_pyfunction!(
        self::realized_vol::close_to_close_vol,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::realized_vol::parkinson_vol,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::realized_vol::garman_klass_vol,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::realized_vol::rogers_satchell_vol,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::realized_vol::yang_zhang_vol,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::realized_vol::vol_cone, m)?)?;
    Ok(())
}
