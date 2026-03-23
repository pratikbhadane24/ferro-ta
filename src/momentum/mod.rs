//! Momentum indicators — RSI, stochastics, ADX, CCI, etc.
//! Each indicator (or small group) lives in its own file for maintainability.

mod adx;
mod apo;
mod aroon;
mod bop;
mod cci;
mod cmo;
mod mfi;
mod mom;
mod ppo;
mod roc;
mod rsi;
mod stoch;
mod stochf;
mod stochrsi;
mod trix;
mod ultosc;
mod willr;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::rsi::rsi, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::mom::mom, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::roc::roc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::roc::rocp, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::roc::rocr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::roc::rocr100, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::willr::willr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::aroon::aroon, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::aroon::aroonosc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cci::cci, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::mfi::mfi, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::bop::bop, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::stochf::stochf, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::stoch::stoch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::stochrsi::stochrsi, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::apo::apo, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::ppo::ppo, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cmo::cmo, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::adx::plus_dm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::adx::minus_dm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::adx::plus_di, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::adx::minus_di, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::adx::dx, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::adx::adx, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::adx::adxr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::trix::trix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::ultosc::ultosc, m)?)?;
    Ok(())
}
