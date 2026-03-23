//! PyO3 wrappers for futures analytics.

mod basis;
mod curve;
mod roll;
mod synthetic;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(
        self::synthetic::synthetic_forward,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::synthetic::synthetic_spot, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::synthetic::parity_gap, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::basis::futures_basis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::basis::annualized_basis, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::basis::implied_carry_rate, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::basis::carry_spread, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::roll::weighted_continuous_contract,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::roll::back_adjusted_continuous_contract,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::roll::ratio_adjusted_continuous_contract,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::roll::roll_yield, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::curve::calendar_spreads, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::curve::curve_slope, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::curve::curve_summary, m)?)?;
    Ok(())
}
