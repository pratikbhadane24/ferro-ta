//! Volatility indicators — measure the magnitude of price fluctuations.
//! Each indicator lives in its own file for maintainability.

mod atr;
mod natr;
mod trange;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::trange::trange, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::atr::atr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::natr::natr, m)?)?;
    Ok(())
}
