//! Volume indicators — require volume data to measure buying and selling pressure.
//! Each indicator lives in its own file for maintainability.

mod ad;
mod adosc;
mod obv;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::ad::ad, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::adosc::adosc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::obv::obv, m)?)?;
    Ok(())
}
