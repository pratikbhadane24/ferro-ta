//! Price transformations — helper functions to synthesize OHLC arrays into single price arrays.
//! Each transform lives in its own file for maintainability.

mod avgprice;
mod medprice;
mod typprice;
mod wclprice;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::avgprice::avgprice, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::medprice::medprice, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::typprice::typprice, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::wclprice::wclprice, m)?)?;
    Ok(())
}
