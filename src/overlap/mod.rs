//! Overlap studies — moving averages and trend indicators.
//! Each indicator lives in its own file for maintainability.

mod bbands;
mod dema;
mod ema;
mod kama;
mod ma_mavp;
mod macd;
mod macdext;
mod mama;
mod midpoint;
mod midprice;
mod sar;
mod sarext;
mod sma;
mod t3;
mod tema;
mod trima;
mod wma;

pub use ma_mavp::{ma, mavp};

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::sma::sma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::ema::ema, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::wma::wma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::dema::dema, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::tema::tema, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::trima::trima, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::kama::kama, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::t3::t3, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::bbands::bbands, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::macd::macd, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::macd::macdfix, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::sar::sar, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::midpoint::midpoint, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::midprice::midprice, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(mavp, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::mama::mama, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::sarext::sarext, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::macdext::macdext, m)?)?;
    Ok(())
}
