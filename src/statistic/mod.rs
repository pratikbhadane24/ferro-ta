//! Statistic functions — rolling window statistical operations on price data.
//! Each function (or closely related group) lives in its own file.

mod beta;
pub(crate) mod common;
mod correl;
mod dtw;
mod linearreg;
mod stddev;
mod var;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::stddev::stddev, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::var::var, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::linearreg::linearreg, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::linearreg::linearreg_slope, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::linearreg::linearreg_intercept,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::linearreg::linearreg_angle, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::linearreg::tsf, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::beta::beta, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::correl::correl, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::dtw::dtw, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::dtw::dtw_distance, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::dtw::batch_dtw, m)?)?;
    Ok(())
}
