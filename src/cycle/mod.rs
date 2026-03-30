//! Cycle indicators — Hilbert Transform-based cycle analysis (Ehlers).
//! The shared HT core computation lives in `common.rs`; each indicator has its own file.
//!
//! All functions use a 63-bar lookback period (first 63 values are NaN).

mod ht_dcperiod;
mod ht_dcphase;
mod ht_phasor;
mod ht_sine;
mod ht_trendline;
mod ht_trendmode;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::ht_trendline::ht_trendline, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::ht_dcperiod::ht_dcperiod, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::ht_dcphase::ht_dcphase, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::ht_phasor::ht_phasor, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::ht_sine::ht_sine, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::ht_trendmode::ht_trendmode, m)?)?;
    Ok(())
}
