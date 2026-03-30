//! PyO3 wrapper around `ferro_ta_core::currency::Currency`.

use ferro_ta_core::currency::Currency as CoreCurrency;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Immutable currency descriptor with formatting support.
///
/// ## Example
/// ```python
/// from ferro_ta._ferro_ta import Currency
///
/// inr = Currency.INR()
/// print(inr.format(123456.78))  # ₹1,23,456.78
///
/// usd = Currency.from_code("USD")
/// print(usd.format(1234567.89))  # $1,234,567.89
/// ```
#[pyclass(name = "Currency", module = "ferro_ta._ferro_ta", frozen)]
#[derive(Clone)]
pub struct PyCurrency {
    pub(crate) inner: &'static CoreCurrency,
}

#[pymethods]
impl PyCurrency {
    /// Format *amount* according to this currency's style.
    pub fn format(&self, amount: f64) -> String {
        self.inner.format(amount)
    }

    #[getter]
    pub fn code(&self) -> &str {
        self.inner.code
    }

    #[getter]
    pub fn symbol(&self) -> &str {
        self.inner.symbol
    }

    #[getter]
    pub fn decimal_places(&self) -> u8 {
        self.inner.decimal_places
    }

    #[getter]
    pub fn lakh_grouping(&self) -> bool {
        self.inner.lakh_grouping
    }

    // ---- Static constructors (presets) ----

    #[staticmethod]
    pub fn from_code(code: &str) -> PyResult<Self> {
        CoreCurrency::from_code(code)
            .map(|c| PyCurrency { inner: c })
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Unknown currency code '{code}'. Supported: INR, USD, EUR, GBP, JPY, USDT"
                ))
            })
    }

    /// Indian Rupee.
    #[staticmethod]
    #[allow(non_snake_case)]
    pub fn INR() -> Self {
        PyCurrency {
            inner: &CoreCurrency::INR,
        }
    }

    /// US Dollar.
    #[staticmethod]
    #[allow(non_snake_case)]
    pub fn USD() -> Self {
        PyCurrency {
            inner: &CoreCurrency::USD,
        }
    }

    /// Euro.
    #[staticmethod]
    #[allow(non_snake_case)]
    pub fn EUR() -> Self {
        PyCurrency {
            inner: &CoreCurrency::EUR,
        }
    }

    /// British Pound.
    #[staticmethod]
    #[allow(non_snake_case)]
    pub fn GBP() -> Self {
        PyCurrency {
            inner: &CoreCurrency::GBP,
        }
    }

    /// Japanese Yen.
    #[staticmethod]
    #[allow(non_snake_case)]
    pub fn JPY() -> Self {
        PyCurrency {
            inner: &CoreCurrency::JPY,
        }
    }

    /// Tether USD.
    #[staticmethod]
    #[allow(non_snake_case)]
    pub fn USDT() -> Self {
        PyCurrency {
            inner: &CoreCurrency::USDT,
        }
    }

    fn __repr__(&self) -> String {
        format!("Currency({:?})", self.inner.code)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner.code == other.inner.code
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.code.hash(&mut hasher);
        hasher.finish()
    }
}
