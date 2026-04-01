//! PyO3 wrapper around `ferro_ta_core::commission::CommissionModel`.
//!
//! Exposes all fields as Python properties, provides static preset constructors,
//! and supports JSON persistence (save/load).

use ferro_ta_core::commission::CommissionModel as CoreModel;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs;

/// Advanced commission and tax model for Indian and global markets.
///
/// All `_rate` fields are fractions (e.g. 0.001 = 0.1%).
/// Per-unit fields (`flat_per_order`, `per_lot`) are in base currency units (e.g. INR).
///
/// ## Example
/// ```python
/// from ferro_ta._ferro_ta import CommissionModel
///
/// # Use a built-in preset
/// m = CommissionModel.equity_delivery_india()
/// cost = m.total_cost(100_000.0, 1.0, True)
/// print(f"Buy cost: ₹{cost:.2f}")
///
/// # Save and reload
/// m.save("/tmp/my_commission.json")
/// m2 = CommissionModel.load("/tmp/my_commission.json")
/// ```
#[pyclass(module = "ferro_ta._ferro_ta", name = "CommissionModel")]
#[derive(Clone, Default)]
pub struct PyCommissionModel {
    pub(crate) inner: CoreModel,
}

#[pymethods]
impl PyCommissionModel {
    /// Create a zero-commission model (all fields = 0, lot_size = 1).
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    // ---- Brokerage fields -----------------------------------------------

    #[getter]
    pub fn flat_per_order(&self) -> f64 {
        self.inner.flat_per_order
    }
    #[setter]
    pub fn set_flat_per_order(&mut self, v: f64) {
        self.inner.flat_per_order = v;
    }

    #[getter]
    pub fn rate_of_value(&self) -> f64 {
        self.inner.rate_of_value
    }
    #[setter]
    pub fn set_rate_of_value(&mut self, v: f64) {
        self.inner.rate_of_value = v;
    }

    #[getter]
    pub fn per_lot(&self) -> f64 {
        self.inner.per_lot
    }
    #[setter]
    pub fn set_per_lot(&mut self, v: f64) {
        self.inner.per_lot = v;
    }

    #[getter]
    pub fn max_brokerage(&self) -> f64 {
        self.inner.max_brokerage
    }
    #[setter]
    pub fn set_max_brokerage(&mut self, v: f64) {
        self.inner.max_brokerage = v;
    }

    #[getter]
    pub fn spread_bps(&self) -> f64 {
        self.inner.spread_bps
    }
    #[setter]
    pub fn set_spread_bps(&mut self, v: f64) {
        self.inner.spread_bps = v;
    }

    // ---- STT fields -----------------------------------------------------

    #[getter]
    pub fn stt_rate(&self) -> f64 {
        self.inner.stt_rate
    }
    #[setter]
    pub fn set_stt_rate(&mut self, v: f64) {
        self.inner.stt_rate = v;
    }

    #[getter]
    pub fn stt_on_buy(&self) -> bool {
        self.inner.stt_on_buy
    }
    #[setter]
    pub fn set_stt_on_buy(&mut self, v: bool) {
        self.inner.stt_on_buy = v;
    }

    #[getter]
    pub fn stt_on_sell(&self) -> bool {
        self.inner.stt_on_sell
    }
    #[setter]
    pub fn set_stt_on_sell(&mut self, v: bool) {
        self.inner.stt_on_sell = v;
    }

    // ---- Exchange / regulatory fields -----------------------------------

    #[getter]
    pub fn exchange_charges_rate(&self) -> f64 {
        self.inner.exchange_charges_rate
    }
    #[setter]
    pub fn set_exchange_charges_rate(&mut self, v: f64) {
        self.inner.exchange_charges_rate = v;
    }

    #[getter]
    pub fn regulatory_charges_rate(&self) -> f64 {
        self.inner.regulatory_charges_rate
    }
    #[setter]
    pub fn set_regulatory_charges_rate(&mut self, v: f64) {
        self.inner.regulatory_charges_rate = v;
    }

    #[getter]
    pub fn gst_rate(&self) -> f64 {
        self.inner.gst_rate
    }
    #[setter]
    pub fn set_gst_rate(&mut self, v: f64) {
        self.inner.gst_rate = v;
    }

    #[getter]
    pub fn stamp_duty_rate(&self) -> f64 {
        self.inner.stamp_duty_rate
    }
    #[setter]
    pub fn set_stamp_duty_rate(&mut self, v: f64) {
        self.inner.stamp_duty_rate = v;
    }

    #[getter]
    pub fn lot_size(&self) -> f64 {
        self.inner.lot_size
    }
    #[setter]
    pub fn set_lot_size(&mut self, v: f64) {
        self.inner.lot_size = v;
    }

    #[getter]
    pub fn short_borrow_rate_annual(&self) -> f64 {
        self.inner.short_borrow_rate_annual
    }
    #[setter]
    pub fn set_short_borrow_rate_annual(&mut self, v: f64) {
        self.inner.short_borrow_rate_annual = v;
    }

    // ---- Compute --------------------------------------------------------

    /// Total transaction cost in absolute currency units.
    ///
    /// Args:
    ///     trade_value: price × quantity in base currency
    ///     num_lots: number of lots transacted
    ///     is_buy: True for buy (entry) leg, False for sell (exit) leg
    pub fn total_cost(&self, trade_value: f64, num_lots: f64, is_buy: bool) -> f64 {
        self.inner.total_cost(trade_value, num_lots, is_buy)
    }

    /// Cost as fraction of `initial_capital` (for normalised equity loops).
    ///
    /// Returns 0.0 if `initial_capital` ≤ 0.
    pub fn cost_fraction(
        &self,
        trade_value: f64,
        num_lots: f64,
        is_buy: bool,
        initial_capital: f64,
    ) -> f64 {
        self.inner
            .cost_fraction(trade_value, num_lots, is_buy, initial_capital)
    }

    // ---- Presets (static constructors) ----------------------------------

    /// Zero-commission model (all fields = 0).
    #[staticmethod]
    pub fn zero() -> Self {
        Self {
            inner: CoreModel::zero(),
        }
    }

    /// Indian equity delivery preset (0.1% brokerage capped ₹20, STT both sides, full levies).
    #[staticmethod]
    pub fn equity_delivery_india() -> Self {
        Self {
            inner: CoreModel::equity_delivery_india(),
        }
    }

    /// Indian equity intraday preset (0.03% brokerage capped ₹20, STT sell only, full levies).
    #[staticmethod]
    pub fn equity_intraday_india() -> Self {
        Self {
            inner: CoreModel::equity_intraday_india(),
        }
    }

    /// Indian index futures preset (₹20 flat, STT sell only, lot_size=25).
    #[staticmethod]
    pub fn futures_india() -> Self {
        Self {
            inner: CoreModel::futures_india(),
        }
    }

    /// Indian index options preset (₹20 flat, STT on premium sell side, lot_size=25).
    #[staticmethod]
    pub fn options_india() -> Self {
        Self {
            inner: CoreModel::options_india(),
        }
    }

    /// Simple proportional model — `rate` fraction applied both ways, no taxes.
    #[staticmethod]
    pub fn proportional(rate: f64) -> Self {
        Self {
            inner: CoreModel::proportional(rate),
        }
    }

    // ---- JSON persistence -----------------------------------------------

    /// Serialize this model to a JSON string.
    pub fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Deserialize a `CommissionModel` from a JSON string.
    #[staticmethod]
    pub fn from_json(s: &str) -> PyResult<Self> {
        CoreModel::from_json(s)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Save this model to a JSON file at `path`.
    pub fn save(&self, path: &str) -> PyResult<()> {
        let json = self.to_json()?;
        fs::write(path, json).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load a `CommissionModel` from a JSON file at `path`.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let s = fs::read_to_string(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Self::from_json(&s)
    }

    fn __repr__(&self) -> String {
        format!(
            "CommissionModel(flat={}, rate_pct={:.4}%, stt={:.4}%, lot_size={})",
            self.inner.flat_per_order,
            self.inner.rate_of_value * 100.0,
            self.inner.stt_rate * 100.0,
            self.inner.lot_size,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
