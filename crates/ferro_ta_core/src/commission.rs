//! Commission, tax, and fee model for Indian and global markets.
//!
//! All `_rate` fields are fractions (0.001 = 0.1%).
//! All per-unit fields (`flat_per_order`, `per_lot`) are in base currency units (e.g., INR).
//! The model is self-contained: pass `trade_value`, `num_lots`, `is_buy` to get total cost.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Advanced commission and tax model.
///
/// # Fields (all public for direct construction)
/// - **Brokerage**: `flat_per_order`, `rate_of_value`, `per_lot`, `max_brokerage`
/// - **STT**: `stt_rate`, `stt_on_buy`, `stt_on_sell`
/// - **Levies**: `exchange_charges_rate`, `regulatory_charges_rate`, `gst_rate`, `stamp_duty_rate`
/// - **Sizing**: `lot_size`
///
/// # Indian market notes
/// - STT (Securities Transaction Tax) is applied on turnover (buy/sell legs vary by segment).
/// - Exchange charges and regulatory body charges are on turnover.
/// - GST (18%) applies on brokerage + exchange charges + regulatory body charges (not STT/stamp).
/// - Stamp duty is on buy-side value only.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CommissionModel {
    // --- Brokerage ---------------------------------------------------------
    /// Fixed fee per order (e.g., ₹20 flat fee per order). 0.0 = none.
    pub flat_per_order: f64,
    /// Proportional brokerage as fraction of `trade_value` (e.g., 0.001 = 0.1%). 0.0 = none.
    pub rate_of_value: f64,
    /// Fixed fee per lot (e.g., ₹2 per lot). 0.0 = none.
    pub per_lot: f64,
    /// Brokerage cap in currency units. 0.0 = no cap.
    /// Effective brokerage = min(flat + rate × value + per_lot × lots, max_brokerage).
    pub max_brokerage: f64,
    /// Bid-ask spread model in basis points. Half-spread is paid on each leg (entry and exit),
    /// so total roundtrip cost = spread_bps in bps. 0.0 = no spread cost.
    pub spread_bps: f64,

    // --- Securities Transaction Tax (STT) ----------------------------------
    /// STT rate as fraction of trade value. 0.0 = no STT.
    pub stt_rate: f64,
    /// Apply STT on the buy leg.
    pub stt_on_buy: bool,
    /// Apply STT on the sell leg.
    pub stt_on_sell: bool,

    // --- Exchange & Regulatory Levies --------------------------------------
    /// Exchange transaction charges rate (fraction of trade value).
    pub exchange_charges_rate: f64,
    /// Regulatory body turnover charges rate (fraction of trade value). Typically ~0.000001.
    pub regulatory_charges_rate: f64,
    /// Indirect tax (GST) rate applied on (brokerage + exchange_charges + regulatory_charges).
    /// Typically 0.18 in India.
    pub gst_rate: f64,
    /// Stamp duty rate on buy side only (fraction of trade value).
    pub stamp_duty_rate: f64,

    // --- Instrument Sizing ------------------------------------------------
    /// Lot size for the instrument.
    /// Equities: 1.0. Index futures/options: contract lot size (e.g., 25, 50, 75).
    /// Used for per_lot cost: cost += per_lot × ceil(quantity / lot_size).
    pub lot_size: f64,

    // --- Short Selling ----------------------------------------------------
    /// Annualised short borrow rate as a fraction (e.g. 0.03 = 3% p.a.).
    /// Applied per bar to short positions. 0.0 = no borrow cost.
    pub short_borrow_rate_annual: f64,
}

impl Default for CommissionModel {
    fn default() -> Self {
        Self {
            flat_per_order: 0.0,
            rate_of_value: 0.0,
            per_lot: 0.0,
            max_brokerage: 0.0,
            spread_bps: 0.0,
            stt_rate: 0.0,
            stt_on_buy: false,
            stt_on_sell: false,
            exchange_charges_rate: 0.0,
            regulatory_charges_rate: 0.0,
            gst_rate: 0.0,
            stamp_duty_rate: 0.0,
            lot_size: 1.0,
            short_borrow_rate_annual: 0.0,
        }
    }
}

impl CommissionModel {
    // ------------------------------------------------------------------
    // Core computation
    // ------------------------------------------------------------------

    /// Compute total transaction cost in **absolute currency units**.
    ///
    /// # Parameters
    /// - `trade_value`: price × quantity in base currency
    /// - `num_lots`: number of lots transacted
    /// - `is_buy`: true for buy (entry) leg, false for sell (exit) leg
    pub fn total_cost(&self, trade_value: f64, num_lots: f64, is_buy: bool) -> f64 {
        // Brokerage (optionally capped)
        let raw_brokerage =
            self.flat_per_order + self.rate_of_value * trade_value + self.per_lot * num_lots;
        let brokerage = if self.max_brokerage > 0.0 {
            raw_brokerage.min(self.max_brokerage)
        } else {
            raw_brokerage
        };

        // STT
        let stt = if (is_buy && self.stt_on_buy) || (!is_buy && self.stt_on_sell) {
            self.stt_rate * trade_value
        } else {
            0.0
        };

        let exchange = self.exchange_charges_rate * trade_value;
        let regulatory = self.regulatory_charges_rate * trade_value;

        // GST on brokerage + exchange + regulatory (NOT on STT or stamp duty)
        let gst = self.gst_rate * (brokerage + exchange + regulatory);

        // Stamp duty only on buy side
        let stamp = if is_buy {
            self.stamp_duty_rate * trade_value
        } else {
            0.0
        };

        // Bid-ask spread: half-spread paid on each leg
        let spread_cost = self.spread_bps / 2.0 / 10_000.0 * trade_value;

        brokerage + stt + exchange + regulatory + gst + stamp + spread_cost
    }

    /// Borrow cost per bar for a short position.
    ///
    /// # Parameters
    /// - `trade_value`: abs(price × quantity)
    /// - `periods_per_year`: 252 for daily, 52 for weekly, etc.
    pub fn short_borrow_cost(&self, trade_value: f64, periods_per_year: f64) -> f64 {
        if self.short_borrow_rate_annual <= 0.0 || periods_per_year <= 0.0 {
            return 0.0;
        }
        self.short_borrow_rate_annual / periods_per_year * trade_value
    }

    /// Compute cost as a **fraction of `initial_capital`** for use in normalised equity loops.
    ///
    /// Returns 0.0 if `initial_capital` ≤ 0.
    pub fn cost_fraction(
        &self,
        trade_value: f64,
        num_lots: f64,
        is_buy: bool,
        initial_capital: f64,
    ) -> f64 {
        if initial_capital <= 0.0 {
            return 0.0;
        }
        self.total_cost(trade_value, num_lots, is_buy) / initial_capital
    }

    // ------------------------------------------------------------------
    // Built-in Presets
    // ------------------------------------------------------------------

    /// Zero commission — useful for clean research/comparison runs.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Indian equity **delivery** (long-term hold).
    ///
    /// Brokerage: 0.1% (capped at ₹20), STT 0.1% both sides,
    /// exchange charges, regulatory body charges, 18% GST, stamp duty.
    pub fn equity_delivery_india() -> Self {
        Self {
            flat_per_order: 0.0,
            rate_of_value: 0.001, // 0.1%
            per_lot: 0.0,
            max_brokerage: 20.0, // ₹20 cap
            spread_bps: 0.0,
            stt_rate: 0.001, // 0.1%
            stt_on_buy: true,
            stt_on_sell: true,
            exchange_charges_rate: 0.0000297,
            regulatory_charges_rate: 0.000001,
            gst_rate: 0.18,
            stamp_duty_rate: 0.00015,
            lot_size: 1.0,
            short_borrow_rate_annual: 0.0,
        }
    }

    /// Indian equity **intraday** (same-day square-off).
    ///
    /// Brokerage: 0.03% (capped at ₹20), STT 0.025% sell side only,
    /// exchange charges, regulatory body charges, 18% GST, stamp duty on buy.
    pub fn equity_intraday_india() -> Self {
        Self {
            flat_per_order: 0.0,
            rate_of_value: 0.0003, // 0.03%
            per_lot: 0.0,
            max_brokerage: 20.0,
            spread_bps: 0.0,
            stt_rate: 0.00025, // 0.025%
            stt_on_buy: false,
            stt_on_sell: true,
            exchange_charges_rate: 0.0000297,
            regulatory_charges_rate: 0.000001,
            gst_rate: 0.18,
            stamp_duty_rate: 0.000003,
            lot_size: 1.0,
            short_borrow_rate_annual: 0.0,
        }
    }

    /// Indian **index futures** (indicative rates per current regulations).
    ///
    /// Flat ₹20 per order, STT 0.05% sell side only, exchange charges,
    /// regulatory body charges, 18% GST, stamp duty on buy.
    /// `lot_size` defaults to 25 — update as needed for the specific contract.
    pub fn futures_india() -> Self {
        Self {
            flat_per_order: 20.0,
            rate_of_value: 0.0,
            per_lot: 0.0,
            max_brokerage: 0.0,
            spread_bps: 0.0,
            stt_rate: 0.0005, // 0.05%
            stt_on_buy: false,
            stt_on_sell: true,
            exchange_charges_rate: 0.0000019,
            regulatory_charges_rate: 0.000001,
            gst_rate: 0.18,
            stamp_duty_rate: 0.00002,
            lot_size: 25.0,
            short_borrow_rate_annual: 0.0,
        }
    }

    /// Indian **index options** (indicative rates per current regulations).
    ///
    /// Flat ₹20 per order, STT 0.15% on premium sell side only, exchange charges,
    /// regulatory body charges, 18% GST, stamp duty on buy.
    /// `lot_size` defaults to 25 — update as needed for the specific contract.
    pub fn options_india() -> Self {
        Self {
            flat_per_order: 20.0,
            rate_of_value: 0.0,
            per_lot: 0.0,
            max_brokerage: 0.0,
            spread_bps: 0.0,
            stt_rate: 0.0015, // 0.15% on premium
            stt_on_buy: false,
            stt_on_sell: true,
            exchange_charges_rate: 0.0000053,
            regulatory_charges_rate: 0.000001,
            gst_rate: 0.18,
            stamp_duty_rate: 0.000003,
            lot_size: 25.0,
            short_borrow_rate_annual: 0.0,
        }
    }

    /// Simple proportional model — e.g., `proportional(0.001)` = 0.1% both sides.
    ///
    /// No taxes, no levies — suitable for non-Indian markets or simplified modelling.
    pub fn proportional(rate: f64) -> Self {
        Self {
            rate_of_value: rate,
            ..Default::default()
        }
    }

    // ------------------------------------------------------------------
    // JSON serialization (requires "serde" feature)
    // ------------------------------------------------------------------

    /// Serialize to a pretty-printed JSON string.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from a JSON string.
    #[cfg(feature = "serde")]
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
