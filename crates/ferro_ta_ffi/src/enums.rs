//! C enums: validated `int32_t` values mapped onto core Rust enums.
//!
//! A raw integer is never transmuted into a Rust enum (an out-of-range value
//! would be undefined behaviour); each conversion goes through
//! [`Param`](crate::marshal::Param) and rejects unknown values with
//! `FT_ERR_INVALID_PARAM`.

use crate::marshal::Param;
use crate::spec::{EnumSpec, ParamKind};
use crate::status::Status;
use ferro_ta_core::options::digital::DigitalKind;
use ferro_ta_core::options::{OptionKind, PricingModel};

pub const OPTION_KIND: EnumSpec = EnumSpec {
    name: "OptionKind",
    c_prefix: "FT_OPTION",
    values: &[("CALL", 0), ("PUT", 1)],
};

pub const DIGITAL_KIND: EnumSpec = EnumSpec {
    name: "DigitalKind",
    c_prefix: "FT_DIGITAL",
    values: &[("CASH_OR_NOTHING", 0), ("ASSET_OR_NOTHING", 1)],
};

pub const PRICING_MODEL: EnumSpec = EnumSpec {
    name: "PricingModel",
    c_prefix: "FT_MODEL",
    values: &[("BLACK_SCHOLES", 0), ("BLACK_76", 1)],
};

/// Every C enum, for the spec dump.
pub const ALL_ENUMS: &[&EnumSpec] = &[&OPTION_KIND, &DIGITAL_KIND, &PRICING_MODEL];

impl Param<i32> for OptionKind {
    fn from_raw(raw: i32, _kind: ParamKind) -> Result<Self, Status> {
        match raw {
            0 => Ok(OptionKind::Call),
            1 => Ok(OptionKind::Put),
            _ => Err(Status::InvalidParam),
        }
    }
}

impl Param<i32> for DigitalKind {
    fn from_raw(raw: i32, _kind: ParamKind) -> Result<Self, Status> {
        match raw {
            0 => Ok(DigitalKind::CashOrNothing),
            1 => Ok(DigitalKind::AssetOrNothing),
            _ => Err(Status::InvalidParam),
        }
    }
}

impl Param<i32> for PricingModel {
    fn from_raw(raw: i32, _kind: ParamKind) -> Result<Self, Status> {
        match raw {
            0 => Ok(PricingModel::BlackScholes),
            1 => Ok(PricingModel::Black76),
            _ => Err(Status::InvalidParam),
        }
    }
}
