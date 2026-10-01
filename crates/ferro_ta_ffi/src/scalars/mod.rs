//! Scalar exports (option pricing, futures analytics, position sizing).

pub mod futures;
pub mod options;
pub mod sizing;

use crate::spec::ScalarSpec;

/// Every module's `SCALAR_SPECS` table, in a stable order.
pub const MODULE_SCALAR_SPECS: &[&[ScalarSpec]] = &[
    options::SCALAR_SPECS,
    futures::SCALAR_SPECS,
    sizing::SCALAR_SPECS,
];
