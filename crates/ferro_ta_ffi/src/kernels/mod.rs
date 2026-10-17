//! Kernel exports: arrays of different, multi-dimensional (row-major) shapes
//! in one call. See [`crate::spec::KernelSpec`].

pub mod backtest;
pub mod backtest_tools;
pub mod overfit;
pub mod sweep;

use crate::spec::KernelSpec;
use crate::status::Status;

/// Every module's `KERNEL_SPECS` table, in a stable order.
pub const MODULE_KERNEL_SPECS: &[&[KernelSpec]] = &[
    overfit::KERNEL_SPECS,
    sweep::KERNEL_SPECS,
    backtest::KERNEL_SPECS,
    backtest_tools::KERNEL_SPECS,
];

/// Core input/domain errors surface as `FT_ERR_INVALID_PARAM`.
pub(crate) fn invalid<E>(_: E) -> Status {
    Status::InvalidParam
}
