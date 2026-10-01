//! Scalar parameter validation shared by every export.
//!
//! Integer rules (period minimums, non-negative counts, matype `0..=8`) mirror
//! the Python binding's `src/validation.rs` and live in `marshal::Param`.

use crate::status::Status;

/// Validate a `double` parameter (multipliers, deviations, acceleration
/// factors, percentages) before it reaches the core.
pub fn check_float(raw: f64) -> Result<f64, Status> {
    // TODO(user): decide how non-finite float params are handled.
    Ok(raw)
}
