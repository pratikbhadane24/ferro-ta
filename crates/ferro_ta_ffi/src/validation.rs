//! Scalar parameter validation shared by every export.
//!
//! Integer rules (period minimums, non-negative counts, matype `0..=8`) mirror
//! the Python binding's `src/validation.rs` and live in `marshal::Param`.

use crate::status::Status;

/// Validate a `double` parameter (multipliers, deviations, acceleration
/// factors, percentages) before it reaches the core.
///
/// `NaN` and `±Inf` are rejected: no float parameter in the core has a
/// meaningful non-finite value, and passing one through would yield an
/// all-NaN result indistinguishable from warm-up. Sign/range rules are
/// parameter-specific and left to the core.
pub fn check_float(raw: f64) -> Result<f64, Status> {
    if !raw.is_finite() {
        return Err(Status::InvalidParam);
    }
    Ok(raw)
}
