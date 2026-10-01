//! Status codes returned by every `ft_*` function.
//!
//! Errors are reported per call through the return value rather than a
//! thread-local "last error": Go goroutines (and .NET async continuations)
//! migrate between OS threads across foreign calls, so a thread-local could be
//! read from a different thread than the one that set it.

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub const FT_OK: i32 = 0;
/// A required pointer was NULL while `len > 0` (or a handle was NULL).
pub const FT_ERR_NULL_PTR: i32 = 1;
/// A scalar parameter is out of range (period below minimum, bad matype, ...).
pub const FT_ERR_INVALID_PARAM: i32 = 2;
/// Input arrays differ in length. The C ABI takes one shared `len`, so this is
/// only produced by language wrappers that accept separately sized arrays.
pub const FT_ERR_LENGTH_MISMATCH: i32 = 3;
/// The Rust core panicked. The panic was contained; outputs are unspecified.
pub const FT_ERR_PANIC: i32 = 4;

/// Internal error type; converted to an `i32` status at the ABI boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    NullPtr,
    InvalidParam,
}

impl Status {
    const fn code(self) -> i32 {
        match self {
            Status::NullPtr => FT_ERR_NULL_PTR,
            Status::InvalidParam => FT_ERR_INVALID_PARAM,
        }
    }
}

/// Run `body` with panics contained, mapping the outcome to a status code.
/// Unwinding across an `extern "C"` boundary aborts the host process, so every
/// exported function goes through here.
pub fn guard(body: impl FnOnce() -> Result<(), Status>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => FT_OK,
        Ok(Err(status)) => status.code(),
        Err(_) => FT_ERR_PANIC,
    }
}

/// Human-readable description of a status code. Never NULL; the string is
/// static and must not be freed.
#[no_mangle]
pub extern "C" fn ft_status_message(code: i32) -> *const c_char {
    let msg: &'static [u8] = match code {
        FT_OK => b"ok\0",
        FT_ERR_NULL_PTR => b"null pointer argument\0",
        FT_ERR_INVALID_PARAM => b"invalid parameter\0",
        FT_ERR_LENGTH_MISMATCH => b"input arrays must have the same length\0",
        FT_ERR_PANIC => b"internal error (panic in ferro_ta_core)\0",
        _ => b"unknown status code\0",
    };
    msg.as_ptr().cast()
}

/// Library version, e.g. `"1.3.1"`. Static; must not be freed.
#[no_mangle]
pub extern "C" fn ft_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}
