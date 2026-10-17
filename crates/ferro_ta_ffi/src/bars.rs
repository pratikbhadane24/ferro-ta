//! Hand-written C ABI for the session-aligned tick → bar builder.
//!
//! Timestamped input and an optional struct output don't fit
//! `ffi_streams!`, so these exports (and their header block in
//! `scripts/ffi_codegen/c_header.py`) are written by hand. Handle rules match
//! the streaming handles: not thread-safe, release once with `_free`.

use crate::status::{guard, Status, FT_ERR_NULL_PTR};
use ferro_ta_core::streaming::{Bar, BarBuilder, BarSpec};

/// C mirror of `ferro_ta_core::streaming::Bar`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FtBar {
    pub start_ms: i64,
    pub end_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub ticks: u32,
}

impl From<Bar> for FtBar {
    fn from(b: Bar) -> Self {
        Self {
            start_ms: b.start_ms,
            end_ms: b.end_ms,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            volume: b.volume,
            ticks: b.ticks,
        }
    }
}

/// Write `bar` (or a zeroed bar) and its presence flag.
///
/// # Safety
/// Both pointers must be non-null and valid for one write.
unsafe fn write_opt_bar(bar: Option<Bar>, out_has: *mut i32, out_bar: *mut FtBar) {
    // SAFETY: caller contract.
    unsafe {
        out_has.write(i32::from(bar.is_some()));
        out_bar.write(bar.map(FtBar::from).unwrap_or_default());
    }
}

/// Create a bar builder. Session times are seconds after local midnight;
/// local time = UTC + `utc_offset_secs`.
/// Requires: 0 < interval_secs <= 86400; 0 <= session_open_secs <
/// session_close_secs <= 86400; |utc_offset_secs| <= 50400.
///
/// # Safety
/// `out_handle` must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ft_bar_builder_new(
    interval_secs: i64,
    session_open_secs: i64,
    session_close_secs: i64,
    utc_offset_secs: i64,
    out_handle: *mut *mut BarBuilder,
) -> i32 {
    if out_handle.is_null() {
        return FT_ERR_NULL_PTR;
    }
    // SAFETY: checked non-null; valid for one write per contract.
    unsafe { out_handle.write(std::ptr::null_mut()) };
    guard(|| {
        let spec = BarSpec {
            interval_secs,
            session_open_secs,
            session_close_secs,
            utc_offset_secs,
        };
        let builder = BarBuilder::new(spec).map_err(|_| Status::InvalidParam)?;
        // SAFETY: as above.
        unsafe { out_handle.write(Box::into_raw(Box::new(builder))) };
        Ok(())
    })
}

/// Feed one tick. `*out_status` is an `FT_TICK_*` code; when the tick closed
/// a bar, `*out_closed` is 1 and `*out_bar` holds it (else 0 / zeroed).
///
/// # Safety
/// `handle` must come from `ft_bar_builder_new` and not be freed; each
/// output pointer must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ft_bar_builder_on_tick(
    handle: *mut BarBuilder,
    ts_ms: i64,
    price: f64,
    qty: f64,
    out_status: *mut i32,
    out_closed: *mut i32,
    out_bar: *mut FtBar,
) -> i32 {
    guard(|| {
        if handle.is_null() || out_status.is_null() || out_closed.is_null() || out_bar.is_null() {
            return Err(Status::NullPtr);
        }
        // SAFETY: live, exclusively-used handle per contract.
        let (status, closed) = unsafe { &mut *handle }.on_tick(ts_ms, price, qty);
        // SAFETY: outputs checked non-null above.
        unsafe {
            out_status.write(status as i32);
            write_opt_bar(closed, out_closed, out_bar);
        }
        Ok(())
    })
}

/// Close the forming bar once `now_ms` reaches its end (no tick needed).
/// `*out_closed` is 1 and `*out_bar` holds it when a bar closed.
///
/// # Safety
/// As `ft_bar_builder_on_tick`.
#[no_mangle]
pub unsafe extern "C" fn ft_bar_builder_on_time(
    handle: *mut BarBuilder,
    now_ms: i64,
    out_closed: *mut i32,
    out_bar: *mut FtBar,
) -> i32 {
    guard(|| {
        if handle.is_null() || out_closed.is_null() || out_bar.is_null() {
            return Err(Status::NullPtr);
        }
        // SAFETY: live, exclusively-used handle per contract.
        let closed = unsafe { &mut *handle }.on_time(now_ms);
        // SAFETY: outputs checked non-null above.
        unsafe { write_opt_bar(closed, out_closed, out_bar) };
        Ok(())
    })
}

/// The bar being built: `*out_has` is 1 and `*out_bar` holds it, else 0.
///
/// # Safety
/// As `ft_bar_builder_on_tick`.
#[no_mangle]
pub unsafe extern "C" fn ft_bar_builder_forming(
    handle: *mut BarBuilder,
    out_has: *mut i32,
    out_bar: *mut FtBar,
) -> i32 {
    guard(|| {
        if handle.is_null() || out_has.is_null() || out_bar.is_null() {
            return Err(Status::NullPtr);
        }
        // SAFETY: live handle per contract; read-only use.
        let forming = unsafe { &*handle }.forming();
        // SAFETY: outputs checked non-null above.
        unsafe { write_opt_bar(forming, out_has, out_bar) };
        Ok(())
    })
}

/// Drop the forming bar and tick history.
///
/// # Safety
/// `handle` must come from `ft_bar_builder_new` and not be freed.
#[no_mangle]
pub unsafe extern "C" fn ft_bar_builder_reset(handle: *mut BarBuilder) -> i32 {
    guard(|| {
        if handle.is_null() {
            return Err(Status::NullPtr);
        }
        // SAFETY: live, exclusively-used handle per contract.
        unsafe { &mut *handle }.reset();
        Ok(())
    })
}

/// Write a versioned JSON snapshot (not NUL-terminated); same contract as
/// the streaming `*_save` exports.
///
/// # Safety
/// `handle` live; `buf` valid for `cap` bytes when `cap > 0`; `out_len`
/// valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ft_bar_builder_save(
    handle: *mut BarBuilder,
    buf: *mut std::ffi::c_char,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if handle.is_null() || out_len.is_null() {
            return Err(Status::NullPtr);
        }
        // SAFETY: live handle per contract; read-only use.
        let json = unsafe { &*handle }.save().map_err(|_| Status::Internal)?;
        // SAFETY: checked non-null; caller contract for `buf`.
        unsafe { crate::streaming::write_snapshot(&json, buf, cap, out_len) }
    })
}

/// Create a bar builder from a snapshot made by `ft_bar_builder_save`.
///
/// # Safety
/// `json` valid for `len` bytes; `out_handle` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ft_bar_builder_load(
    json: *const std::ffi::c_char,
    len: usize,
    out_handle: *mut *mut BarBuilder,
) -> i32 {
    if out_handle.is_null() {
        return FT_ERR_NULL_PTR;
    }
    // SAFETY: checked non-null; valid for one write per contract.
    unsafe { out_handle.write(std::ptr::null_mut()) };
    guard(|| {
        // SAFETY: valid for `len` bytes per contract.
        let text = unsafe { crate::streaming::read_snapshot(json, len) }?;
        let builder = BarBuilder::load(text).map_err(|_| Status::InvalidParam)?;
        // SAFETY: as above.
        unsafe { out_handle.write(Box::into_raw(Box::new(builder))) };
        Ok(())
    })
}

/// Release a bar builder. NULL is a no-op.
///
/// # Safety
/// `handle` must come from `ft_bar_builder_new` and must not be used (or
/// freed) again afterwards.
#[no_mangle]
pub unsafe extern "C" fn ft_bar_builder_free(handle: *mut BarBuilder) {
    if !handle.is_null() {
        // SAFETY: produced by Box::into_raw in `_new`, freed once.
        drop(unsafe { Box::from_raw(handle) });
    }
}
