//! Bar builder over the C ABI.

use ferro_ta_ffi::*;
use std::ptr;

const IST_OFFSET_SECS: i64 = 19_800;

/// UTC epoch ms for an IST wall-clock time on a fixed day.
fn ist(h: i64, m: i64) -> i64 {
    20_377 * 86_400_000 + (h * 60 + m) * 60_000 - IST_OFFSET_SECS * 1000
}

fn nse_5min() -> *mut ferro_ta_core::streaming::BarBuilder {
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_bar_builder_new(300, 33_300, 55_800, IST_OFFSET_SECS, &mut h) },
        FT_OK
    );
    h
}

#[test]
fn ticks_close_bars_on_boundary_and_time() {
    let h = nse_5min();
    let (mut status, mut closed, mut bar) = (-1, -1, FtBar::default());
    let tick = |ts, p, s: &mut i32, c: &mut i32, b: &mut FtBar| unsafe {
        ft_bar_builder_on_tick(h, ts, p, 1.0, s, c, b)
    };
    assert_eq!(
        tick(ist(9, 16), 100.0, &mut status, &mut closed, &mut bar),
        FT_OK
    );
    assert_eq!((status, closed), (0, 0));
    assert_eq!(
        tick(ist(9, 18), 104.0, &mut status, &mut closed, &mut bar),
        FT_OK
    );
    assert_eq!(
        tick(ist(9, 20), 101.0, &mut status, &mut closed, &mut bar),
        FT_OK
    );
    assert_eq!((status, closed), (0, 1));
    assert_eq!((bar.start_ms, bar.end_ms), (ist(9, 15), ist(9, 20)));
    assert_eq!(
        (bar.open, bar.high, bar.close, bar.ticks),
        (100.0, 104.0, 104.0, 2)
    );

    assert_eq!(
        tick(ist(9, 10), 99.0, &mut status, &mut closed, &mut bar),
        FT_OK
    );
    assert_eq!(status, 3, "out of session");

    let mut has = -1;
    assert_eq!(
        unsafe { ft_bar_builder_forming(h, &mut has, &mut bar) },
        FT_OK
    );
    assert_eq!((has, bar.start_ms), (1, ist(9, 20)));
    assert_eq!(
        unsafe { ft_bar_builder_on_time(h, ist(9, 25), &mut closed, &mut bar) },
        FT_OK
    );
    assert_eq!((closed, bar.end_ms), (1, ist(9, 25)));
    assert_eq!(
        unsafe { ft_bar_builder_forming(h, &mut has, &mut bar) },
        FT_OK
    );
    assert_eq!((has, bar), (0, FtBar::default()));
    unsafe { ft_bar_builder_free(h) };
}

#[test]
fn rejects_bad_spec_and_null_pointers() {
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_bar_builder_new(0, 33_300, 55_800, 0, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert!(h.is_null());
    assert_eq!(
        unsafe { ft_bar_builder_new(300, 0, 100, 0, ptr::null_mut()) },
        FT_ERR_NULL_PTR
    );
    let h = nse_5min();
    let mut bar = FtBar::default();
    assert_eq!(
        unsafe {
            ft_bar_builder_on_tick(h, 0, 1.0, 1.0, ptr::null_mut(), ptr::null_mut(), &mut bar)
        },
        FT_ERR_NULL_PTR
    );
    assert_eq!(
        unsafe { ft_bar_builder_reset(ptr::null_mut()) },
        FT_ERR_NULL_PTR
    );
    unsafe { ft_bar_builder_free(h) };
    unsafe { ft_bar_builder_free(ptr::null_mut()) };
}

#[test]
fn bar_builder_save_load_keeps_forming_bar() {
    let h = nse_5min();
    let (mut st, mut closed, mut bar) = (0, 0, FtBar::default());
    unsafe { ft_bar_builder_on_tick(h, ist(9, 16), 100.0, 1.0, &mut st, &mut closed, &mut bar) };
    let mut need = 0usize;
    assert_eq!(
        unsafe { ft_bar_builder_save(h, ptr::null_mut(), 0, &mut need) },
        FT_ERR_LENGTH_MISMATCH
    );
    let mut buf = vec![0u8; need];
    assert_eq!(
        unsafe { ft_bar_builder_save(h, buf.as_mut_ptr().cast(), need, &mut need) },
        FT_OK
    );
    let mut restored = ptr::null_mut();
    assert_eq!(
        unsafe { ft_bar_builder_load(buf.as_ptr().cast(), need, &mut restored) },
        FT_OK
    );
    let (mut has, mut forming) = (0, FtBar::default());
    unsafe { ft_bar_builder_forming(restored, &mut has, &mut forming) };
    assert_eq!(
        (has, forming.start_ms, forming.open),
        (1, ist(9, 15), 100.0)
    );
    unsafe { ft_bar_builder_free(h) };
    unsafe { ft_bar_builder_free(restored) };
}
