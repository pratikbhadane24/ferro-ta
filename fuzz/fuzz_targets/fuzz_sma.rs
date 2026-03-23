/*!
Fuzz target for `ferro_ta_core::overlap::sma`.

The fuzzer generates arbitrary byte sequences and interprets them as
`f64` values plus a `timeperiod`.  The invariant under test is that the
function **never panics** for any input — it may return `NaN`, `Inf`, or
an all-NaN slice, but it must not crash.
*/

#![no_main]

use libfuzzer_sys::fuzz_target;
use ferro_ta_core::overlap;

fuzz_target!(|data: &[u8]| {
    // Need at least 1 byte for timeperiod + 8 bytes for one f64
    if data.len() < 2 {
        return;
    }

    // Extract timeperiod from first byte (1-64 to keep runs fast)
    let timeperiod = ((data[0] as usize) % 64) + 1;

    // Interpret remaining bytes as f64 values (skip incomplete trailing bytes)
    let float_bytes = &data[1..];
    let n_floats = float_bytes.len() / 8;
    if n_floats == 0 {
        return;
    }

    let close: Vec<f64> = (0..n_floats)
        .map(|i| {
            let chunk: [u8; 8] = float_bytes[i * 8..(i + 1) * 8].try_into().unwrap();
            f64::from_le_bytes(chunk)
        })
        .collect();

    // Must not panic for any input
    let result = overlap::sma(&close, timeperiod);

    // Result length must match input length
    assert_eq!(result.len(), close.len(), "SMA output length mismatch");

    // The first (timeperiod - 1) values must be NaN
    for i in 0..(timeperiod.min(close.len()).saturating_sub(1)) {
        assert!(
            result[i].is_nan(),
            "SMA result[{i}] should be NaN (warm-up period)"
        );
    }
});
