/*!
Fuzz target for `ferro_ta_core::momentum::rsi`.

Generates arbitrary f64 slices (via raw bytes) and arbitrary timeperiods,
verifying that RSI never panics and that all finite output values lie in
the range [0, 100].
*/

#![no_main]

use libfuzzer_sys::fuzz_target;
use ferro_ta_core::momentum;

fuzz_target!(|data: &[u8]| {
    // Need at least 1 byte for timeperiod + 8 bytes for one f64
    if data.len() < 2 {
        return;
    }

    // Extract timeperiod from first byte (1-64)
    let timeperiod = ((data[0] as usize) % 64) + 1;

    // Interpret remaining bytes as f64 values
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

    // Must not panic
    let result = momentum::rsi(&close, timeperiod);

    // Result length must match input
    assert_eq!(result.len(), close.len(), "RSI output length mismatch");

    // All finite output values must be in [0, 100]
    for (i, &v) in result.iter().enumerate() {
        if v.is_finite() {
            assert!(
                v >= 0.0 && v <= 100.0,
                "RSI result[{i}] = {v} is out of [0, 100]"
            );
        }
    }
});
