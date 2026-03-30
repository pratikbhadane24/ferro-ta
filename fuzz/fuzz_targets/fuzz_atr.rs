/*!
Fuzz target for `ferro_ta_core::volatility::atr`.

Verifies that ATR never panics, output length matches input, and all
finite values are non-negative (ATR is always >= 0).
*/

#![no_main]

use libfuzzer_sys::fuzz_target;
use ferro_ta_core::volatility;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let timeperiod = ((data[0] as usize) % 64) + 1;

    // Need 3 f64s per bar (high, low, close)
    let float_bytes = &data[1..];
    let n_floats = float_bytes.len() / 8;
    let n_bars = n_floats / 3;
    if n_bars == 0 {
        return;
    }

    let all_floats: Vec<f64> = (0..n_bars * 3)
        .map(|i| {
            let chunk: [u8; 8] = float_bytes[i * 8..(i + 1) * 8].try_into().unwrap();
            f64::from_le_bytes(chunk)
        })
        .collect();

    let high = &all_floats[..n_bars];
    let low = &all_floats[n_bars..n_bars * 2];
    let close = &all_floats[n_bars * 2..n_bars * 3];

    let result = volatility::atr(high, low, close, timeperiod);
    assert_eq!(result.len(), high.len(), "ATR output length mismatch");

    // ATR values should be non-negative when finite
    for (i, &v) in result.iter().enumerate() {
        if v.is_finite() {
            assert!(v >= 0.0, "ATR result[{i}] = {v} is negative");
        }
    }
});
