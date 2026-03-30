/*!
Fuzz target for `ferro_ta_core::overlap::wma`.

Verifies that WMA never panics and that the output length always
matches the input length.
*/

#![no_main]

use libfuzzer_sys::fuzz_target;
use ferro_ta_core::overlap;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let timeperiod = ((data[0] as usize) % 64) + 1;

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

    let result = overlap::wma(&close, timeperiod);
    assert_eq!(result.len(), close.len(), "WMA output length mismatch");
});
