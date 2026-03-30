/*!
Fuzz target for `ferro_ta_core::overlap::bbands`.

Verifies that BBANDS never panics and that the three output vectors
(upper, middle, lower) always have the same length as the input.
When finite, upper >= middle >= lower must hold.
*/

#![no_main]

use libfuzzer_sys::fuzz_target;
use ferro_ta_core::overlap;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let timeperiod = ((data[0] as usize) % 64) + 1;
    // Use second byte for deviation multipliers (1.0 - 4.0 range)
    let nbdevup = 1.0 + (data[1] as f64 / 255.0) * 3.0;
    let nbdevdn = 1.0 + (data[2] as f64 / 255.0) * 3.0;

    let float_bytes = &data[3..];
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

    let (upper, middle, lower) = overlap::bbands(&close, timeperiod, nbdevup, nbdevdn);

    assert_eq!(upper.len(), close.len(), "BBANDS upper length mismatch");
    assert_eq!(middle.len(), close.len(), "BBANDS middle length mismatch");
    assert_eq!(lower.len(), close.len(), "BBANDS lower length mismatch");

    // When all three are finite, upper >= middle >= lower
    for i in 0..close.len() {
        if upper[i].is_finite() && middle[i].is_finite() && lower[i].is_finite() {
            assert!(
                upper[i] >= middle[i],
                "BBANDS upper[{i}] ({}) < middle[{i}] ({})",
                upper[i],
                middle[i]
            );
            assert!(
                middle[i] >= lower[i],
                "BBANDS middle[{i}] ({}) < lower[{i}] ({})",
                middle[i],
                lower[i]
            );
        }
    }
});
