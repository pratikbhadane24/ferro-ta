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
    if data.len() < 4 {
        return;
    }

    let timeperiod = ((data[0] as usize) % 64) + 1;
    // Use second byte for deviation multipliers (1.0 - 4.0 range)
    let nbdevup = 1.0 + (data[1] as f64 / 255.0) * 3.0;
    let nbdevdn = 1.0 + (data[2] as f64 / 255.0) * 3.0;
    // `% 12` so out-of-range values (9..=11) are reachable and exercise the
    // documented all-NaN contract, not just the valid 0..=8 range.
    let matype = data[3] % 12;

    let float_bytes = &data[4..];
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

    let (upper, middle, lower) = overlap::bbands(&close, timeperiod, nbdevup, nbdevdn, matype);

    assert_eq!(upper.len(), close.len(), "BBANDS upper length mismatch");
    assert_eq!(middle.len(), close.len(), "BBANDS middle length mismatch");
    assert_eq!(lower.len(), close.len(), "BBANDS lower length mismatch");

    // `upper >= middle >= lower` holds for **every** `matype`, and the assertion
    // is deliberately not gated on it.
    //
    // TA-Lib does compute the half-width about the SMA rather than about the
    // selected MA (`ta_BBANDS.c` calls `TA_STDDEV` without passing
    // `optInMAType`), and ferro-ta matches that — but both outer bands are
    // offsets from the *same* centre (`write_bands`: `centre +/- nbdev * std`),
    // and `std` does not depend on `matype`. So the SMA asymmetry changes what
    // `nbdev` *means* relative to the price distribution, never the ordering.
    // Gating this on `matype == 0` would silently retire the invariant for
    // eight of the nine valid values.
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
