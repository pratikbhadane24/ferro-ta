/*!
Fuzz target for `ferro_ta_core::overlap::macd`.

Verifies that MACD never panics and that all three output vectors
(macd, signal, histogram) match the input length.
*/

#![no_main]

use libfuzzer_sys::fuzz_target;
use ferro_ta_core::overlap;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    // Extract periods from first 3 bytes (1-64 range each)
    let fastperiod = ((data[0] as usize) % 32) + 1;
    let slowperiod = ((data[1] as usize) % 32) + fastperiod + 1; // slow > fast
    let signalperiod = ((data[2] as usize) % 32) + 1;

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

    let (macd, signal, hist) = overlap::macd(&close, fastperiod, slowperiod, signalperiod);

    assert_eq!(macd.len(), close.len(), "MACD line length mismatch");
    assert_eq!(signal.len(), close.len(), "MACD signal length mismatch");
    assert_eq!(hist.len(), close.len(), "MACD histogram length mismatch");
});
