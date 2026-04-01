/*!
Fuzz target for `ferro_ta_core::momentum::stoch`.

Verifies that STOCH never panics, output lengths match, and finite
values lie in [0, 100].
*/

#![no_main]

use libfuzzer_sys::fuzz_target;
use ferro_ta_core::momentum;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let fastk_period = ((data[0] as usize) % 32) + 1;
    let slowk_period = ((data[1] as usize) % 16) + 1;
    let slowd_period = ((data[2] as usize) % 16) + 1;

    // Need 3 f64s per bar (high, low, close)
    let float_bytes = &data[3..];
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

    let (slowk, slowd) = momentum::stoch(high, low, close, fastk_period, slowk_period, slowd_period);

    assert_eq!(slowk.len(), high.len(), "STOCH slowk length mismatch");
    assert_eq!(slowd.len(), high.len(), "STOCH slowd length mismatch");

    // Finite values should be in [0, 100]
    for (i, &v) in slowk.iter().enumerate() {
        if v.is_finite() {
            assert!(
                v >= 0.0 && v <= 100.0,
                "STOCH slowk[{i}] = {v} is out of [0, 100]"
            );
        }
    }
    for (i, &v) in slowd.iter().enumerate() {
        if v.is_finite() {
            assert!(
                v >= 0.0 && v <= 100.0,
                "STOCH slowd[{i}] = {v} is out of [0, 100]"
            );
        }
    }
});
