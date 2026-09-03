/*!
Fuzz target for `ferro_ta_core::momentum::stoch`.

Verifies that STOCH never panics, output lengths match, and finite
values lie in [0, 100].

Both matypes are drawn from the input as well. Values above the valid range
are deliberately reachable: the core must return an all-`NaN` pair for them
rather than falling back to SMA, and `ma_lookback`'s saturating arithmetic
must keep an absurd period from overflowing in a debug build.
*/

#![no_main]

use libfuzzer_sys::fuzz_target;
use ferro_ta_core::momentum;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let fastk_period = ((data[0] as usize) % 32) + 1;
    let slowk_period = ((data[1] as usize) % 16) + 1;
    let slowd_period = ((data[2] as usize) % 16) + 1;
    // `% 12` leaves the out-of-range values 9..=11 reachable so the
    // reject-rather-than-fall-back-to-SMA path is exercised too.
    let slowk_matype = data[3] % 12;
    let slowd_matype = data[4] % 12;

    // Need 3 f64s per bar (high, low, close)
    let float_bytes = &data[5..];
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

    let (slowk, slowd) = momentum::stoch(
        high,
        low,
        close,
        fastk_period,
        slowk_period,
        slowk_matype,
        slowd_period,
        slowd_matype,
    );

    assert_eq!(slowk.len(), high.len(), "STOCH slowk length mismatch");
    assert_eq!(slowd.len(), high.len(), "STOCH slowd length mismatch");

    // The [0, 100] bound is a property of *well-formed* bars only: %K is
    // `(close - low) / (high - low) * 100`, so a bar whose close sits outside
    // its own high/low range (which random bytes produce freely, including via
    // denormals) legitimately lands outside it. Gate the bound on the input
    // actually being an OHLC series; the never-panic and length invariants
    // above hold unconditionally and are the point of this target.
    let well_formed = (0..n_bars).all(|i| {
        let (h, l, c) = (high[i], low[i], close[i]);
        h.is_finite() && l.is_finite() && c.is_finite() && l <= c && c <= h
    });
    if !well_formed {
        return;
    }

    // Only the *convex* MA types keep a smoothed series inside its input's
    // envelope: SMA/EMA/WMA/TRIMA/KAMA are weighted means with non-negative
    // weights summing to 1. DEMA (`3`), TEMA (`4`) and T3 (`7`, `8`) are
    // polynomial combinations with negative coefficients, so they overshoot a
    // rising %K past 100 by design — not a bug, so not asserted. Even a convex
    // type can land a rounding step outside the range (a constant-100 series
    // smoothed by EMA can read 100.00000000000011), hence the epsilon.
    const EPS: f64 = 1e-9;
    let is_convex = |m: u8| matches!(m, 0 | 1 | 2 | 5 | 6);
    let check = |name: &str, values: &[f64]| {
        for (i, &v) in values.iter().enumerate() {
            if v.is_finite() {
                assert!(
                    (-EPS..=100.0 + EPS).contains(&v),
                    "STOCH {name}[{i}] = {v} is out of [0, 100]"
                );
            }
        }
    };

    if is_convex(slowk_matype) {
        check("slowk", &slowk);
    }
    // Slow %D is `MA(slow %K, slowd_period, slowd_matype)`, so it inherits
    // slow %K's envelope only when *both* legs are convex.
    if is_convex(slowk_matype) && is_convex(slowd_matype) {
        check("slowd", &slowd);
    }
});
