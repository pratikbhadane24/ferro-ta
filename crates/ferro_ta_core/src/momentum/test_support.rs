//! Shared oracles and assertions for the `momentum` submodule tests.
//!
//! Equivalence oracles for the O(n) rewrites.
//!
//! The audit found this module had range and edge assertions but *no golden
//! vectors* for `aroon`, `cci` or `ultosc`, so each rewritten kernel keeps a
//! verbatim copy of its pre-change implementation here and is asserted
//! against it: `to_bits()` where the change is bit-identical, an explicit
//! tolerance with a stated reason where it is not.

/// Adversarial-but-finite series. The low-cardinality ones are the point:
/// they force constant ties, which is the only thing that can silently go
/// wrong in the Aroon rewrite.
pub(super) fn oracle_series() -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
    let ties: Vec<f64> = (0..200).map(|i| (i % 5) as f64).collect();
    let coarse: Vec<f64> = (0..200).map(|i| (i % 3) as f64).collect();
    let up: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let down: Vec<f64> = (0..200).map(|i| -(i as f64)).collect();
    let flat = vec![7.5_f64; 200];
    let mut state = 987_654_321_u64;
    let walk: Vec<f64> = (0..400)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            100.0 + ((state >> 33) % 11) as f64
        })
        .collect();
    let walk_lo: Vec<f64> = walk.iter().map(|x| x - 1.0).collect();
    vec![
        ("empty", vec![], vec![]),
        ("single", vec![42.0], vec![41.0]),
        ("ties", ties.clone(), coarse.clone()),
        ("ties_swapped", coarse, ties),
        (
            "monotone_up",
            up.clone(),
            up.iter().map(|x| x - 1.0).collect(),
        ),
        (
            "monotone_down",
            down.clone(),
            down.iter().map(|x| x - 1.0).collect(),
        ),
        ("all_equal", flat.clone(), flat),
        ("walk", walk, walk_lo),
    ]
}

pub(super) fn oracle_periods(n: usize) -> Vec<usize> {
    let mut p = vec![0, 1, 2, 3, 5, 14, 28];
    if n > 0 {
        p.push(n - 1);
        p.push(n);
        p.push(n + 1);
    }
    p
}

pub(super) fn assert_bits(got: &[f64], want: &[f64], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{ctx}: index {i}: {g} vs {w}");
    }
}

pub(super) fn ultosc_ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state = 24_680_u64;
    let mut close = Vec::with_capacity(n);
    let mut price = 100.0_f64;
    for _ in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        price += (((state >> 33) % 21) as f64 - 10.0) * 0.05;
        close.push(price);
    }
    let high: Vec<f64> = close.iter().map(|c| c + 0.4).collect();
    let low: Vec<f64> = close.iter().map(|c| c - 0.4).collect();
    (high, low, close)
}
