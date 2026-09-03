//! Shared fixtures, bitwise assertions and equivalence oracles for the
//! extended-catalog tests.
//!
//! # Equivalence against the pre-change implementations
//!
//! `reference_*` below are verbatim copies of the two-pass forms that the
//! fused single-pass kernels replaced. They call `math::sliding_max` /
//! `math::sliding_min`, which is exactly what the old kernels called, so a
//! `to_bits()` match against them proves bit-identity.

use crate::extended::*;

// Shared test data: 10-bar OHLCV
pub(super) fn sample_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let high = vec![11.0, 12.0, 13.0, 14.0, 15.0, 14.5, 15.5, 16.0, 15.0, 14.0];
    let low = vec![9.0, 10.0, 11.0, 12.0, 13.0, 12.5, 13.5, 14.0, 13.0, 12.0];
    let close = vec![10.0, 11.0, 12.0, 13.0, 14.0, 13.5, 14.5, 15.0, 14.0, 13.0];
    let volume = vec![
        100.0, 150.0, 200.0, 250.0, 300.0, 200.0, 350.0, 400.0, 180.0, 220.0,
    ];
    (high, low, close, volume)
}

/// Bitwise identity, treating any two `NaN`s as equal (the payload of a
/// `NaN` is not part of any kernel's contract, the `NaN`-ness is).
pub(super) fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

pub(super) fn assert_bit_eq(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(same(a, e), "{label}[{i}]: {a:?} != reference {e:?}");
    }
}

pub(super) fn assert_close(actual: &[f64], expected: &[f64], rtol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if same(a, e) {
            continue;
        }
        assert!(
            a.is_nan() == e.is_nan(),
            "{label}[{i}]: NaN mismatch {a:?} vs reference {e:?}"
        );
        let tol = rtol * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: {a} vs reference {e} (tol {tol})"
        );
    }
}

/// A named `(high, low, close)` triple.
pub(super) type StressCase = (String, Vec<f64>, Vec<f64>, Vec<f64>);

/// Deterministic adversarial series: plateaus of equal values, monotone
/// runs (which keep the monotonic deques from ever draining), a mid-series
/// `NaN` in each input separately, degenerate lengths, and a 20 000-bar
/// series that crosses `RESEED_INTERVAL = 8192` twice.
pub(super) fn stress_cases() -> Vec<StressCase> {
    fn from_close(name: &str, close: Vec<f64>) -> StressCase {
        let high = close.iter().map(|c| c + 1.0).collect();
        let low = close.iter().map(|c| c - 1.0).collect();
        (name.to_string(), high, low, close)
    }

    let mut plateau = vec![5.0; 30];
    plateau.extend(vec![9.0; 30]);
    plateau.extend(vec![5.0; 30]);

    let mut cases = vec![
        from_close("empty", vec![]),
        from_close("single", vec![7.0]),
        from_close("plateau", plateau),
        from_close("monotone_up", (0..80).map(|i| 100.0 + i as f64).collect()),
        from_close("monotone_down", (0..80).map(|i| 200.0 - i as f64).collect()),
        from_close(
            "sawtooth",
            (0..90)
                .map(|i| 50.0 + ((i % 7) as f64) * 1.5 - ((i % 3) as f64))
                .collect(),
        ),
        from_close(
            "long_reseed",
            (0..20_000)
                .map(|i| 100.0 + ((i % 251) as f64) * 0.25)
                .collect(),
        ),
    ];

    // A mid-series NaN in each of the three inputs, one at a time.
    let base: Vec<f64> = (0..40).map(|i| 20.0 + ((i % 6) as f64)).collect();
    for which in 0..3 {
        let names = ["nan_high", "nan_low", "nan_close"];
        let (_, mut h, mut l, mut c) = from_close(names[which], base.clone());
        match which {
            0 => h[17] = f64::NAN,
            1 => l[17] = f64::NAN,
            _ => c[17] = f64::NAN,
        }
        cases.push((names[which].to_string(), h, l, c));
    }
    cases
}

/// Periods worth exercising for a series of `n` bars: degenerate, tiny,
/// typical, exactly `n`, and past the end.
pub(super) fn periods_for(n: usize) -> Vec<usize> {
    let mut p = vec![0usize, 1, 2, 3, 14, n + 1];
    if n > 0 {
        p.push(n);
    }
    p
}

/// Every multi-slice kernel in this module derives `n` from `high` (or
/// `close`) and indexes the rest, so a short or long companion must be
/// rejected up front rather than panicking. All-`NaN` of the expected
/// length, matching `utils::crossover` and `extended::trend::alligator`.
#[test]
fn mismatched_lengths_return_nan_without_panicking() {
    let long = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
    let short = vec![10.0, 11.0, 12.0];
    let longer = vec![1.0; long.len() + 4];
    let n = long.len();
    let all_nan = |v: &[f64]| v.len() == n && v.iter().all(|x| x.is_nan());

    assert!(all_nan(&vwap(&long, &short, &long, &long, 0)));
    assert!(all_nan(&vwap(&long, &long, &short, &long, 3)));
    assert!(all_nan(&vwap(&long, &long, &long, &longer, 3)));
    assert!(all_nan(&vwma(&long, &short, 3)));
    assert!(all_nan(&vwma(&long, &longer, 3)));
    assert!(all_nan(&donchian(&long, &short, 3).0));
    assert!(all_nan(&donchian(&long, &longer, 3).2));
    assert!(all_nan(&choppiness_index(&long, &short, &long, 3)));
    assert!(all_nan(&choppiness_index(&long, &long, &longer, 3)));
    assert!(all_nan(&chandelier_exit(&long, &long, &short, 3, 2.0).0));
    assert!(all_nan(&chandelier_exit(&long, &longer, &long, 3, 2.0).1));
    assert!(all_nan(
        &keltner_channels(&long, &short, &long, 3, 3, 1.5).0
    ));
    assert!(all_nan(
        &keltner_channels(&long, &long, &longer, 3, 3, 1.5).2
    ));
    assert!(all_nan(&ichimoku(&long, &short, &long, 2, 3, 4, 2).0));
    assert!(all_nan(&ichimoku(&long, &long, &longer, 2, 3, 4, 2).4));
    assert!(all_nan(&pivot_points(&long, &short, &long, "classic").0));
    assert!(all_nan(&pivot_points(&long, &long, &longer, "classic").3));

    let (st, dir) = supertrend(&long, &short, &long, 3, 2.0);
    assert!(all_nan(&st));
    assert!(dir.len() == n && dir.iter().all(|&d| d == 0));
}
