//! Deterministic test data shared by the streaming tests.
//!
//! An LCG keeps the tests free of an RNG dependency and bit-identical across
//! platforms.

/// Seed of [`bars`] (the live-bar and registry tests).
const LIVE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// `(high, low, close, volume)` random walk from `seed`.
pub(crate) fn bars_seeded(seed: u64, n: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut close = 100.0;
    (0..n)
        .map(|_| {
            close += next() - 0.5;
            let high = close + next();
            let low = close - next();
            (high, low, close, 1.0 + 10.0 * next())
        })
        .collect()
}

/// `(high, low, close, volume)` random walk used by most streaming tests.
pub(crate) fn bars(n: usize) -> Vec<(f64, f64, f64, f64)> {
    bars_seeded(LIVE_SEED, n)
}

/// Column for a batch input argument, by its FFI name, over `bars(n)`.
///
/// `open` is the previous close clamped into this bar's range (the walk has
/// no opens of its own). Inputs that are not raw OHLCV map onto close- or
/// open-derived columns so two-input streams see two different series.
///
/// # Panics
/// On an input name with no mapping yet: add it here.
pub(crate) fn input_column(name: &str, n: usize) -> Vec<f64> {
    let data = bars(n);
    let close: Vec<f64> = data.iter().map(|b| b.2).collect();
    let open: Vec<f64> = data
        .iter()
        .enumerate()
        .map(|(i, &(h, l, _, _))| {
            let prev = if i == 0 { 100.0 } else { data[i - 1].2 };
            prev.clamp(l, h)
        })
        .collect();
    match name {
        "open" => open,
        "high" => data.iter().map(|b| b.0).collect(),
        "low" => data.iter().map(|b| b.1).collect(),
        "close" => close,
        "volume" => data.iter().map(|b| b.3).collect(),
        "real" | "real0" | "value" | "x" | "series" | "iv_series" | "asset" | "a" | "front"
        | "position_size" => close,
        "real1" | "b" | "benchmark" | "next" => open,
        "periods" => (0..n).map(|i| 5.0 + (i % 7) as f64).collect(),
        "next_weights" => (0..n).map(|i| (i as f64 / 300.0).min(1.0)).collect(),
        "funding_rate" => (0..n).map(|i| 1e-4 * (i as f64).sin()).collect(),
        other => panic!("no fixture column for input {other:?}; add it to input_column"),
    }
}
