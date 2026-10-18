//! Deterministic test data shared by the streaming tests.
//!
//! An LCG keeps the tests free of an RNG dependency and bit-identical across
//! platforms.

/// Bars in the trailing mean that makes the `slow` column.
const SLOW_MEAN_BARS: usize = 8;

/// Mean of the last `k` values (fewer while warming up).
fn trailing_mean(v: &[f64], k: usize) -> Vec<f64> {
    (0..v.len())
        .map(|i| {
            let w = &v[(i + 1).saturating_sub(k)..=i];
            w.iter().sum::<f64>() / w.len() as f64
        })
        .collect()
}

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
        "real" | "real0" | "value" | "x" | "asset" | "a" | "front" | "position_size" => close,
        "real1" | "b" | "benchmark" | "next" => open,
        "fast" => close,
        "slow" => trailing_mean(&close, SLOW_MEAN_BARS),
        // Calm walk with a volatility burst every third 40-bar block.
        "series" => close
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if (i / 40) % 3 == 2 {
                    c + 3.0 * (2.1 * i as f64).sin()
                } else {
                    c
                }
            })
            .collect(),
        // Positive implied-vol-like level.
        "iv_series" => (0..n)
            .map(|i| 0.2 + 0.05 * (0.05 * i as f64).sin() + 0.03 * (0.31 * i as f64).sin())
            .collect(),
        // ADX crossing the default 25 threshold; ATR / close crossing 0.005.
        "adx" => (0..n)
            .map(|i| 32.5 + 27.5 * (0.21 * i as f64).sin())
            .collect(),
        "atr" => (0..n)
            .map(|i| 0.5 + 0.3 * (0.37 * i as f64).sin())
            .collect(),
        "asset_returns" => (0..n)
            .map(|i| 0.01 * (0.7 * i as f64).sin() + 0.002 * (1.9 * i as f64).cos())
            .collect(),
        "benchmark_returns" => (0..n).map(|i| 0.008 * (0.45 * i as f64).cos()).collect(),
        "periods" => (0..n).map(|i| 5.0 + (i % 7) as f64).collect(),
        "next_weights" => (0..n).map(|i| (i as f64 / 300.0).min(1.0)).collect(),
        "funding_rate" => (0..n).map(|i| 1e-4 * (i as f64).sin()).collect(),
        other => panic!("no fixture column for input {other:?}; add it to input_column"),
    }
}
