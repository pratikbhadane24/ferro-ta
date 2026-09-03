//! Shared synthetic-input generators for the ferro_ta_core bench targets.
//!
//! Included by both `benches/indicators.rs` (legacy core sweep) and
//! `benches/extended.rs` (extended catalog); each target uses a subset, hence
//! the blanket `dead_code` allow.
#![allow(dead_code)]

use ferro_ta_core::utils;

/// Sizes for kernels touched by the rolling-window optimization work.
pub const HOT_SIZES: [usize; 2] = [10_000, 100_000];
/// Single representative size for the broad extended catalog.
pub const CATALOG_SIZE: usize = 100_000;

pub fn synthetic_close(n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut price = 100.0_f64;
    for i in 0..n {
        price += ((i as f64 * 0.1).sin()) * 0.5;
        v.push(price);
    }
    v
}

pub fn synthetic_high_low_close(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let close = synthetic_close(n);
    let high: Vec<f64> = close.iter().map(|&c| c + 0.5).collect();
    let low: Vec<f64> = close.iter().map(|&c| c - 0.5).collect();
    (high, low, close)
}

/// Full OHLCV bars — several extended kernels need `open` and/or `volume`,
/// which the close-only helpers above do not produce. Deterministic (no RNG
/// dependency); volumes are strictly positive and non-constant so that
/// volume-weighted kernels do not degenerate to an unweighted average.
pub struct Ohlcv {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

pub fn synthetic_ohlcv(n: usize) -> Ohlcv {
    let close = synthetic_close(n);
    let open: Vec<f64> = close
        .iter()
        .enumerate()
        .map(|(i, &c)| c - (i as f64 * 0.1).cos() * 0.3)
        .collect();
    let high: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .map(|(&o, &c)| o.max(c) + 0.5)
        .collect();
    let low: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .map(|(&o, &c)| o.min(c) - 0.5)
        .collect();
    // 1250..2250, never zero, never flat.
    let volume: Vec<f64> = (0..n)
        .map(|i| 1_000.0 + ((i as f64 * 0.05).sin() + 1.5) * 500.0)
        .collect();
    Ohlcv {
        open,
        high,
        low,
        close,
        volume,
    }
}

/// Two series that oscillate around the same level with a quarter-period phase
/// offset, so they cross repeatedly. Benchmarking `crossover` and friends
/// against a pair that never crosses would measure the wrong branch.
pub fn synthetic_crossing_pair(n: usize) -> (Vec<f64>, Vec<f64>) {
    let fast: Vec<f64> = (0..n)
        .map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0)
        .collect();
    let slow: Vec<f64> = (0..n)
        .map(|i| 100.0 + (i as f64 * 0.1 + std::f64::consts::FRAC_PI_2).sin() * 1.5)
        .collect();
    (fast, slow)
}

/// Binary buy/sell signal series derived from a crossing pair — the natural
/// input shape for `exrem` / `flip` / `valuewhen`.
pub fn synthetic_signals(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (fast, slow) = synthetic_crossing_pair(n);
    (
        utils::crossover(&fast, &slow),
        utils::crossunder(&fast, &slow),
    )
}
