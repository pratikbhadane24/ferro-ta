//! Criterion benchmarks for ferro_ta_core — pure Rust indicator throughput.
//!
//! Run from repo root:  cargo bench -p ferro_ta_core
//! Or:  cd crates/ferro_ta_core && cargo bench
//!
//! Input sizes: 1k, 10k, 100k, and 1M bars for key indicators.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferro_ta_core::{momentum, overlap, volatility};

fn synthetic_close(n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut price = 100.0_f64;
    for i in 0..n {
        price += ((i as f64 * 0.1).sin()) * 0.5;
        v.push(price);
    }
    v
}

fn synthetic_high_low_close(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let close = synthetic_close(n);
    let high: Vec<f64> = close.iter().map(|&c| c + 0.5).collect();
    let low: Vec<f64> = close.iter().map(|&c| c - 0.5).collect();
    (high, low, close)
}

fn bench_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group("SMA");
    for size in [1_000_usize, 10_000, 100_000, 1_000_000] {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| overlap::sma(black_box(close), 14))
        });
    }
    group.finish();
}

fn bench_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group("EMA");
    for size in [1_000_usize, 10_000, 100_000, 1_000_000] {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| overlap::ema(black_box(close), 14))
        });
    }
    group.finish();
}

fn bench_rsi(c: &mut Criterion) {
    let mut group = c.benchmark_group("RSI");
    for size in [1_000_usize, 10_000, 100_000, 1_000_000] {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| momentum::rsi(black_box(close), 14))
        });
    }
    group.finish();
}

fn bench_atr(c: &mut Criterion) {
    let mut group = c.benchmark_group("ATR");
    for size in [1_000_usize, 10_000, 100_000, 1_000_000] {
        let (high, low, close) = synthetic_high_low_close(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(high.clone(), low.clone(), close),
            |b, (high, low, close)| {
                b.iter(|| volatility::atr(black_box(high), black_box(low), black_box(close), 14))
            },
        );
    }
    group.finish();
}

fn bench_bbands(c: &mut Criterion) {
    let mut group = c.benchmark_group("BBANDS");
    for size in [1_000_usize, 10_000, 100_000, 1_000_000] {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| overlap::bbands(black_box(close), 20, 2.0, 2.0))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sma,
    bench_ema,
    bench_rsi,
    bench_atr,
    bench_bbands
);
criterion_main!(benches);
