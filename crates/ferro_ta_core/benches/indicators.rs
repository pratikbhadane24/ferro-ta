//! Criterion benchmarks for ferro_ta_core — pure Rust indicator throughput.
//!
//! Run from repo root:  cargo bench -p ferro_ta_core
//! Or:  cd crates/ferro_ta_core && cargo bench
//!
//! Input sizes: 1k, 10k, 100k, and 1M bars for key indicators.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ferro_ta_core::{futures, momentum, options, overlap, volatility};
use std::hint::black_box;

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

fn bench_bsm_price(c: &mut Criterion) {
    let mut group = c.benchmark_group("BSM_PRICE");
    for size in [1_000_usize, 10_000, 100_000] {
        let close = synthetic_close(size);
        let strikes: Vec<f64> = close.iter().map(|_| 100.0).collect();
        let vols: Vec<f64> = close.iter().map(|_| 0.2).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| {
                close
                    .iter()
                    .zip(strikes.iter())
                    .zip(vols.iter())
                    .map(|((&spot, &strike), &vol)| {
                        options::pricing::black_scholes_price(
                            black_box(spot),
                            black_box(strike),
                            black_box(0.02),
                            black_box(0.0),
                            black_box(0.5),
                            black_box(vol),
                            options::OptionKind::Call,
                        )
                    })
                    .collect::<Vec<_>>()
            })
        });
    }
    group.finish();
}

fn bench_implied_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("IMPLIED_VOL");
    for size in [1_000_usize, 10_000] {
        let prices: Vec<f64> = (0..size)
            .map(|i| {
                let spot = 90.0 + (i % 20) as f64;
                options::pricing::black_scholes_price(
                    spot,
                    100.0,
                    0.02,
                    0.0,
                    0.5,
                    0.2,
                    options::OptionKind::Call,
                )
            })
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &prices, |b, prices| {
            b.iter(|| {
                prices
                    .iter()
                    .enumerate()
                    .map(|(i, &price)| {
                        options::iv::implied_volatility(
                            options::OptionContract {
                                model: options::PricingModel::BlackScholes,
                                underlying: black_box(90.0 + (i % 20) as f64),
                                strike: black_box(100.0),
                                rate: black_box(0.02),
                                carry: black_box(0.0),
                                time_to_expiry: black_box(0.5),
                                kind: options::OptionKind::Call,
                            },
                            black_box(price),
                            options::IvSolverConfig {
                                initial_guess: black_box(0.25),
                                tolerance: black_box(1e-8),
                                max_iterations: black_box(100),
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            })
        });
    }
    group.finish();
}

fn bench_smile_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("SMILE_METRICS");
    let strikes: Vec<f64> = (0..41).map(|i| 80.0 + i as f64).collect();
    let vols: Vec<f64> = strikes
        .iter()
        .map(|&k| 0.18 + ((k - 100.0).abs() / 100.0) * 0.15)
        .collect();
    group.bench_function("single_chain", |b| {
        b.iter(|| {
            options::surface::smile_metrics(
                black_box(&strikes),
                black_box(&vols),
                black_box(100.0),
                black_box(0.02),
                black_box(0.0),
                black_box(0.5),
                options::PricingModel::BlackScholes,
            )
        })
    });
    group.finish();
}

fn bench_curve_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("FUTURES_CURVE");
    let tenors = vec![0.1, 0.25, 0.5, 0.75, 1.0];
    let prices = vec![101.0, 101.8, 102.7, 103.4, 104.1];
    group.bench_function("curve_summary", |b| {
        b.iter(|| {
            futures::curve::curve_summary(black_box(100.0), black_box(&tenors), black_box(&prices))
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_sma,
    bench_ema,
    bench_rsi,
    bench_atr,
    bench_bbands,
    bench_bsm_price,
    bench_implied_volatility,
    bench_smile_metrics,
    bench_curve_summary
);
criterion_main!(benches);
