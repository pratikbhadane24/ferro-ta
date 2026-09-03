//! Criterion benchmarks for the ferro_ta_core extended indicator catalog
//! (`extended::*` and `utils::*`).
//!
//! Run:  cargo bench -p ferro_ta_core --bench extended
//!
//! # Size strategy
//!
//! The extended catalog is large enough (~70 kernels) that putting every one
//! on the legacy `[1k, 10k, 100k, 1M]` sweep would make a single run take
//! hours, and a bench nobody runs is worthless. So this target has two tiers:
//!
//! * **Hot rolling-window kernels** (`HOT_SIZES` = `[10k, 100k]`) — the
//!   kernels sitting directly on the shared rolling-window machinery
//!   (`math::sliding_max` / `sliding_min` and the rolling module) plus the
//!   two-series signal utilities. These are what the imminent optimization
//!   pass rewrites, so they get two sizes: a level regression and a scaling
//!   regression look different, and one size cannot tell them apart.
//! * **Broad catalog** (`CATALOG_SIZE` = 100k, group `EXTENDED_100K`) — every
//!   remaining extended kernel at one representative size. Enough to make a
//!   regression anywhere in the catalog visible, cheap enough to actually run.
//!
//! Note that CI's `rust_bench` check is `cargo bench --no-run` (compile-only),
//! so nothing here is enforced automatically: these benches exist for humans
//! doing before/after comparisons, and must at minimum keep compiling.
//!
//! Periods match the defaults declared on the PyO3 wrappers in `src/extended/`
//! (and `benchmarks/wrapper_registry.py`'s `INDICATOR_DEFAULTS` where a name
//! appears there), so the Rust and Python numbers describe the same work.
mod common;

use common::{
    synthetic_close, synthetic_crossing_pair, synthetic_high_low_close, synthetic_ohlcv,
    synthetic_signals, CATALOG_SIZE, HOT_SIZES,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ferro_ta_core::{extended, utils};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Hot rolling-window kernels — HOT_SIZES = [10k, 100k].
//
// These sit directly on the shared sliding-window machinery that the upcoming
// optimization pass rewrites, so they get two sizes: a level regression and a
// scaling regression look different, and one size cannot tell them apart.
// ---------------------------------------------------------------------------

fn bench_donchian(c: &mut Criterion) {
    let mut group = c.benchmark_group("DONCHIAN");
    for size in HOT_SIZES {
        let (high, low, _) = synthetic_high_low_close(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(high, low),
            |b, (high, low)| b.iter(|| extended::donchian(black_box(high), black_box(low), 20)),
        );
    }
    group.finish();
}

fn bench_ichimoku(c: &mut Criterion) {
    let mut group = c.benchmark_group("ICHIMOKU");
    for size in HOT_SIZES {
        let (high, low, close) = synthetic_high_low_close(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(high, low, close),
            |b, (high, low, close)| {
                b.iter(|| {
                    extended::ichimoku(
                        black_box(high),
                        black_box(low),
                        black_box(close),
                        9,
                        26,
                        52,
                        26,
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_chandelier_exit(c: &mut Criterion) {
    let mut group = c.benchmark_group("CHANDELIER_EXIT");
    for size in HOT_SIZES {
        let (high, low, close) = synthetic_high_low_close(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(high, low, close),
            |b, (high, low, close)| {
                b.iter(|| {
                    extended::chandelier_exit(
                        black_box(high),
                        black_box(low),
                        black_box(close),
                        22,
                        3.0,
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_choppiness_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("CHOPPINESS_INDEX");
    for size in HOT_SIZES {
        let (high, low, close) = synthetic_high_low_close(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(high, low, close),
            |b, (high, low, close)| {
                b.iter(|| {
                    extended::choppiness_index(
                        black_box(high),
                        black_box(low),
                        black_box(close),
                        14,
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_hull_ma(c: &mut Criterion) {
    let mut group = c.benchmark_group("HULL_MA");
    for size in HOT_SIZES {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| extended::hull_ma(black_box(close), 16))
        });
    }
    group.finish();
}

fn bench_median(c: &mut Criterion) {
    let mut group = c.benchmark_group("MEDIAN");
    for size in HOT_SIZES {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| extended::median(black_box(close), 3))
        });
    }
    group.finish();
}

fn bench_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("MODE");
    for size in HOT_SIZES {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| extended::mode(black_box(close), 20, 10))
        });
    }
    group.finish();
}

fn bench_vortex(c: &mut Criterion) {
    let mut group = c.benchmark_group("VORTEX");
    for size in HOT_SIZES {
        let (high, low, close) = synthetic_high_low_close(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(high, low, close),
            |b, (high, low, close)| {
                b.iter(|| extended::vortex(black_box(high), black_box(low), black_box(close), 14))
            },
        );
    }
    group.finish();
}

fn bench_kvo(c: &mut Criterion) {
    let mut group = c.benchmark_group("KVO");
    for size in HOT_SIZES {
        let d = synthetic_ohlcv(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &d, |b, d| {
            b.iter(|| {
                extended::kvo(
                    black_box(&d.high),
                    black_box(&d.low),
                    black_box(&d.close),
                    black_box(&d.volume),
                    34,
                    55,
                    13,
                )
            })
        });
    }
    group.finish();
}

fn bench_stc(c: &mut Criterion) {
    let mut group = c.benchmark_group("STC");
    for size in HOT_SIZES {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| extended::stc(black_box(close), 23, 50, 10, 3, 3))
        });
    }
    group.finish();
}

fn bench_highest(c: &mut Criterion) {
    let mut group = c.benchmark_group("HIGHEST");
    for size in HOT_SIZES {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| utils::highest(black_box(close), 30))
        });
    }
    group.finish();
}

fn bench_lowest(c: &mut Criterion) {
    let mut group = c.benchmark_group("LOWEST");
    for size in HOT_SIZES {
        let close = synthetic_close(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &close, |b, close| {
            b.iter(|| utils::lowest(black_box(close), 30))
        });
    }
    group.finish();
}

fn bench_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("CROSSOVER");
    for size in HOT_SIZES {
        let pair = synthetic_crossing_pair(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &pair,
            |b, (fast, slow)| b.iter(|| utils::crossover(black_box(fast), black_box(slow))),
        );
    }
    group.finish();
}

fn bench_crossunder(c: &mut Criterion) {
    let mut group = c.benchmark_group("CROSSUNDER");
    for size in HOT_SIZES {
        let pair = synthetic_crossing_pair(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &pair,
            |b, (fast, slow)| b.iter(|| utils::crossunder(black_box(fast), black_box(slow))),
        );
    }
    group.finish();
}

fn bench_cross(c: &mut Criterion) {
    let mut group = c.benchmark_group("CROSS");
    for size in HOT_SIZES {
        let pair = synthetic_crossing_pair(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &pair,
            |b, (fast, slow)| b.iter(|| utils::cross(black_box(fast), black_box(slow))),
        );
    }
    group.finish();
}

fn bench_valuewhen(c: &mut Criterion) {
    let mut group = c.benchmark_group("VALUEWHEN");
    for size in HOT_SIZES {
        let (fast, _) = synthetic_crossing_pair(size);
        let (buy, _) = synthetic_signals(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(buy, fast),
            |b, (buy, fast)| b.iter(|| utils::valuewhen(black_box(buy), black_box(fast), 1)),
        );
    }
    group.finish();
}

fn bench_exrem(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXREM");
    for size in HOT_SIZES {
        let signals = synthetic_signals(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &signals,
            |b, (buy, sell)| b.iter(|| utils::exrem(black_box(buy), black_box(sell))),
        );
    }
    group.finish();
}

fn bench_flip(c: &mut Criterion) {
    let mut group = c.benchmark_group("FLIP");
    for size in HOT_SIZES {
        let signals = synthetic_signals(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &signals,
            |b, (buy, sell)| b.iter(|| utils::flip(black_box(buy), black_box(sell))),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Broad extended catalog — every remaining kernel at CATALOG_SIZE (100k).
//
// All of these share the single logical group `EXTENDED_100K` (split across
// several small fns purely to keep each one readable), so one Criterion filter
// runs the whole catalog:  cargo bench -p ferro_ta_core -- EXTENDED_100K
// One size is enough here: the point is that a regression *anywhere* in the
// new catalog shows up, not to characterize each kernel's scaling curve.
// ---------------------------------------------------------------------------

fn bench_catalog_trend(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXTENDED_100K");
    let d = synthetic_ohlcv(CATALOG_SIZE);
    group.bench_function("alma", |b| {
        b.iter(|| extended::alma(black_box(&d.close), 21, 0.85, 6.0))
    });
    group.bench_function("zlema", |b| {
        b.iter(|| extended::zlema(black_box(&d.close), 14))
    });
    group.bench_function("frama", |b| {
        b.iter(|| extended::frama(black_box(&d.close), 16))
    });
    group.bench_function("mcginley", |b| {
        b.iter(|| extended::mcginley(black_box(&d.close), 14))
    });
    group.bench_function("vidya", |b| {
        b.iter(|| extended::vidya(black_box(&d.close), 14, 9))
    });
    group.bench_function("alligator", |b| {
        b.iter(|| extended::alligator(black_box(&d.high), black_box(&d.low), 13, 8, 8, 5, 5, 3))
    });
    group.bench_function("ma_envelopes", |b| {
        b.iter(|| extended::ma_envelopes(black_box(&d.close), 20, 2.5, 0))
    });
    group.bench_function("chande_kroll_stop", |b| {
        b.iter(|| {
            extended::chande_kroll_stop(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                10,
                1.0,
                9,
            )
        })
    });
    group.finish();
}

fn bench_catalog_momentum(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXTENDED_100K");
    let d = synthetic_ohlcv(CATALOG_SIZE);
    group.bench_function("elder_ray", |b| {
        b.iter(|| {
            extended::elder_ray(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                13,
            )
        })
    });
    group.bench_function("fisher", |b| {
        b.iter(|| extended::fisher(black_box(&d.high), black_box(&d.low), 9))
    });
    group.bench_function("crsi", |b| {
        b.iter(|| extended::crsi(black_box(&d.close), 3, 2, 100))
    });
    group.finish();
}

fn bench_catalog_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXTENDED_100K");
    let d = synthetic_ohlcv(CATALOG_SIZE);
    group.bench_function("chaikin_vol", |b| {
        b.iter(|| extended::chaikin_vol(black_box(&d.high), black_box(&d.low), 10, 10))
    });
    group.bench_function("mass", |b| {
        b.iter(|| extended::mass(black_box(&d.high), black_box(&d.low), 9, 25))
    });
    group.bench_function("bbpercent", |b| {
        b.iter(|| extended::bbpercent(black_box(&d.close), 5, 2.0, 2.0))
    });
    group.bench_function("bbwidth", |b| {
        b.iter(|| extended::bbwidth(black_box(&d.close), 5, 2.0, 2.0))
    });
    group.bench_function("historical_volatility", |b| {
        b.iter(|| extended::historical_volatility(black_box(&d.close), 20, 252.0))
    });
    group.bench_function("ulcer_index", |b| {
        b.iter(|| extended::ulcer_index(black_box(&d.close), 14))
    });
    group.bench_function("starc", |b| {
        b.iter(|| {
            extended::starc(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                15,
                15,
                2.0,
            )
        })
    });
    group.finish();
}

fn bench_catalog_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXTENDED_100K");
    let d = synthetic_ohlcv(CATALOG_SIZE);
    group.bench_function("obv_smoothed", |b| {
        b.iter(|| extended::obv_smoothed(black_box(&d.close), black_box(&d.volume), 20, 1))
    });
    group.bench_function("cmf", |b| {
        b.iter(|| {
            extended::cmf(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                black_box(&d.volume),
                20,
            )
        })
    });
    group.bench_function("emv", |b| {
        b.iter(|| {
            extended::emv(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.volume),
                14,
                10_000.0,
            )
        })
    });
    group.bench_function("force_index", |b| {
        b.iter(|| extended::force_index(black_box(&d.close), black_box(&d.volume), 13))
    });
    group.bench_function("nvi", |b| {
        b.iter(|| extended::nvi(black_box(&d.close), black_box(&d.volume)))
    });
    group.bench_function("nvi_with_ema", |b| {
        b.iter(|| extended::nvi_with_ema(black_box(&d.close), black_box(&d.volume), 255))
    });
    group.bench_function("pvi", |b| {
        b.iter(|| extended::pvi(black_box(&d.close), black_box(&d.volume)))
    });
    group.bench_function("pvi_with_signal", |b| {
        b.iter(|| extended::pvi_with_signal(black_box(&d.close), black_box(&d.volume), 255, 1))
    });
    group.bench_function("volosc", |b| {
        b.iter(|| extended::volosc(black_box(&d.volume), 5, 10))
    });
    group.bench_function("vroc", |b| {
        b.iter(|| extended::vroc(black_box(&d.volume), 25))
    });
    group.bench_function("pvt", |b| {
        b.iter(|| extended::pvt(black_box(&d.close), black_box(&d.volume)))
    });
    group.bench_function("rvol", |b| {
        b.iter(|| extended::rvol(black_box(&d.volume), 20))
    });
    group.finish();
}

fn bench_catalog_oscillators(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXTENDED_100K");
    let d = synthetic_ohlcv(CATALOG_SIZE);
    group.bench_function("ao", |b| {
        b.iter(|| extended::ao(black_box(&d.high), black_box(&d.low), 5, 34))
    });
    group.bench_function("ac", |b| {
        b.iter(|| extended::ac(black_box(&d.high), black_box(&d.low), 5, 34, 5))
    });
    group.bench_function("po", |b| {
        b.iter(|| extended::po(black_box(&d.close), 10, 21))
    });
    group.bench_function("dpo", |b| b.iter(|| extended::dpo(black_box(&d.close), 20)));
    group.bench_function("rvi", |b| {
        b.iter(|| {
            extended::rvi(
                black_box(&d.open),
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                10,
            )
        })
    });
    group.bench_function("cho", |b| {
        b.iter(|| {
            extended::cho(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                black_box(&d.volume),
                3,
                10,
            )
        })
    });
    group.bench_function("kst", |b| {
        b.iter(|| extended::kst(black_box(&d.close), 10, 15, 20, 30, 10, 10, 10, 15, 9))
    });
    group.bench_function("tsi", |b| {
        b.iter(|| extended::tsi(black_box(&d.close), 25, 13, 13))
    });
    group.bench_function("gator", |b| {
        b.iter(|| extended::gator(black_box(&d.high), black_box(&d.low), 13, 8, 8, 5, 5, 3))
    });
    group.bench_function("coppock", |b| {
        b.iter(|| extended::coppock(black_box(&d.close), 10, 14, 11))
    });
    group.finish();
}

fn bench_catalog_stat_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXTENDED_100K");
    let d = synthetic_ohlcv(CATALOG_SIZE);
    group.bench_function("median_bands", |b| {
        b.iter(|| {
            extended::median_bands(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                3,
                14,
                2.0,
            )
        })
    });
    group.bench_function("dmi", |b| {
        b.iter(|| {
            extended::dmi(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                14,
            )
        })
    });
    group.bench_function("williams_fractals", |b| {
        b.iter(|| extended::williams_fractals(black_box(&d.high), black_box(&d.low), 2))
    });
    group.bench_function("rwi", |b| {
        b.iter(|| {
            extended::rwi(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                14,
            )
        })
    });
    group.finish();
}

fn bench_catalog_existing(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXTENDED_100K");
    let d = synthetic_ohlcv(CATALOG_SIZE);
    group.bench_function("vwap", |b| {
        b.iter(|| {
            extended::vwap(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                black_box(&d.volume),
                0,
            )
        })
    });
    group.bench_function("vwma", |b| {
        b.iter(|| extended::vwma(black_box(&d.close), black_box(&d.volume), 20))
    });
    group.bench_function("supertrend", |b| {
        b.iter(|| {
            extended::supertrend(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                7,
                3.0,
            )
        })
    });
    group.bench_function("keltner_channels", |b| {
        b.iter(|| {
            extended::keltner_channels(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                20,
                10,
                2.0,
            )
        })
    });
    group.bench_function("pivot_points", |b| {
        b.iter(|| {
            extended::pivot_points(
                black_box(&d.high),
                black_box(&d.low),
                black_box(&d.close),
                "classic",
            )
        })
    });
    group.finish();
}

fn bench_catalog_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXTENDED_100K");
    let close = synthetic_close(CATALOG_SIZE);
    group.bench_function("change", |b| b.iter(|| utils::change(black_box(&close), 1)));
    group.bench_function("rising", |b| b.iter(|| utils::rising(black_box(&close), 1)));
    group.bench_function("falling", |b| {
        b.iter(|| utils::falling(black_box(&close), 1))
    });
    group.finish();
}
criterion_group!(
    benches,
    // Hot rolling-window kernels (10k, 100k)
    bench_donchian,
    bench_ichimoku,
    bench_chandelier_exit,
    bench_choppiness_index,
    bench_hull_ma,
    bench_median,
    bench_mode,
    bench_vortex,
    bench_kvo,
    bench_stc,
    bench_highest,
    bench_lowest,
    bench_crossover,
    bench_crossunder,
    bench_cross,
    bench_valuewhen,
    bench_exrem,
    bench_flip,
    // Broad extended catalog (100k)
    bench_catalog_trend,
    bench_catalog_momentum,
    bench_catalog_volatility,
    bench_catalog_volume,
    bench_catalog_oscillators,
    bench_catalog_stat_hybrid,
    bench_catalog_existing,
    bench_catalog_utils
);
criterion_main!(benches);
