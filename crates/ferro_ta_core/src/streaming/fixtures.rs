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

/// Bars in [`pattern_fixture`].
pub(crate) const PATTERN_BARS: usize = 2000;

/// Seed of [`pattern_fixture`].
const PATTERN_SEED: u64 = 0xC0FF_EE15_BADD_CAFE;

/// Candle shape of one [`pattern_fixture`] bar.
struct Candle {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

impl Candle {
    /// Body from `open` to `close`, wicks `upper` / `lower` beyond it.
    fn new(open: f64, close: f64, upper: f64, lower: f64) -> Self {
        Self {
            open,
            high: open.max(close) + upper,
            low: open.min(close) - lower,
            close,
        }
    }
}

/// Gap kinds drawn per bar: none, small, large (cumulative thresholds).
const GAP_NONE_BELOW: f64 = 0.45;
const GAP_SMALL_BELOW: f64 = 0.7;
/// Body kinds: doji, small, marubozu-like, otherwise long (cumulative).
const BODY_DOJI_BELOW: f64 = 0.15;
const BODY_SMALL_BELOW: f64 = 0.35;
/// Chance a bar follows the current trend direction.
const TREND_FOLLOW: f64 = 0.7;
/// Chance per bar that the trend direction flips.
const TREND_FLIP: f64 = 0.2;
/// Chance a wick is absent (marubozu / belt-hold / hammer shapes).
const WICK_ABSENT: f64 = 0.25;
/// Price band the walk is pulled back into.
const PATTERN_BASE: f64 = 1000.0;
const PATTERN_BAND: f64 = 60.0;

/// `(open, high, low, close)` of `n` varied candles: dojis, marubozus, long
/// and short bodies, gaps and trend runs, so that most candlestick rules fire
/// by chance, then [`inject_shapes`] adds the rare multi-bar shapes. A pure
/// random walk fires almost none of the 61 patterns, which would make
/// parity against batch vacuous.
pub(crate) fn pattern_fixture(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state = PATTERN_SEED;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut candles: Vec<Candle> = Vec::with_capacity(n);
    let mut close = PATTERN_BASE;
    let mut dir = 1.0;
    for _ in 0..n {
        if next() < TREND_FLIP {
            dir = -dir;
        }
        if close > PATTERN_BASE + PATTERN_BAND {
            dir = -1.0;
        } else if close < PATTERN_BASE - PATTERN_BAND {
            dir = 1.0;
        }
        let g = next();
        let gap = if g < GAP_NONE_BELOW {
            0.0
        } else if g < GAP_SMALL_BELOW {
            (next() - 0.5) * 0.8
        } else {
            (next() - 0.5) * 6.0
        };
        let open = close + gap;
        let b = next();
        let size = if b < BODY_DOJI_BELOW {
            next() * 0.05
        } else if b < BODY_SMALL_BELOW {
            0.2 + next() * 0.6
        } else {
            1.0 + next() * 3.0
        };
        let side = if next() < TREND_FOLLOW { dir } else { -dir };
        let shadow = |r: f64, scale: f64| if r < WICK_ABSENT { 0.0 } else { r * scale };
        let wick_scale = if b < BODY_DOJI_BELOW { 3.0 } else { 1.5 };
        let (ru, rl) = (next(), next());
        let candle = Candle::new(
            open,
            open + side * size,
            shadow(ru, wick_scale),
            shadow(rl, wick_scale),
        );
        close = candle.close;
        candles.push(candle);
    }
    inject_shapes(&mut candles);
    (
        candles.iter().map(|c| c.open).collect(),
        candles.iter().map(|c| c.high).collect(),
        candles.iter().map(|c| c.low).collect(),
        candles.iter().map(|c| c.close).collect(),
    )
}

/// A scripted shape: bars as `[open, high, low, close]` offsets from the
/// close before it.
type Shape = &'static [[f64; 4]];

/// Shapes the random candles almost never produce, one per rare pattern
/// (bullish where the rule has a direction): 2 crows, 3 black crows,
/// 3-line strike, 3 stars in the south, 3 white soldiers, abandoned baby,
/// concealing baby swallow, kicking (by length), mat hold / rising three
/// methods, tri-star, on-neck.
const SHAPES: &[Shape] = &[
    &[
        [0.0, 5.5, -0.5, 5.0],
        [7.0, 7.5, 5.5, 6.0],
        [6.5, 6.8, 2.5, 3.0],
    ],
    &[
        [10.0, 10.5, 3.5, 4.0],
        [7.0, 7.5, 0.5, 1.0],
        [4.0, 4.5, -2.5, -2.0],
    ],
    &[
        [10.0, 10.2, 7.8, 8.0],
        [8.5, 8.7, 5.8, 6.0],
        [6.5, 6.7, 3.8, 4.0],
        [3.5, 11.2, 3.4, 11.0],
    ],
    &[
        [10.0, 10.5, 3.5, 4.0],
        [9.0, 9.2, 4.8, 5.0],
        [7.0, 7.0, 5.5, 5.5],
    ],
    &[
        [0.0, 6.5, -0.2, 6.0],
        [3.0, 9.5, 2.8, 9.0],
        [6.0, 12.5, 5.8, 12.0],
    ],
    &[
        [10.0, 10.2, 3.8, 4.0],
        [2.0, 2.5, 1.5, 2.05],
        [3.0, 9.2, 2.8, 9.0],
    ],
    &[
        [20.0, 20.0, 14.0, 14.0],
        [13.0, 13.0, 7.0, 7.0],
        [6.0, 7.5, 3.8, 4.0],
        [6.5, 8.0, 3.4, 3.5],
    ],
    &[[10.0, 10.0, 5.0, 5.0], [12.0, 18.0, 12.0, 18.0]],
    &[
        [0.0, 10.5, -0.3, 10.0],
        [9.0, 9.2, 7.9, 8.0],
        [8.0, 8.2, 6.9, 7.0],
        [7.0, 7.2, 5.4, 5.5],
        [6.0, 12.2, 5.9, 12.0],
    ],
    &[
        [10.0, 11.0, 9.0, 10.05],
        [9.0, 10.0, 8.0, 9.05],
        [9.0, 10.5, 7.5, 9.2],
    ],
    &[[10.0, 10.2, 4.0, 4.5], [3.0, 4.1, 2.9, 4.0]],
];

/// Bars between scripted shapes, and the first bar that may hold one.
const SHAPE_STRIDE: usize = 40;
const SHAPE_FIRST: usize = 20;

/// Write [`SHAPES`] over the random candles, cycling through them, so every
/// pattern fires several times (see `every_pattern_fires_on_fixture`).
fn inject_shapes(candles: &mut [Candle]) {
    let mut pos = SHAPE_FIRST;
    for shape in SHAPES.iter().cycle() {
        if pos + shape.len() > candles.len() {
            break;
        }
        let base = candles[pos - 1].close;
        for (k, bar) in shape.iter().enumerate() {
            candles[pos + k] = Candle {
                open: base + bar[0],
                high: base + bar[1],
                low: base + bar[2],
                close: base + bar[3],
            };
        }
        pos += SHAPE_STRIDE;
    }
}
