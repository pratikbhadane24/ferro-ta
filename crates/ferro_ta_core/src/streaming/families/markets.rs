//! Regime, portfolio, futures and utility streams.
//!
//! Stateless and fixed-window kernels are windowed replays; the four that
//! carry running state (CUSUM, relative strength, funding PnL, RSI
//! thresholds) are native structs below.

use std::collections::VecDeque;

use super::momentum::{rsi_value, GainLoss};
use crate::streaming::registry::{check_float, check_period};
use crate::streaming::StreamingError;

/// Registry entries for the regime, portfolio, futures and utility streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_markets {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                regime_adx => StreamingRegimeAdx, doc "Streaming trend (1) / range (0) label from ADX against `threshold` (-1 where ADX is NaN).",
                    params(threshold: float = 25.0), inputs(adx), outputs(out: i8),
                    windowed(lookback = 1, batch = $crate::regime::regime_adx(adx, threshold));
                regime_combined => StreamingRegimeCombined, doc "Streaming trend (1) / range (0) label: ADX above `adx_threshold` and ATR / close above `atr_pct_threshold`.",
                    params(adx_threshold: float = 25.0, atr_pct_threshold: float = 0.005), inputs(adx, atr, close), outputs(out: i8),
                    windowed(lookback = 1, batch = $crate::regime::regime_combined(adx, atr, close, adx_threshold, atr_pct_threshold));
                detect_breaks_cusum => StreamingDetectBreaksCusum, doc "Streaming CUSUM structural-break flag (1 at a break bar) against the previous `window` bars.",
                    params(window: period(2) = 20, threshold: float = 3.0, slack: float = 0.5), inputs(series), outputs(out: i8),
                    native(lookback = window + 1, batch = $crate::regime::detect_breaks_cusum(series, window, threshold, slack));
                rolling_variance_break => StreamingRollingVarianceBreak, doc "Streaming volatility-regime break flag: short / long window variance ratio above `threshold` (current bar excluded).",
                    params(short_window: period(2) = 10, long_window: period = 50, threshold: float = 2.0), inputs(series), outputs(out: i8),
                    windowed(lookback = long_window + 1, batch = $crate::regime::rolling_variance_break(series, short_window, long_window, threshold), requires = long_window > short_window);
                rolling_beta => StreamingRollingBeta, doc "Streaming rolling beta of `asset` against `benchmark` over `window` bars.",
                    params(window: period(2) = 20), inputs(asset, benchmark), outputs(out: f64),
                    windowed(lookback = window, batch = $crate::portfolio::rolling_beta(asset, benchmark, window));
                relative_strength => StreamingRelativeStrength, doc "Streaming relative strength: cumulative product of (1 + asset return) over that of the benchmark.",
                    params(), inputs(asset_returns, benchmark_returns), outputs(out: f64),
                    native(lookback = 1, batch = $crate::portfolio::relative_strength(asset_returns, benchmark_returns));
                spread => StreamingSpread, doc "Streaming spread a - hedge * b.",
                    params(hedge: float = 1.0), inputs(a, b), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::portfolio::spread(a, b, hedge));
                ratio => StreamingRatio, doc "Streaming ratio a / b (NaN where b is 0).",
                    params(), inputs(a, b), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::portfolio::ratio(a, b));
                zscore_series => StreamingZscoreSeries, doc "Streaming rolling z-score of `x` over `window` bars.",
                    params(window: period(2) = 20), inputs(x), outputs(out: f64),
                    windowed(lookback = window, batch = $crate::portfolio::zscore_series(x, window));
                weighted_continuous => StreamingWeightedContinuous, doc "Streaming weighted contract stitching: front * (1 - w) + next * w.",
                    params(), inputs(front, next, next_weights), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::futures::roll::weighted_continuous(front, next, next_weights));
                check_cross => StreamingCheckCross, doc "Streaming cross event between `fast` and `slow`: 1 cross-over, -1 cross-under, 0 none.",
                    params(), inputs(fast, slow), outputs(out: i8),
                    windowed(lookback = 2, batch = $crate::alerts::check_cross(fast, slow));
                funding_cumulative_pnl => StreamingFundingCumulativePnl, doc "Streaming cumulative funding PnL: running sum of -position_size * funding_rate.",
                    params(), inputs(position_size, funding_rate), outputs(out: f64),
                    native(lookback = 1, batch = $crate::crypto::funding_cumulative_pnl(position_size, funding_rate));
                rsi_threshold_signals => StreamingRsiThresholdSignals, doc "Streaming RSI threshold signal: +1 at or below `oversold`, -1 at or above `overbought`, else 0 (NaN during warm-up).",
                    params(timeperiod: period = 14, oversold: float = 30.0, overbought: float = 70.0), inputs(close), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::backtest::rsi_threshold_signals(close, timeperiod, oversold, overbought));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_markets! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Native: CUSUM breaks
// ---------------------------------------------------------------------------

/// Streaming CUSUM breaks (`regime::detect_breaks_cusum`).
///
/// Each bar is scored against the `window` bars *before* it (its own value is
/// excluded from the mean and sample deviation). A bar with no full previous
/// window, a zero / NaN deviation or a NaN value is skipped: it flags 0 and
/// leaves both sums untouched. A break resets both sums.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingDetectBreaksCusum {
    window: usize,
    threshold: f64,
    slack: f64,
    #[cfg_attr(feature = "serde", serde(with = "crate::streaming::snapshot::nan_seq"))]
    previous: VecDeque<f64>,
    cusum_pos: f64,
    cusum_neg: f64,
}

impl StreamingDetectBreaksCusum {
    pub fn new(window: usize, threshold: f64, slack: f64) -> Result<Self, StreamingError> {
        let stream = Self {
            window,
            threshold,
            slack,
            previous: VecDeque::new(),
            cusum_pos: 0.0,
            cusum_neg: 0.0,
        };
        stream.validate()?;
        Ok(stream)
    }

    /// z-score of `x` against the previous window, `None` when skipped.
    fn z_score(&self, x: f64) -> Option<f64> {
        if self.previous.len() < self.window {
            return None;
        }
        let mean = self.previous.iter().sum::<f64>() / self.window as f64;
        let var = self
            .previous
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>()
            / (self.window - 1) as f64;
        let std = var.sqrt();
        (std != 0.0 && !std.is_nan() && !x.is_nan()).then(|| (x - mean) / std)
    }

    /// Commit one value; `1` at a break bar.
    pub fn update(&mut self, series: f64) -> i8 {
        let mut flag = 0;
        if let Some(z) = self.z_score(series) {
            self.cusum_pos = (self.cusum_pos + z - self.slack).max(0.0);
            self.cusum_neg = (self.cusum_neg - z - self.slack).max(0.0);
            if self.cusum_pos > self.threshold || self.cusum_neg > self.threshold {
                flag = 1;
                self.cusum_pos = 0.0;
                self.cusum_neg = 0.0;
            }
        }
        self.previous.push_back(series);
        if self.previous.len() > self.window {
            self.previous.pop_front();
        }
        flag
    }

    pub fn reset(&mut self) {
        self.previous.clear();
        self.cusum_pos = 0.0;
        self.cusum_neg = 0.0;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.window, "window", 2)?;
        check_float(self.threshold, "threshold")?;
        check_float(self.slack, "slack")?;
        let sums_ok = |s: f64| s.is_finite() && s >= 0.0;
        if self.previous.len() > self.window || !sums_ok(self.cusum_pos) || !sums_ok(self.cusum_neg)
        {
            return Err(StreamingError(
                "detect_breaks_cusum: malformed state".to_string(),
            ));
        }
        Ok(())
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn slack(&self) -> f64 {
        self.slack
    }
}

// ---------------------------------------------------------------------------
// Native: relative strength
// ---------------------------------------------------------------------------

/// Streaming relative strength (`portfolio::relative_strength`): the running
/// products of `1 + return` for asset and benchmark, and their ratio.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingRelativeStrength {
    cum_asset: f64,
    cum_benchmark: f64,
}

impl StreamingRelativeStrength {
    pub fn new() -> Result<Self, StreamingError> {
        Ok(Self {
            cum_asset: 1.0,
            cum_benchmark: 1.0,
        })
    }

    /// Commit one bar of returns (NaN where the benchmark product is 0).
    pub fn update(&mut self, asset_returns: f64, benchmark_returns: f64) -> f64 {
        self.cum_asset *= 1.0 + asset_returns;
        self.cum_benchmark *= 1.0 + benchmark_returns;
        if self.cum_benchmark == 0.0 {
            f64::NAN
        } else {
            self.cum_asset / self.cum_benchmark
        }
    }

    pub fn reset(&mut self) {
        self.cum_asset = 1.0;
        self.cum_benchmark = 1.0;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.cum_asset, "cum_asset")?;
        check_float(self.cum_benchmark, "cum_benchmark")
    }
}

// ---------------------------------------------------------------------------
// Native: funding PnL
// ---------------------------------------------------------------------------

/// Streaming cumulative funding PnL (`crypto::funding_cumulative_pnl`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingFundingCumulativePnl {
    total: f64,
}

impl StreamingFundingCumulativePnl {
    pub fn new() -> Result<Self, StreamingError> {
        Ok(Self { total: 0.0 })
    }

    /// Commit one funding payment and return the running total.
    pub fn update(&mut self, position_size: f64, funding_rate: f64) -> f64 {
        self.total += -position_size * funding_rate;
        self.total
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.total, "total")
    }
}

// ---------------------------------------------------------------------------
// Native: RSI threshold signals
// ---------------------------------------------------------------------------

/// Streaming RSI threshold signal (`backtest::rsi_threshold_signals`): the
/// batch RSI mapped to +1 / -1 / 0; NaN until the RSI is seeded.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingRsiThresholdSignals {
    averages: GainLoss,
    oversold: f64,
    overbought: f64,
}

impl StreamingRsiThresholdSignals {
    pub fn new(timeperiod: usize, oversold: f64, overbought: f64) -> Result<Self, StreamingError> {
        check_float(oversold, "oversold")?;
        check_float(overbought, "overbought")?;
        Ok(Self {
            averages: GainLoss::new(timeperiod)?,
            oversold,
            overbought,
        })
    }

    /// Commit one bar (NaN for the first `timeperiod` bars).
    pub fn update(&mut self, close: f64) -> f64 {
        let Some((gain, loss)) = self.averages.update(close) else {
            return f64::NAN;
        };
        let rsi = rsi_value(gain, loss);
        if rsi <= self.oversold {
            1.0
        } else if rsi >= self.overbought {
            -1.0
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.averages.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.oversold, "oversold")?;
        check_float(self.overbought, "overbought")?;
        self.averages.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.averages.period()
    }

    pub fn oversold(&self) -> f64 {
        self.oversold
    }

    pub fn overbought(&self) -> f64 {
        self.overbought
    }
}
