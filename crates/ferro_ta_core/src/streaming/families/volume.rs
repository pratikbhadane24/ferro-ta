//! Volume and volatility streams: MFI, TRANGE, NATR, OBV, AD and ADOSC.

use super::extended_common::{check_fast_below_slow, check_zero_exact, seed_sum, Ring};
use crate::streaming::primitives::Wilder;
use crate::streaming::registry::check_period;
use crate::streaming::StreamingError;
use crate::window_sums::ZeroExactSum;

/// Registry entries for the volume and volatility streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_volume {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                mfi => StreamingMfi, doc "Streaming money flow index over `timeperiod` bars of typical-price money flow. Mirrors the batch zero-exact running flow sums, so a window without money flow (e.g. all-zero volume) is 0 exactly as in the batch function.",
                    params(timeperiod: period = 14), inputs(high, low, close, volume), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::volume::mfi(high, low, close, volume, timeperiod));
                trange => StreamingTrange, doc "Streaming true range (NaN on the first bar: no previous close).",
                    params(), inputs(high, low, close), outputs(out: f64),
                    windowed(lookback = 2, batch = $crate::volatility::trange(high, low, close));
                natr => StreamingNatr, doc "Streaming normalized ATR: Wilder ATR / close * 100 (NaN where close is 0).",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::volatility::natr(high, low, close, timeperiod));
                obv => StreamingObv, doc "Streaming on-balance volume; the first bar is the first volume.",
                    params(), inputs(close, volume), outputs(out: f64),
                    native(lookback = 1, batch = $crate::volume::obv(close, volume));
                ad => StreamingAd, doc "Streaming Chaikin accumulation / distribution line.",
                    params(), inputs(high, low, close, volume), outputs(out: f64),
                    native(lookback = 1, batch = $crate::volume::ad(high, low, close, volume));
                adosc => StreamingAdosc, doc "Streaming Chaikin A/D oscillator: fast EMA - slow EMA of the A/D line, both seeded from its first value.",
                    params(fastperiod: period = 3, slowperiod: period = 10), inputs(high, low, close, volume), outputs(out: f64),
                    native(lookback = slowperiod, batch = $crate::volume::adosc(high, low, close, volume, fastperiod, slowperiod));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_volume! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Native: NATR
// ---------------------------------------------------------------------------

/// Streaming NATR (`volatility::natr`): `atr` (SMA seed of `TR[1..=p]`, then
/// Wilder) divided by the same bar's close.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingNatr {
    atr: Wilder,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_close: f64,
    has_prev: bool,
}

impl StreamingNatr {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            atr: Wilder::new(timeperiod)?,
            prev_close: f64::NAN,
            has_prev: false,
        })
    }

    /// Commit one bar (NaN for the first `timeperiod` bars).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let prev_close = std::mem::replace(&mut self.prev_close, close);
        if !std::mem::replace(&mut self.has_prev, true) {
            return f64::NAN;
        }
        let atr = self
            .atr
            .update(crate::volatility::true_range(high, low, prev_close));
        if atr.is_nan() || close == 0.0 {
            f64::NAN
        } else {
            atr / close * 100.0
        }
    }

    pub fn reset(&mut self) {
        self.atr.reset();
        self.prev_close = f64::NAN;
        self.has_prev = false;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.atr.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.atr.period()
    }
}

// ---------------------------------------------------------------------------
// Native: OBV
// ---------------------------------------------------------------------------

/// Streaming OBV (`volume::obv`): bar 0 is `volume[0]`, then `+volume` on a
/// higher close, `-volume` on a lower one and `+0.0` otherwise (NaN included).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingObv {
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    total: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_close: f64,
    has_prev: bool,
}

impl StreamingObv {
    pub fn new() -> Result<Self, StreamingError> {
        Ok(Self {
            total: 0.0,
            prev_close: f64::NAN,
            has_prev: false,
        })
    }

    /// Commit one bar; never NaN for finite input.
    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        let prev = std::mem::replace(&mut self.prev_close, close);
        if !std::mem::replace(&mut self.has_prev, true) {
            self.total = volume;
            return self.total;
        }
        let delta = if close > prev {
            volume
        } else if close < prev {
            -volume
        } else {
            0.0
        };
        self.total += delta;
        self.total
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
        self.prev_close = f64::NAN;
        self.has_prev = false;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Native: AD
// ---------------------------------------------------------------------------

/// Close-location value times volume (`volume::ad`'s per-bar term).
fn money_flow_volume(high: f64, low: f64, close: f64, volume: f64) -> f64 {
    let range = high - low;
    let clv = if range != 0.0 {
        ((close - low) - (high - close)) / range
    } else {
        0.0
    };
    clv * volume
}

/// Streaming Chaikin A/D line (`volume::ad`): a running sum from 0.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingAd {
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    total: f64,
}

impl StreamingAd {
    pub fn new() -> Result<Self, StreamingError> {
        Ok(Self::default())
    }

    /// Commit one bar; never NaN for finite input.
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        self.total += money_flow_volume(high, low, close, volume);
        self.total
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Native: ADOSC
// ---------------------------------------------------------------------------

/// Streaming ADOSC (`volume::adosc`): both EMAs are seeded with the first A/D
/// value (no SMA seed) and step as `k * a + (1 - k) * prev`; the first output
/// is at bar `slowperiod - 1`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingAdosc {
    fastperiod: usize,
    slowperiod: usize,
    /// Committed bars (saturating).
    bars: usize,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    ad: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    fast: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    slow: f64,
}

impl StreamingAdosc {
    pub fn new(fastperiod: usize, slowperiod: usize) -> Result<Self, StreamingError> {
        check_period(fastperiod, "fastperiod", 1)?;
        check_period(slowperiod, "slowperiod", 1)?;
        check_fast_below_slow(fastperiod, slowperiod)?;
        Ok(Self {
            fastperiod,
            slowperiod,
            bars: 0,
            ad: 0.0,
            fast: f64::NAN,
            slow: f64::NAN,
        })
    }

    /// Commit one bar (NaN for the first `slowperiod - 1` bars).
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let bar = self.bars;
        self.bars = self.bars.saturating_add(1);
        self.ad += money_flow_volume(high, low, close, volume);
        if bar == 0 {
            self.fast = self.ad;
            self.slow = self.ad;
        } else {
            let fast_k = 2.0 / (self.fastperiod as f64 + 1.0);
            let slow_k = 2.0 / (self.slowperiod as f64 + 1.0);
            self.fast = fast_k * self.ad + (1.0 - fast_k) * self.fast;
            self.slow = slow_k * self.ad + (1.0 - slow_k) * self.slow;
        }
        if bar < self.slowperiod - 1 {
            f64::NAN
        } else {
            self.fast - self.slow
        }
    }

    pub fn reset(&mut self) {
        self.bars = 0;
        self.ad = 0.0;
        self.fast = f64::NAN;
        self.slow = f64::NAN;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.fastperiod, "fastperiod", 1)?;
        check_period(self.slowperiod, "slowperiod", 1)?;
        check_fast_below_slow(self.fastperiod, self.slowperiod)
    }

    pub fn fastperiod(&self) -> usize {
        self.fastperiod
    }

    pub fn slowperiod(&self) -> usize {
        self.slowperiod
    }
}

// ---------------------------------------------------------------------------
// Native: MFI
// ---------------------------------------------------------------------------

/// Streaming MFI (`volume::mfi`): positive / negative typical-price money
/// flow from bar 1, sequential seed sums over flows `1..=p`, then zero-exact
/// slides (`ZeroExactSum`: `sum += new - old`, read as exactly 0 while the
/// window holds no flow of that sign). Native, not windowed, so the running sums —
/// and their last ulps — are the batch function's own.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMfi {
    timeperiod: usize,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    tp_prev: f64,
    has_prev: bool,
    pos_flows: Ring,
    neg_flows: Ring,
    pos_sum: ZeroExactSum,
    neg_sum: ZeroExactSum,
}

impl StreamingMfi {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            tp_prev: f64::NAN,
            has_prev: false,
            pos_flows: Ring::default(),
            neg_flows: Ring::default(),
            pos_sum: ZeroExactSum::default(),
            neg_sum: ZeroExactSum::default(),
        })
    }

    /// Commit one bar (NaN for the first `timeperiod` bars).
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let tp = (high + low + close) / 3.0;
        let tp_prev = std::mem::replace(&mut self.tp_prev, tp);
        if !std::mem::replace(&mut self.has_prev, true) {
            return f64::NAN;
        }
        let rmf = tp * volume;
        let (pos, neg) = if tp > tp_prev {
            (rmf, 0.0)
        } else if tp < tp_prev {
            (0.0, rmf)
        } else {
            (0.0, 0.0)
        };
        let p = self.timeperiod;
        let old_pos = self.pos_flows.push(pos, p);
        let old_neg = self.neg_flows.push(neg, p);
        match (old_pos, old_neg) {
            (Some(op), Some(on)) => {
                self.pos_sum.slide(pos, op);
                self.neg_sum.slide(neg, on);
            }
            _ if self.pos_flows.len() == p => {
                self.pos_sum = seed_sum(&self.pos_flows);
                self.neg_sum = seed_sum(&self.neg_flows);
            }
            _ => return f64::NAN,
        }
        let (pos_sum, neg_sum) = (self.pos_sum.value(), self.neg_sum.value());
        let denom = pos_sum + neg_sum;
        if denom == 0.0 {
            0.0
        } else {
            (100.0 * pos_sum / denom).clamp(0.0, 100.0)
        }
    }

    pub fn reset(&mut self) {
        self.tp_prev = f64::NAN;
        self.has_prev = false;
        self.pos_flows.clear();
        self.neg_flows.clear();
        self.pos_sum = ZeroExactSum::default();
        self.neg_sum = ZeroExactSum::default();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 1)?;
        self.pos_flows
            .validate(self.timeperiod, "mfi positive flows")?;
        self.neg_flows
            .validate(self.timeperiod, "mfi negative flows")?;
        if self.pos_flows.len() != self.neg_flows.len() {
            return Err(StreamingError(
                "mfi flow windows disagree on their length".to_string(),
            ));
        }
        check_zero_exact(
            &self.pos_sum,
            &self.pos_flows,
            self.timeperiod,
            "mfi positive flow",
        )?;
        check_zero_exact(
            &self.neg_sum,
            &self.neg_flows,
            self.timeperiod,
            "mfi negative flow",
        )
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }
}
