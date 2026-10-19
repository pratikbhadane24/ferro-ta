//! Extended moving averages: Hull and ALMA (windowed); Alligator, Gator,
//! ZLEMA, FRAMA, McGinley and VIDYA (native).

use super::extended_common::{finite2, Ring};
use super::StreamingCmo;
use crate::streaming::primitives::{EmaFirstFinite, Smma};
use crate::streaming::registry::{check_count, check_period};
use crate::streaming::StreamingError;

/// Registry entries for the extended moving averages.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_extended_trend {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                hull_ma => StreamingHullMa, doc "Streaming Hull moving average: WMA(2 * WMA(n/2) - WMA(n), sqrt(n)).",
                    params(timeperiod: period = 16), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod + ((timeperiod as f64).sqrt() as usize).max(1) - 1, batch = $crate::extended::hull_ma(close, timeperiod));
                alma => StreamingAlma, doc "Streaming Arnaud Legoux moving average: Gaussian-weighted mean of the last `timeperiod` closes.",
                    params(timeperiod: period = 21, offset: float = 0.85, sigma: float = 6.0), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::extended::alma(close, timeperiod, offset, sigma), requires = sigma > 0.0);
                alligator => StreamingAlligator, doc "Streaming Bill Williams Alligator: (jaw, teeth, lips) SMMAs of the median price, each shifted forward by its shift.",
                    params(jaw_period: period = 13, jaw_shift: count = 8, teeth_period: period = 8, teeth_shift: count = 5, lips_period: period = 5, lips_shift: count = 3),
                    inputs(high, low), outputs(out_jaw: f64, out_teeth: f64, out_lips: f64),
                    native(lookback = (jaw_period + jaw_shift).max(teeth_period + teeth_shift).max(lips_period + lips_shift),
                        batch = $crate::extended::alligator(high, low, jaw_period, jaw_shift, teeth_period, teeth_shift, lips_period, lips_shift));
                gator => StreamingGator, doc "Streaming Gator oscillator: (abs(jaw - teeth), -abs(teeth - lips)) of the Alligator lines.",
                    params(jaw_period: period = 13, jaw_shift: count = 8, teeth_period: period = 8, teeth_shift: count = 5, lips_period: period = 5, lips_shift: count = 3),
                    inputs(high, low), outputs(out_upper: f64, out_lower: f64),
                    native(lookback = (jaw_period + jaw_shift).max(teeth_period + teeth_shift).max(lips_period + lips_shift),
                        batch = $crate::extended::gator(high, low, jaw_period, jaw_shift, teeth_period, teeth_shift, lips_period, lips_shift));
                zlema => StreamingZlema, doc "Streaming zero-lag EMA: EMA of close + (close - close `(timeperiod - 1) / 2` bars ago), seeded on the first full finite window.",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    native(lookback = (timeperiod - 1) / 2 + timeperiod, batch = $crate::extended::zlema(close, timeperiod));
                frama => StreamingFrama, doc "Streaming Ehlers fractal adaptive moving average, seeded with the close at bar `timeperiod - 1`.",
                    params(timeperiod: period(2) = 16), inputs(close), outputs(out: f64),
                    native(lookback = timeperiod, batch = $crate::extended::frama(close, timeperiod));
                mcginley => StreamingMcginley, doc "Streaming McGinley dynamic, seeded with the SMA of the first `timeperiod` closes.",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    native(lookback = timeperiod, batch = $crate::extended::mcginley(close, timeperiod));
                vidya => StreamingVidya, doc "Streaming Chande VIDYA: an EMA whose factor is scaled by abs(CMO(`cmo_period`)) / 100, seeded with the close on the first CMO bar.",
                    params(timeperiod: period = 14, cmo_period: period = 9), inputs(close), outputs(out: f64),
                    native(lookback = cmo_period + 1, batch = $crate::extended::vidya(close, timeperiod, cmo_period));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_extended_trend! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Native: ALLIGATOR / GATOR
// ---------------------------------------------------------------------------

/// One Alligator line: SMMA of the median price, read `shift` bars back.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct ShiftedSmma {
    smma: Smma,
    shift: usize,
    /// The last `shift + 1` SMMA values.
    history: Ring,
}

impl ShiftedSmma {
    fn new(
        period: usize,
        shift: usize,
        period_name: &str,
        shift_name: &str,
    ) -> Result<Self, StreamingError> {
        check_period(period, period_name, 1)?;
        check_count(shift, shift_name)?;
        Ok(Self {
            smma: Smma::new(period)?,
            shift,
            history: Ring::default(),
        })
    }

    fn update(&mut self, median: f64) -> f64 {
        let v = self.smma.update(median);
        self.history.push(v, self.shift + 1);
        self.history.ago(self.shift)
    }

    fn reset(&mut self) {
        self.smma.reset();
        self.history.clear();
    }

    fn validate(&self, shift_name: &str) -> Result<(), StreamingError> {
        check_count(self.shift, shift_name)?;
        self.smma.validate()?;
        self.history
            .validate(self.shift + 1, "alligator line history")
    }
}

/// Jaw, teeth and lips of `extended::alligator`: SMMA (`(prev * (p - 1) +
/// x) / p` after an SMA seed) of `(high + low) * 0.5`, shifted forward.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct AlligatorLines {
    jaw: ShiftedSmma,
    teeth: ShiftedSmma,
    lips: ShiftedSmma,
}

impl AlligatorLines {
    fn new(
        jaw: (usize, usize),
        teeth: (usize, usize),
        lips: (usize, usize),
    ) -> Result<Self, StreamingError> {
        Ok(Self {
            jaw: ShiftedSmma::new(jaw.0, jaw.1, "jaw_period", "jaw_shift")?,
            teeth: ShiftedSmma::new(teeth.0, teeth.1, "teeth_period", "teeth_shift")?,
            lips: ShiftedSmma::new(lips.0, lips.1, "lips_period", "lips_shift")?,
        })
    }

    fn update(&mut self, high: f64, low: f64) -> (f64, f64, f64) {
        let median = (high + low) * 0.5;
        (
            self.jaw.update(median),
            self.teeth.update(median),
            self.lips.update(median),
        )
    }

    fn reset(&mut self) {
        self.jaw.reset();
        self.teeth.reset();
        self.lips.reset();
    }

    fn validate(&self) -> Result<(), StreamingError> {
        self.jaw.validate("jaw_shift")?;
        self.teeth.validate("teeth_shift")?;
        self.lips.validate("lips_shift")
    }
}

/// Param getters shared by the Alligator and Gator streams.
macro_rules! alligator_getters {
    ($Ty:ident) => {
        impl $Ty {
            pub fn jaw_period(&self) -> usize {
                self.lines.jaw.smma.period()
            }
            pub fn jaw_shift(&self) -> usize {
                self.lines.jaw.shift
            }
            pub fn teeth_period(&self) -> usize {
                self.lines.teeth.smma.period()
            }
            pub fn teeth_shift(&self) -> usize {
                self.lines.teeth.shift
            }
            pub fn lips_period(&self) -> usize {
                self.lines.lips.smma.period()
            }
            pub fn lips_shift(&self) -> usize {
                self.lines.lips.shift
            }
            pub fn reset(&mut self) {
                self.lines.reset();
            }
            pub fn validate(&self) -> Result<(), StreamingError> {
                self.lines.validate()
            }
        }
    };
}

/// Streaming Alligator (`extended::alligator`): `(jaw, teeth, lips)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingAlligator {
    lines: AlligatorLines,
}

impl StreamingAlligator {
    pub fn new(
        jaw_period: usize,
        jaw_shift: usize,
        teeth_period: usize,
        teeth_shift: usize,
        lips_period: usize,
        lips_shift: usize,
    ) -> Result<Self, StreamingError> {
        Ok(Self {
            lines: AlligatorLines::new(
                (jaw_period, jaw_shift),
                (teeth_period, teeth_shift),
                (lips_period, lips_shift),
            )?,
        })
    }

    /// Commit one bar: `(jaw, teeth, lips)`.
    pub fn update(&mut self, high: f64, low: f64) -> (f64, f64, f64) {
        self.lines.update(high, low)
    }
}

alligator_getters!(StreamingAlligator);

/// Streaming Gator oscillator (`extended::gator`): `(|jaw - teeth|,
/// -|teeth - lips|)`, each NaN unless both lines are finite.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingGator {
    lines: AlligatorLines,
}

impl StreamingGator {
    pub fn new(
        jaw_period: usize,
        jaw_shift: usize,
        teeth_period: usize,
        teeth_shift: usize,
        lips_period: usize,
        lips_shift: usize,
    ) -> Result<Self, StreamingError> {
        Ok(Self {
            lines: AlligatorLines::new(
                (jaw_period, jaw_shift),
                (teeth_period, teeth_shift),
                (lips_period, lips_shift),
            )?,
        })
    }

    /// Commit one bar: `(upper, lower)`.
    pub fn update(&mut self, high: f64, low: f64) -> (f64, f64) {
        let (j, t, l) = self.lines.update(high, low);
        (
            finite2(j, t, |j, t| (j - t).abs()),
            finite2(t, l, |t, l| -(t - l).abs()),
        )
    }
}

alligator_getters!(StreamingGator);

// ---------------------------------------------------------------------------
// Native: ZLEMA
// ---------------------------------------------------------------------------

/// Streaming ZLEMA (`extended::zlema`): `close + (close - close[lag])` with
/// `lag = (timeperiod - 1) / 2` (NaN for the first `lag` bars), then
/// `ema_from_first_finite`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingZlema {
    ema: EmaFirstFinite,
    /// The last `lag + 1` closes.
    closes: Ring,
}

impl StreamingZlema {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            ema: EmaFirstFinite::new(timeperiod)?,
            closes: Ring::default(),
        })
    }

    fn lag(&self) -> usize {
        self.ema.period().saturating_sub(1) / 2
    }

    /// Commit one bar.
    pub fn update(&mut self, close: f64) -> f64 {
        let lag = self.lag();
        self.closes.push(close, lag + 1);
        let adjusted = if self.closes.len() > lag {
            close + (close - self.closes.front())
        } else {
            f64::NAN
        };
        self.ema.update(adjusted)
    }

    pub fn reset(&mut self) {
        self.ema.reset();
        self.closes.clear();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.ema.validate()?;
        self.closes.validate(self.lag() + 1, "zlema lag window")
    }

    pub fn timeperiod(&self) -> usize {
        self.ema.period()
    }
}

// ---------------------------------------------------------------------------
// Native: FRAMA
// ---------------------------------------------------------------------------

/// Streaming FRAMA (`extended::frama`): fractal dimension from the ranges of
/// the two `timeperiod / 2` halves and the full window, `alpha =
/// exp(-4.6 * (dim - 1))` clipped to `[0.01, 1]` (1 when a range is 0),
/// seeded with the close at bar `timeperiod - 1`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingFrama {
    timeperiod: usize,
    closes: Ring,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    value: f64,
    seeded: bool,
}

impl StreamingFrama {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 2)?;
        Ok(Self {
            timeperiod,
            closes: Ring::default(),
            value: f64::NAN,
            seeded: false,
        })
    }

    /// `alpha` of the current full window.
    fn alpha(&self) -> f64 {
        let half = (self.timeperiod / 2).max(1);
        let half_f = half as f64;
        let n1 = (self.closes.max_last(half) - self.closes.min_last(half)) / half_f;
        // The previous half: the `half` closes ending `half` bars ago.
        let older: Vec<f64> = self
            .closes
            .iter()
            .rev()
            .skip(half)
            .take(half)
            .copied()
            .collect();
        let hh = older.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let ll = older.iter().fold(f64::INFINITY, |m, &v| m.min(v));
        let n2 = (hh - ll) / half_f;
        let n3 = (self.closes.max() - self.closes.min()) / self.timeperiod as f64;
        if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
            let dim = ((n1 + n2).ln() - n3.ln()) / std::f64::consts::LN_2;
            (-4.6 * (dim - 1.0)).exp().clamp(0.01, 1.0)
        } else {
            1.0
        }
    }

    /// Commit one bar (NaN for the first `timeperiod - 1` bars).
    pub fn update(&mut self, close: f64) -> f64 {
        self.closes.push(close, self.timeperiod);
        if self.closes.len() < self.timeperiod {
            return f64::NAN;
        }
        let alpha = self.alpha();
        self.value = if std::mem::replace(&mut self.seeded, true) {
            alpha * close + (1.0 - alpha) * self.value
        } else {
            close
        };
        self.value
    }

    pub fn reset(&mut self) {
        self.closes.clear();
        self.value = f64::NAN;
        self.seeded = false;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 2)?;
        self.closes.validate(self.timeperiod, "frama window")
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }
}

// ---------------------------------------------------------------------------
// Native: MCGINLEY
// ---------------------------------------------------------------------------

/// Streaming McGinley dynamic (`extended::mcginley`): SMA seed of the first
/// `timeperiod` closes, then `md + (x - md) / (p * (x / md)^4)` (falling back
/// to `x` on a degenerate step).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMcginley {
    timeperiod: usize,
    /// Closes summed into the seed so far (`timeperiod` once seeded).
    seeded: usize,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    seed_sum: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    md: f64,
}

impl StreamingMcginley {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            seeded: 0,
            seed_sum: 0.0,
            md: f64::NAN,
        })
    }

    fn step(md: f64, price: f64, period: f64) -> f64 {
        if !md.is_finite() || md == 0.0 || !price.is_finite() {
            return price;
        }
        let denom = period * (price / md).powi(4);
        if !denom.is_finite() || denom == 0.0 {
            return price;
        }
        md + (price - md) / denom
    }

    /// Commit one bar (NaN for the first `timeperiod - 1` bars).
    pub fn update(&mut self, close: f64) -> f64 {
        let p = self.timeperiod as f64;
        if self.seeded < self.timeperiod {
            self.seed_sum += close;
            self.seeded += 1;
            if self.seeded == self.timeperiod {
                self.md = self.seed_sum / p;
            }
            return self.md;
        }
        self.md = Self::step(self.md, close, p);
        self.md
    }

    pub fn reset(&mut self) {
        self.seeded = 0;
        self.seed_sum = 0.0;
        self.md = f64::NAN;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 1)?;
        if self.seeded > self.timeperiod {
            return Err(StreamingError(format!(
                "mcginley seed holds {} closes but its period is {}",
                self.seeded, self.timeperiod
            )));
        }
        Ok(())
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }
}

// ---------------------------------------------------------------------------
// Native: VIDYA
// ---------------------------------------------------------------------------

/// Streaming VIDYA (`extended::vidya`): on every bar with a finite
/// `momentum::cmo`, `sc = k * |cmo| / 100` and `v = sc * x + (1 - sc) * v`,
/// seeded with the close on the first such bar; NaN on other bars.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingVidya {
    timeperiod: usize,
    cmo: StreamingCmo,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev: f64,
}

impl StreamingVidya {
    pub fn new(timeperiod: usize, cmo_period: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            cmo: StreamingCmo::new(cmo_period)?,
            prev: f64::NAN,
        })
    }

    /// Commit one bar.
    pub fn update(&mut self, close: f64) -> f64 {
        let c = self.cmo.update(close);
        if !c.is_finite() {
            return f64::NAN;
        }
        let k = 2.0 / (self.timeperiod as f64 + 1.0);
        let sc = k * (c.abs() / 100.0);
        self.prev = if self.prev.is_finite() {
            sc * close + (1.0 - sc) * self.prev
        } else {
            close
        };
        self.prev
    }

    pub fn reset(&mut self) {
        self.cmo.reset();
        self.prev = f64::NAN;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 1)?;
        self.cmo.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }

    pub fn cmo_period(&self) -> usize {
        self.cmo.timeperiod()
    }
}
