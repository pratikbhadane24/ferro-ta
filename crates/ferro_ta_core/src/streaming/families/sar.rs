//! Parabolic SAR and SAREXT (native): O(1) state mirroring `overlap::sar` /
//! `overlap::sarext`.

use crate::streaming::registry::check_float;
use crate::streaming::StreamingError;

/// Registry entries for the parabolic SAR streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_sar {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                sar => StreamingSar, doc "Streaming parabolic SAR; first value at bar 1, initial direction rising when `high[1] >= high[0]`.",
                    params(acceleration: float = 0.02, maximum: float = 0.2), inputs(high, low), outputs(out: f64),
                    native(lookback = 2, batch = $crate::overlap::sar(high, low, acceleration, maximum));
                sarext => StreamingSarext, doc "Streaming extended parabolic SAR: signed start value, offset on reverse, separate long / short acceleration; first value at bar 1.",
                    params(startvalue: float = 0.0, offsetonreverse: float = 0.0, accelerationinitlong: float = 0.02, accelerationlong: float = 0.02, accelerationmaxlong: float = 0.2, accelerationinitshort: float = 0.02, accelerationshort: float = 0.02, accelerationmaxshort: float = 0.2),
                    inputs(high, low), outputs(out: f64),
                    native(lookback = 2, batch = $crate::overlap::sarext(high, low, startvalue, offsetonreverse, accelerationinitlong, accelerationlong, accelerationmaxlong, accelerationinitshort, accelerationshort, accelerationmaxshort));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_sar! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Acceleration schedule of one trend direction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Accel {
    init: f64,
    step: f64,
    max: f64,
}

impl Accel {
    fn validate(&self, prefix: &str) -> Result<(), StreamingError> {
        check_float(self.init, &format!("{prefix}init"))?;
        check_float(self.step, &format!("{prefix}step"))?;
        check_float(self.max, &format!("{prefix}max"))
    }
}

/// SAREXT's recurrence (SAR is the case `start = offset = 0`, one shared
/// schedule — `sarext_reduces_to_sar_bitwise` pins that in batch).
///
/// Bar 0 is only remembered; bar 1 picks the direction from
/// `high[1] >= high[0]` and seeds `ep` / `sar`; every later bar steps the
/// SAR, clamps it by the previous two lows (rising) or highs (falling), and
/// reverses when the bar crosses it.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct SarCore {
    startvalue: f64,
    offsetonreverse: f64,
    long: Accel,
    short: Accel,
    /// Committed bars (saturating).
    bars: usize,
    is_rising: bool,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    af: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    ep: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    sar: f64,
    /// `high[i - 1]`, `high[i - 2]`, `low[i - 1]`, `low[i - 2]`.
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    high_1: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    high_2: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    low_1: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    low_2: f64,
}

impl SarCore {
    fn new(startvalue: f64, offsetonreverse: f64, long: Accel, short: Accel) -> Self {
        Self {
            startvalue,
            offsetonreverse,
            long,
            short,
            bars: 0,
            is_rising: true,
            af: 0.0,
            ep: 0.0,
            sar: 0.0,
            high_1: 0.0,
            high_2: 0.0,
            low_1: 0.0,
            low_2: 0.0,
        }
    }

    fn accel(&self) -> Accel {
        if self.is_rising {
            self.long
        } else {
            self.short
        }
    }

    fn update(&mut self, high: f64, low: f64) -> f64 {
        let bar = self.bars;
        self.bars = self.bars.saturating_add(1);
        let out = match bar {
            0 => f64::NAN,
            1 => self.seed(high, low),
            _ => self.step(high, low),
        };
        self.high_2 = self.high_1;
        self.high_1 = high;
        self.low_2 = self.low_1;
        self.low_1 = low;
        out
    }

    /// Bar 1 (`high_1` / `low_1` still hold bar 0).
    fn seed(&mut self, high: f64, low: f64) -> f64 {
        self.is_rising = high >= self.high_1;
        self.af = self.accel().init;
        if self.is_rising {
            self.ep = high;
            self.sar = if self.startvalue != 0.0 {
                self.startvalue
            } else {
                self.low_1
            };
        } else {
            self.ep = low;
            self.sar = if self.startvalue != 0.0 {
                -self.startvalue
            } else {
                self.high_1
            };
        }
        self.sar
    }

    fn step(&mut self, high: f64, low: f64) -> f64 {
        let prev_sar = self.sar;
        let mut sar = prev_sar + self.af * (self.ep - prev_sar);
        if self.is_rising {
            sar = sar.min(self.low_1).min(self.low_2);
            if low < sar {
                self.is_rising = false;
                sar = self.ep + sar.abs() * self.offsetonreverse;
                self.ep = low;
                self.af = self.short.init;
            } else if high > self.ep {
                self.ep = high;
                self.af = (self.af + self.long.step).min(self.long.max);
            }
        } else {
            sar = sar.max(self.high_1).max(self.high_2);
            if high > sar {
                self.is_rising = true;
                sar = self.ep - sar.abs() * self.offsetonreverse;
                self.ep = high;
                self.af = self.long.init;
            } else if low < self.ep {
                self.ep = low;
                self.af = (self.af + self.short.step).min(self.short.max);
            }
        }
        self.sar = sar;
        sar
    }

    fn reset(&mut self) {
        *self = Self::new(self.startvalue, self.offsetonreverse, self.long, self.short);
    }

    fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.startvalue, "startvalue")?;
        check_float(self.offsetonreverse, "offsetonreverse")?;
        self.long.validate("acceleration_long_")?;
        self.short.validate("acceleration_short_")
    }
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

/// Streaming parabolic SAR (`overlap::sar`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingSar {
    core: SarCore,
}

impl StreamingSar {
    pub fn new(acceleration: f64, maximum: f64) -> Result<Self, StreamingError> {
        check_float(acceleration, "acceleration")?;
        check_float(maximum, "maximum")?;
        let accel = Accel {
            init: acceleration,
            step: acceleration,
            max: maximum,
        };
        Ok(Self {
            core: SarCore::new(0.0, 0.0, accel, accel),
        })
    }

    /// Commit one bar (NaN on bar 0).
    pub fn update(&mut self, high: f64, low: f64) -> f64 {
        self.core.update(high, low)
    }

    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// The core must still be plain SAR: one schedule, no start / offset.
    pub fn validate(&self) -> Result<(), StreamingError> {
        self.core.validate()?;
        let c = &self.core;
        let plain = c.startvalue == 0.0
            && c.offsetonreverse == 0.0
            && c.long == c.short
            && c.long.init == c.long.step;
        if !plain {
            return Err(StreamingError(
                "sar snapshot carries SAREXT-only settings".into(),
            ));
        }
        Ok(())
    }

    pub fn acceleration(&self) -> f64 {
        self.core.long.step
    }

    pub fn maximum(&self) -> f64 {
        self.core.long.max
    }
}

/// Streaming extended parabolic SAR (`overlap::sarext`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingSarext {
    core: SarCore,
}

impl StreamingSarext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        startvalue: f64,
        offsetonreverse: f64,
        accelerationinitlong: f64,
        accelerationlong: f64,
        accelerationmaxlong: f64,
        accelerationinitshort: f64,
        accelerationshort: f64,
        accelerationmaxshort: f64,
    ) -> Result<Self, StreamingError> {
        let floats = [
            (startvalue, "startvalue"),
            (offsetonreverse, "offsetonreverse"),
            (accelerationinitlong, "accelerationinitlong"),
            (accelerationlong, "accelerationlong"),
            (accelerationmaxlong, "accelerationmaxlong"),
            (accelerationinitshort, "accelerationinitshort"),
            (accelerationshort, "accelerationshort"),
            (accelerationmaxshort, "accelerationmaxshort"),
        ];
        for (value, name) in floats {
            check_float(value, name)?;
        }
        let long = Accel {
            init: accelerationinitlong,
            step: accelerationlong,
            max: accelerationmaxlong,
        };
        let short = Accel {
            init: accelerationinitshort,
            step: accelerationshort,
            max: accelerationmaxshort,
        };
        Ok(Self {
            core: SarCore::new(startvalue, offsetonreverse, long, short),
        })
    }

    /// Commit one bar (NaN on bar 0).
    pub fn update(&mut self, high: f64, low: f64) -> f64 {
        self.core.update(high, low)
    }

    pub fn reset(&mut self) {
        self.core.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.core.validate()
    }

    pub fn startvalue(&self) -> f64 {
        self.core.startvalue
    }

    pub fn offsetonreverse(&self) -> f64 {
        self.core.offsetonreverse
    }

    pub fn accelerationinitlong(&self) -> f64 {
        self.core.long.init
    }

    pub fn accelerationlong(&self) -> f64 {
        self.core.long.step
    }

    pub fn accelerationmaxlong(&self) -> f64 {
        self.core.long.max
    }

    pub fn accelerationinitshort(&self) -> f64 {
        self.core.short.init
    }

    pub fn accelerationshort(&self) -> f64 {
        self.core.short.step
    }

    pub fn accelerationmaxshort(&self) -> f64 {
        self.core.short.max
    }
}
