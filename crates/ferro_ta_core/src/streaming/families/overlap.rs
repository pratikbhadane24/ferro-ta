//! Overlap studies: DEMA, TEMA, T3, TRIMA, WMA, MIDPOINT, MIDPRICE, KAMA,
//! MA and MAVP.

use crate::overlap::ma_lookback;
use std::collections::VecDeque;

use crate::streaming::primitives::{window_ma, KamaCore, MaStream, T3};
use crate::streaming::registry::{check_matype, check_period};
use crate::streaming::StreamingError;

/// Registry entries for the `overlap` moving averages.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_overlap {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                dema => StreamingDema, doc "Streaming double exponential moving average: 2*EMA - EMA(EMA).",
                    params(timeperiod: period = 30), inputs(close), outputs(out: f64),
                    native(lookback = 2 * (timeperiod - 1) + 1, batch = $crate::overlap::dema(close, timeperiod));
                tema => StreamingTema, doc "Streaming triple exponential moving average: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)).",
                    params(timeperiod: period = 30), inputs(close), outputs(out: f64),
                    native(lookback = 3 * (timeperiod - 1) + 1, batch = $crate::overlap::tema(close, timeperiod));
                t3 => StreamingT3, doc "Streaming Tillson T3: six cascaded EMAs combined with volume factor `vfactor`.",
                    params(timeperiod: period = 5, vfactor: float = 0.7), inputs(close), outputs(out: f64),
                    native(lookback = 6 * (timeperiod - 1) + 1, batch = $crate::overlap::t3(close, timeperiod, vfactor));
                trima => StreamingTrima, doc "Streaming triangular moving average over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::overlap::trima(close, timeperiod));
                wma => StreamingWma, doc "Streaming linearly weighted moving average over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::overlap::wma(close, timeperiod));
                midpoint => StreamingMidpoint, doc "Streaming midpoint: (max(close) + min(close)) / 2 over `timeperiod` bars.",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::overlap::midpoint(close, timeperiod));
                midprice => StreamingMidprice, doc "Streaming midprice: (highest high + lowest low) / 2 over `timeperiod` bars.",
                    params(timeperiod: period = 14), inputs(high, low), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::overlap::midprice(high, low, timeperiod));
                kama => StreamingKama, doc "Streaming Kaufman adaptive moving average (TA-Lib); first value at bar `timeperiod`.",
                    params(timeperiod: period = 30), inputs(close), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::overlap::kama(close, timeperiod));
                ma => StreamingMa, doc "Streaming moving average of type `matype` (0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7/8=T3).",
                    params(timeperiod: period = 30, matype: matype = 0), inputs(close), outputs(out: f64),
                    native(lookback = $crate::overlap::ma_lookback(timeperiod, matype) + 1, batch = $crate::overlap::ma(close, timeperiod, matype));
                mavp => StreamingMavp, doc "Streaming moving average with a per-bar period: `periods` is rounded and clamped to [minperiod, maxperiod].",
                    params(minperiod: period = 2, maxperiod: period = 30, matype: matype = 0), inputs(close, periods), outputs(out: f64),
                    native(lookback = $crate::overlap::ma_lookback(maxperiod, matype) + 1, batch = $crate::overlap::mavp(close, periods, minperiod, maxperiod, matype));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_overlap! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Native: single moving averages
// ---------------------------------------------------------------------------

/// A stream that is one [`MaStream`] of a fixed `matype` over `close`.
macro_rules! fixed_ma_stream {
    ($(#[$doc:meta])* $Ty:ident, $matype:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $Ty {
            inner: MaStream,
        }

        impl $Ty {
            pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
                Ok(Self {
                    inner: MaStream::new(timeperiod, $matype)?,
                })
            }

            /// Commit one bar (NaN during warm-up).
            pub fn update(&mut self, close: f64) -> f64 {
                self.inner.update(close)
            }

            pub fn reset(&mut self) {
                self.inner.reset();
            }

            pub fn validate(&self) -> Result<(), StreamingError> {
                self.inner.validate_as(self.inner.period(), $matype)
            }

            pub fn timeperiod(&self) -> usize {
                self.inner.period()
            }
        }
    };
}

fixed_ma_stream!(
    /// Streaming DEMA (`overlap::dema`): two cascaded first-finite EMAs.
    StreamingDema,
    3
);
fixed_ma_stream!(
    /// Streaming TEMA (`overlap::tema`): three cascaded first-finite EMAs.
    StreamingTema,
    4
);
fixed_ma_stream!(
    /// Streaming KAMA (`overlap::kama`, TA-Lib); first value at bar `timeperiod`.
    StreamingKama,
    6
);

/// Streaming Tillson T3 (`overlap::t3`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingT3 {
    inner: T3,
}

impl StreamingT3 {
    pub fn new(timeperiod: usize, vfactor: f64) -> Result<Self, StreamingError> {
        Ok(Self {
            inner: T3::new(timeperiod, vfactor)?,
        })
    }

    /// Commit one bar (NaN for the first `6 * (timeperiod - 1)` bars).
    pub fn update(&mut self, close: f64) -> f64 {
        self.inner.update(close)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.inner.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.inner.period()
    }

    pub fn vfactor(&self) -> f64 {
        self.inner.vfactor()
    }
}

/// Streaming generic moving average (`overlap::ma`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMa {
    matype: u8,
    inner: MaStream,
}

impl StreamingMa {
    pub fn new(timeperiod: usize, matype: u8) -> Result<Self, StreamingError> {
        Ok(Self {
            matype,
            inner: MaStream::new(timeperiod, matype)?,
        })
    }

    /// Commit one bar (NaN during the `matype`'s warm-up).
    pub fn update(&mut self, close: f64) -> f64 {
        self.inner.update(close)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.inner.validate_as(self.inner.period(), self.matype)
    }

    pub fn timeperiod(&self) -> usize {
        self.inner.period()
    }

    pub fn matype(&self) -> u8 {
        self.matype
    }
}

// ---------------------------------------------------------------------------
// Native: MAVP
// ---------------------------------------------------------------------------

/// Largest `maxperiod - minperiod + 1` [`StreamingMavp`] accepts (checked at
/// construction and on snapshot restore): it keeps recursive state per period
/// in the range, so the range bounds its memory and per-bar work.
pub const MAX_MAVP_RANGE: usize = 4096;

/// Per-period state of a recursive MAVP `matype`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum MavpLeg {
    /// EMA, DEMA, TEMA, T3: O(1) state of their own.
    Ma(MaStream),
    /// KAMA: reads prices from the shared ring.
    Kama(KamaCore),
}

/// Streaming moving average with a per-bar period (`overlap::mavp`).
///
/// All periods share one price ring of `maxperiod` values (`+ 2` for
/// KAMA). Window types (SMA, WMA, TRIMA) compute the selected period from
/// the ring on demand; recursive types keep O(1) state per period in
/// `[minperiod, maxperiod]`, all updated every bar as the batch kernel runs
/// a full MA per period. `periods` is rounded, then clamped. NaN for the
/// first `ma_lookback(maxperiod, matype)` bars whatever the per-bar period.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMavp {
    minperiod: usize,
    maxperiod: usize,
    matype: u8,
    /// Latest prices, newest last; `min(bars, ring_cap)` of them.
    ring: VecDeque<f64>,
    /// Committed bars (saturating).
    bars: usize,
    /// `legs[k]` has period `minperiod + k`; empty for the window types.
    legs: Vec<MavpLeg>,
}

impl StreamingMavp {
    pub fn new(minperiod: usize, maxperiod: usize, matype: u8) -> Result<Self, StreamingError> {
        Self::check(minperiod, maxperiod, matype)?;
        let legs = if Self::is_window_type(matype) {
            Vec::new()
        } else {
            (minperiod..=maxperiod)
                .map(|p| Self::leg(p, matype))
                .collect::<Result<_, _>>()?
        };
        Ok(Self {
            minperiod,
            maxperiod,
            matype,
            ring: VecDeque::new(),
            bars: 0,
            legs,
        })
    }

    fn is_window_type(matype: u8) -> bool {
        matches!(matype, 0 | 2 | 5)
    }

    fn leg(period: usize, matype: u8) -> Result<MavpLeg, StreamingError> {
        Ok(if matype == 6 {
            MavpLeg::Kama(KamaCore::new(period)?)
        } else {
            MavpLeg::Ma(MaStream::new(period, matype)?)
        })
    }

    fn ring_cap(&self) -> usize {
        if self.matype == 6 {
            KamaCore::ring_len(self.maxperiod)
        } else {
            self.maxperiod
        }
    }

    fn check(minperiod: usize, maxperiod: usize, matype: u8) -> Result<(), StreamingError> {
        check_period(minperiod, "minperiod", 1)?;
        check_period(maxperiod, "maxperiod", 1)?;
        check_matype(matype, "matype")?;
        if maxperiod < minperiod {
            return Err(StreamingError(format!(
                "maxperiod must be >= minperiod, got {maxperiod} < {minperiod}"
            )));
        }
        let range = maxperiod - minperiod + 1;
        if range > MAX_MAVP_RANGE {
            return Err(StreamingError(format!(
                "maxperiod - minperiod + 1 must be <= {MAX_MAVP_RANGE}, got {range}"
            )));
        }
        Ok(())
    }

    /// Commit one bar; `periods` is this bar's period, rounded and clamped
    /// to `[minperiod, maxperiod]` like the batch kernel.
    pub fn update(&mut self, close: f64, periods: f64) -> f64 {
        let i = self.bars;
        self.bars = self.bars.saturating_add(1);
        self.ring.push_back(close);
        if self.ring.len() > self.ring_cap() {
            self.ring.pop_front();
        }
        let chosen = (periods.round() as usize).clamp(self.minperiod, self.maxperiod);
        let mut out = f64::NAN;
        for (period, leg) in (self.minperiod..).zip(&mut self.legs) {
            let v = match leg {
                MavpLeg::Ma(ma) => ma.update(close),
                MavpLeg::Kama(core) => core.update(&self.ring),
            };
            if period == chosen {
                out = v;
            }
        }
        if i < ma_lookback(self.maxperiod, self.matype) {
            return f64::NAN;
        }
        window_ma(self.matype, &self.ring, chosen).unwrap_or(out)
    }

    pub fn reset(&mut self) {
        self.ring.clear();
        self.bars = 0;
        for leg in &mut self.legs {
            match leg {
                MavpLeg::Ma(ma) => ma.reset(),
                MavpLeg::Kama(core) => core.reset(),
            }
        }
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        Self::check(self.minperiod, self.maxperiod, self.matype)?;
        if self.ring.len() != self.bars.min(self.ring_cap()) {
            return Err(StreamingError(format!(
                "mavp ring holds {} prices after {} bars",
                self.ring.len(),
                self.bars
            )));
        }
        let expected = if Self::is_window_type(self.matype) {
            0
        } else {
            self.maxperiod - self.minperiod + 1
        };
        if self.legs.len() != expected {
            return Err(StreamingError(format!(
                "mavp holds {} averages, expected {expected}",
                self.legs.len()
            )));
        }
        (self.minperiod..)
            .zip(&self.legs)
            .try_for_each(|(p, leg)| match leg {
                MavpLeg::Ma(ma) if self.matype != 6 => ma.validate_as(p, self.matype),
                MavpLeg::Kama(core) if self.matype == 6 && core.period() == p => {
                    core.validate(&self.ring)
                }
                _ => Err(StreamingError(format!(
                    "mavp state for period {p} does not match matype {}",
                    self.matype
                ))),
            })
    }

    pub fn minperiod(&self) -> usize {
        self.minperiod
    }

    pub fn maxperiod(&self) -> usize {
        self.maxperiod
    }

    pub fn matype(&self) -> u8 {
        self.matype
    }
}
