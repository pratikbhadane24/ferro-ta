//! Overlap studies: DEMA, TEMA, T3, TRIMA, WMA, MIDPOINT, MIDPRICE, KAMA,
//! MA and MAVP.

use crate::overlap::ma_lookback;
use crate::streaming::primitives::{MaStream, T3};
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

/// Streaming moving average with a per-bar period (`overlap::mavp`).
///
/// Holds one [`MaStream`] per period in `[minperiod, maxperiod]`, all fed
/// every bar (as the batch kernel runs a full MA per requested period), and
/// emits the one `periods` selects — rounded, then clamped. NaN for the first
/// `ma_lookback(maxperiod, matype)` bars whatever the per-bar period.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMavp {
    minperiod: usize,
    maxperiod: usize,
    matype: u8,
    /// `mas[k]` has period `minperiod + k`.
    mas: Vec<MaStream>,
    /// Committed bars (saturating).
    bars: usize,
}

impl StreamingMavp {
    pub fn new(minperiod: usize, maxperiod: usize, matype: u8) -> Result<Self, StreamingError> {
        Self::check(minperiod, maxperiod, matype)?;
        let mas = (minperiod..=maxperiod)
            .map(|p| MaStream::new(p, matype))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            minperiod,
            maxperiod,
            matype,
            mas,
            bars: 0,
        })
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
        Ok(())
    }

    /// Commit one bar; `periods` is this bar's period, rounded and clamped
    /// to `[minperiod, maxperiod]` like the batch kernel.
    pub fn update(&mut self, close: f64, periods: f64) -> f64 {
        let i = self.bars;
        self.bars = self.bars.saturating_add(1);
        let chosen = (periods.round() as usize).clamp(self.minperiod, self.maxperiod);
        let mut out = f64::NAN;
        for (period, ma) in (self.minperiod..).zip(&mut self.mas) {
            let v = ma.update(close);
            if period == chosen {
                out = v;
            }
        }
        if i < ma_lookback(self.maxperiod, self.matype) {
            return f64::NAN;
        }
        out
    }

    pub fn reset(&mut self) {
        self.mas.iter_mut().for_each(MaStream::reset);
        self.bars = 0;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        Self::check(self.minperiod, self.maxperiod, self.matype)?;
        if self.mas.len() != self.maxperiod - self.minperiod + 1 {
            return Err(StreamingError(format!(
                "mavp holds {} averages, expected {}",
                self.mas.len(),
                self.maxperiod - self.minperiod + 1
            )));
        }
        (self.minperiod..)
            .zip(&self.mas)
            .try_for_each(|(p, ma)| ma.validate_as(p, self.matype))
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
