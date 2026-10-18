//! MA-composition oscillators: MACDFIX, MACDEXT, APO, PPO and TRIX.

use crate::overlap::ma_lookback;
use crate::streaming::primitives::{EmaCascade, EmaFirstFinite, EmaMacd, MaStream};
use crate::streaming::registry::{check_matype, check_period};
use crate::streaming::StreamingError;

const NAN3: (f64, f64, f64) = (f64::NAN, f64::NAN, f64::NAN);

/// `fastperiod < slowperiod`, as every fast/slow batch kernel requires.
fn check_fast_slow(fastperiod: usize, slowperiod: usize) -> Result<(), StreamingError> {
    check_period(fastperiod, "fastperiod", 1)?;
    check_period(slowperiod, "slowperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(StreamingError(format!(
            "fastperiod must be < slowperiod, got {fastperiod} >= {slowperiod}"
        )));
    }
    Ok(())
}

/// `a - b` when both are non-NaN.
fn diff(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a - b
    }
}

/// Registry entries for the MACD variants and the MA-difference oscillators.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_ma_osc {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                macdfix => StreamingMacdfix, doc "Streaming MACD with fixed 12/26 EMAs: (macd, signal, hist), NaN until the signal is valid.",
                    params(signalperiod: period = 9), inputs(close), outputs(out_macd: f64, out_signal: f64, out_hist: f64),
                    native(lookback = 26 + signalperiod - 1, batch = $crate::overlap::macdfix(close, signalperiod));
                macdext => StreamingMacdext, doc "Streaming MACD with a moving-average type per leg: (macd, signal, hist), NaN until the signal is valid.",
                    params(fastperiod: period = 12, fastmatype: matype = 1, slowperiod: period = 26, slowmatype: matype = 1, signalperiod: period = 9, signalmatype: matype = 1),
                    inputs(close), outputs(out_macd: f64, out_signal: f64, out_hist: f64),
                    native(lookback = $crate::overlap::ma_lookback(fastperiod, fastmatype).max($crate::overlap::ma_lookback(slowperiod, slowmatype)) + $crate::overlap::ma_lookback(signalperiod, signalmatype) + 1,
                        batch = $crate::overlap::macdext(close, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype));
                apo => StreamingApo, doc "Streaming absolute price oscillator: fast MA - slow MA (both of type `matype`).",
                    params(fastperiod: period = 12, slowperiod: period = 26, matype: matype = 1), inputs(close), outputs(out: f64),
                    native(lookback = $crate::overlap::ma_lookback(slowperiod, matype) + 1, batch = $crate::momentum::apo(close, fastperiod, slowperiod, matype));
                ppo => StreamingPpo, doc "Streaming percentage price oscillator: (ppo, signal, hist); the signal is an EMA of the PPO line.",
                    params(fastperiod: period = 12, slowperiod: period = 26, signalperiod: period = 9, matype: matype = 1), inputs(close),
                    outputs(out_ppo: f64, out_signal: f64, out_hist: f64),
                    native(lookback = $crate::overlap::ma_lookback(slowperiod, matype) + signalperiod, batch = $crate::momentum::ppo(close, fastperiod, slowperiod, signalperiod, matype));
                trix => StreamingTrix, doc "Streaming TRIX: one-bar percent rate of change of a triple EMA.",
                    params(timeperiod: period = 30), inputs(close), outputs(out: f64),
                    native(lookback = 3 * (timeperiod - 1) + 2, batch = $crate::momentum::trix(close, timeperiod));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_ma_osc! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Native: MACDFIX
// ---------------------------------------------------------------------------

/// Streaming MACDFIX (`overlap::macdfix` = `macd(close, 12, 26, signalperiod)`):
/// SMA-seeded unfused EMAs; all three outputs NaN until the signal exists.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMacdfix {
    fast: EmaMacd,
    slow: EmaMacd,
    signal: EmaMacd,
}

impl StreamingMacdfix {
    const FAST: usize = 12;
    const SLOW: usize = 26;

    pub fn new(signalperiod: usize) -> Result<Self, StreamingError> {
        check_period(signalperiod, "signalperiod", 1)?;
        Ok(Self {
            fast: EmaMacd::new(Self::FAST)?,
            slow: EmaMacd::new(Self::SLOW)?,
            signal: EmaMacd::new(signalperiod)?,
        })
    }

    /// Commit one bar; returns `(macd, signal, hist)`.
    pub fn update(&mut self, close: f64) -> (f64, f64, f64) {
        let fast = self.fast.update(close);
        let slow = self.slow.update(close);
        if slow.is_nan() {
            return NAN3;
        }
        // The signal is seeded from the first `signalperiod` MACD values.
        let macd = fast - slow;
        let signal = self.signal.update(macd);
        if signal.is_nan() {
            return NAN3;
        }
        (macd, signal, macd - signal)
    }

    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.signal.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.signal.period(), "signalperiod", 1)?;
        if self.fast.period() != Self::FAST || self.slow.period() != Self::SLOW {
            return Err(StreamingError("macdfix legs must be 12 / 26".to_string()));
        }
        self.fast.validate()?;
        self.slow.validate()?;
        self.signal.validate()
    }

    pub fn signalperiod(&self) -> usize {
        self.signal.period()
    }
}

// ---------------------------------------------------------------------------
// Native: MACDEXT
// ---------------------------------------------------------------------------

/// Streaming MACDEXT (`overlap::macdext`): the MACD line starts at
/// `max(ma_lookback(fast), ma_lookback(slow))`, the signal MA is fed from
/// there, and the MACD line stays NaN until the first signal value.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMacdext {
    fastmatype: u8,
    slowmatype: u8,
    signalmatype: u8,
    fast: MaStream,
    slow: MaStream,
    signal: MaStream,
    /// Committed bars (saturating).
    bars: usize,
    /// A signal value has been emitted, so the MACD line is unmasked.
    signal_seen: bool,
}

impl StreamingMacdext {
    pub fn new(
        fastperiod: usize,
        fastmatype: u8,
        slowperiod: usize,
        slowmatype: u8,
        signalperiod: usize,
        signalmatype: u8,
    ) -> Result<Self, StreamingError> {
        check_fast_slow(fastperiod, slowperiod)?;
        check_period(signalperiod, "signalperiod", 1)?;
        check_matype(fastmatype, "fastmatype")?;
        check_matype(slowmatype, "slowmatype")?;
        check_matype(signalmatype, "signalmatype")?;
        Ok(Self {
            fastmatype,
            slowmatype,
            signalmatype,
            fast: MaStream::new(fastperiod, fastmatype)?,
            slow: MaStream::new(slowperiod, slowmatype)?,
            signal: MaStream::new(signalperiod, signalmatype)?,
            bars: 0,
            signal_seen: false,
        })
    }

    /// Commit one bar; returns `(macd, signal, hist)`.
    pub fn update(&mut self, close: f64) -> (f64, f64, f64) {
        let i = self.bars;
        self.bars = self.bars.saturating_add(1);
        let fast = self.fast.update(close);
        let slow = self.slow.update(close);
        let macd_start = ma_lookback(self.fast.period(), self.fastmatype)
            .max(ma_lookback(self.slow.period(), self.slowmatype));
        if i < macd_start {
            return NAN3;
        }
        let macd = diff(fast, slow);
        let raw = self.signal.update(macd);
        let warmup =
            macd_start.saturating_add(ma_lookback(self.signal.period(), self.signalmatype));
        let signal = if i >= warmup { raw } else { f64::NAN };
        self.signal_seen |= !signal.is_nan();
        let shown = if self.signal_seen { macd } else { f64::NAN };
        (shown, signal, diff(macd, signal))
    }

    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.signal.reset();
        self.bars = 0;
        self.signal_seen = false;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_fast_slow(self.fast.period(), self.slow.period())?;
        self.fast.validate_as(self.fast.period(), self.fastmatype)?;
        self.slow.validate_as(self.slow.period(), self.slowmatype)?;
        self.signal
            .validate_as(self.signal.period(), self.signalmatype)
    }

    pub fn fastperiod(&self) -> usize {
        self.fast.period()
    }

    pub fn fastmatype(&self) -> u8 {
        self.fastmatype
    }

    pub fn slowperiod(&self) -> usize {
        self.slow.period()
    }

    pub fn slowmatype(&self) -> u8 {
        self.slowmatype
    }

    pub fn signalperiod(&self) -> usize {
        self.signal.period()
    }

    pub fn signalmatype(&self) -> u8 {
        self.signalmatype
    }
}

// ---------------------------------------------------------------------------
// Native: APO / PPO
// ---------------------------------------------------------------------------

/// Fast and slow [`MaStream`]s of one `matype` with the slow leg's warm-up
/// gate — the shared core of `apo` and `ppo`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct FastSlow {
    matype: u8,
    fast: MaStream,
    slow: MaStream,
    /// Committed bars (saturating).
    bars: usize,
}

impl FastSlow {
    fn new(fastperiod: usize, slowperiod: usize, matype: u8) -> Result<Self, StreamingError> {
        check_fast_slow(fastperiod, slowperiod)?;
        Ok(Self {
            matype,
            fast: MaStream::new(fastperiod, matype)?,
            slow: MaStream::new(slowperiod, matype)?,
            bars: 0,
        })
    }

    /// `(bar index, fast, slow)`; both NaN before `ma_lookback(slow)`.
    fn update(&mut self, close: f64) -> (usize, f64, f64) {
        let i = self.bars;
        self.bars = self.bars.saturating_add(1);
        let fast = self.fast.update(close);
        let slow = self.slow.update(close);
        if i < self.warmup() {
            return (i, f64::NAN, f64::NAN);
        }
        (i, fast, slow)
    }

    fn warmup(&self) -> usize {
        ma_lookback(self.slow.period(), self.matype)
    }

    fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.bars = 0;
    }

    fn validate(&self) -> Result<(), StreamingError> {
        check_fast_slow(self.fast.period(), self.slow.period())?;
        self.fast.validate_as(self.fast.period(), self.matype)?;
        self.slow.validate_as(self.slow.period(), self.matype)
    }
}

/// Streaming APO (`momentum::apo`): fast MA - slow MA.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingApo {
    legs: FastSlow,
}

impl StreamingApo {
    pub fn new(fastperiod: usize, slowperiod: usize, matype: u8) -> Result<Self, StreamingError> {
        Ok(Self {
            legs: FastSlow::new(fastperiod, slowperiod, matype)?,
        })
    }

    /// Commit one bar (NaN for the first `ma_lookback(slowperiod, matype)` bars).
    pub fn update(&mut self, close: f64) -> f64 {
        let (_, fast, slow) = self.legs.update(close);
        diff(fast, slow)
    }

    pub fn reset(&mut self) {
        self.legs.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.legs.validate()
    }

    pub fn fastperiod(&self) -> usize {
        self.legs.fast.period()
    }

    pub fn slowperiod(&self) -> usize {
        self.legs.slow.period()
    }

    pub fn matype(&self) -> u8 {
        self.legs.matype
    }
}

/// Streaming PPO (`momentum::ppo`): `(fast - slow) / slow * 100` (NaN where
/// the slow MA is 0), a first-finite EMA signal of it, and the histogram.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingPpo {
    legs: FastSlow,
    signal: EmaFirstFinite,
}

impl StreamingPpo {
    pub fn new(
        fastperiod: usize,
        slowperiod: usize,
        signalperiod: usize,
        matype: u8,
    ) -> Result<Self, StreamingError> {
        check_matype(matype, "matype")?;
        Ok(Self {
            legs: FastSlow::new(fastperiod, slowperiod, matype)?,
            signal: EmaFirstFinite::new(signalperiod)?,
        })
    }

    /// Commit one bar; returns `(ppo, signal, hist)`.
    pub fn update(&mut self, close: f64) -> (f64, f64, f64) {
        let (i, fast, slow) = self.legs.update(close);
        let ppo = if slow != 0.0 {
            diff(fast, slow) / slow * 100.0
        } else {
            f64::NAN
        };
        // Fed every bar, NaN warm-up included, like the batch first-finite EMA.
        let raw = self.signal.update(ppo);
        let warmup = self.legs.warmup().saturating_add(self.signal.period() - 1);
        if i < warmup || ppo.is_nan() || raw.is_nan() {
            return (ppo, f64::NAN, f64::NAN);
        }
        (ppo, raw, ppo - raw)
    }

    pub fn reset(&mut self) {
        self.legs.reset();
        self.signal.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.legs.validate()?;
        self.signal.validate()
    }

    pub fn fastperiod(&self) -> usize {
        self.legs.fast.period()
    }

    pub fn slowperiod(&self) -> usize {
        self.legs.slow.period()
    }

    pub fn signalperiod(&self) -> usize {
        self.signal.period()
    }

    pub fn matype(&self) -> u8 {
        self.legs.matype
    }
}

// ---------------------------------------------------------------------------
// Native: TRIX
// ---------------------------------------------------------------------------

/// Streaming TRIX (`momentum::trix`): one-bar percent change of the third
/// first-finite EMA stage (NaN where the previous value is NaN or 0).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingTrix {
    cascade: EmaCascade,
    /// Previous bar's third-stage EMA.
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev: f64,
}

impl StreamingTrix {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            cascade: EmaCascade::new(timeperiod, 3)?,
            prev: f64::NAN,
        })
    }

    /// Commit one bar (NaN for the first `3 * (timeperiod - 1) + 1` bars).
    pub fn update(&mut self, close: f64) -> f64 {
        let e3 = self.cascade.update(close)[2];
        let prev = std::mem::replace(&mut self.prev, e3);
        if e3.is_nan() || prev.is_nan() || prev == 0.0 {
            return f64::NAN;
        }
        (e3 - prev) / prev * 100.0
    }

    pub fn reset(&mut self) {
        self.cascade.reset();
        self.prev = f64::NAN;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.cascade.validate(3)
    }

    pub fn timeperiod(&self) -> usize {
        self.cascade.period()
    }
}
