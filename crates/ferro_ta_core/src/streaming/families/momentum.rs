//! Momentum oscillators: range / rate-of-change windows, CMO, StochF and
//! StochRSI. The directional-movement family lives in `directional.rs`.

use std::collections::VecDeque;

use crate::overlap::ma_lookback;
use crate::streaming::primitives::{MaStream, Wilder};
use crate::streaming::registry::{check_matype, check_period};
use crate::streaming::StreamingError;

const NAN2: (f64, f64) = (f64::NAN, f64::NAN);

/// Registry entries for the momentum streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_momentum {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                willr => StreamingWillr, doc "Streaming Williams %R: -100 * (highest high - close) / (highest high - lowest low) over `timeperiod` bars.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::momentum::willr(high, low, close, timeperiod));
                cci => StreamingCci, doc "Streaming Commodity Channel Index over `timeperiod` bars.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::momentum::cci(high, low, close, timeperiod));
                bop => StreamingBop, doc "Streaming Balance of Power: (close - open) / (high - low).",
                    params(), inputs(open, high, low, close), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::momentum::bop(open, high, low, close));
                mom => StreamingMom, doc "Streaming momentum: close - close[timeperiod bars ago].",
                    params(timeperiod: period = 10), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::momentum::mom(close, timeperiod));
                roc => StreamingRoc, doc "Streaming rate of change: (close - prev) / prev * 100 (NaN where prev is 0).",
                    params(timeperiod: period = 10), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::momentum::roc(close, timeperiod));
                rocp => StreamingRocp, doc "Streaming rate of change percentage: (close - prev) / prev (NaN where prev is 0).",
                    params(timeperiod: period = 10), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::momentum::rocp(close, timeperiod));
                rocr => StreamingRocr, doc "Streaming rate of change ratio: close / prev (NaN where prev is 0).",
                    params(timeperiod: period = 10), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::momentum::rocr(close, timeperiod));
                rocr100 => StreamingRocr100, doc "Streaming rate of change ratio x 100: close / prev * 100 (NaN where prev is 0).",
                    params(timeperiod: period = 10), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::momentum::rocr100(close, timeperiod));
                aroon => StreamingAroon, doc "Streaming Aroon: (aroon_down, aroon_up) over a `timeperiod + 1` bar window; ties go to the most recent bar.",
                    params(timeperiod: period = 14), inputs(high, low), outputs(out_aroondown: f64, out_aroonup: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::momentum::aroon(high, low, timeperiod));
                aroonosc => StreamingAroonosc, doc "Streaming Aroon oscillator: aroon_up - aroon_down.",
                    params(timeperiod: period = 14), inputs(high, low), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::momentum::aroonosc(high, low, timeperiod));
                ultosc => StreamingUltosc, doc "Streaming Ultimate Oscillator: buying pressure over three periods weighted 4 / 2 / 1.",
                    params(timeperiod1: period = 7, timeperiod2: period = 14, timeperiod3: period = 28),
                    inputs(high, low, close), outputs(out: f64),
                    windowed(lookback = timeperiod1.max(timeperiod2).max(timeperiod3) + 1,
                        batch = $crate::momentum::ultosc(high, low, close, timeperiod1, timeperiod2, timeperiod3));
                cmo => StreamingCmo, doc "Streaming Chande momentum oscillator (Wilder smoothed gains and losses).",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::momentum::cmo(close, timeperiod));
                stochf => StreamingStochf, doc "Streaming fast stochastic: (fastk, fastd); both NaN until %D is valid.",
                    params(fastk_period: period = 5, fastd_period: period = 3, fastd_matype: matype = 0),
                    inputs(high, low, close), outputs(out_fastk: f64, out_fastd: f64),
                    native(lookback = fastk_period + $crate::overlap::ma_lookback(fastd_period, fastd_matype),
                        batch = $crate::momentum::stochf(high, low, close, fastk_period, fastd_period, fastd_matype));
                stochrsi => StreamingStochrsi, doc "Streaming stochastic RSI: (fastk, fastd); both NaN until %D is valid.",
                    params(timeperiod: period = 14, fastk_period: period = 5, fastd_period: period = 3, fastd_matype: matype = 0),
                    inputs(close), outputs(out_fastk: f64, out_fastd: f64),
                    native(lookback = timeperiod + fastk_period + $crate::overlap::ma_lookback(fastd_period, fastd_matype),
                        batch = $crate::momentum::stochrsi(close, timeperiod, fastk_period, fastd_period, fastd_matype));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_momentum! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Shared: Wilder-smoothed gains and losses (RSI / CMO)
// ---------------------------------------------------------------------------

/// Average gain and loss of the close-to-close changes, as `rsi` / `cmo`
/// compute them: gains `(d + |d|) * 0.5`, losses `(|d| - d) * 0.5`, SMA seed
/// over the first `timeperiod` changes, then `(avg * (p - 1) + x) * (1 / p)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct GainLoss {
    gain: Wilder,
    loss: Wilder,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev: f64,
    has_prev: bool,
}

impl GainLoss {
    pub(crate) fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            gain: Wilder::new(timeperiod)?,
            loss: Wilder::new(timeperiod)?,
            prev: f64::NAN,
            has_prev: false,
        })
    }

    /// `(avg_gain, avg_loss)` once seeded (at the bar index `timeperiod`).
    pub(crate) fn update(&mut self, close: f64) -> Option<(f64, f64)> {
        if !self.has_prev {
            self.has_prev = true;
            self.prev = close;
            return None;
        }
        let diff = close - self.prev;
        let abs_diff = diff.abs();
        self.prev = close;
        let gain = self.gain.update((diff + abs_diff) * 0.5);
        let loss = self.loss.update((abs_diff - diff) * 0.5);
        (!gain.is_nan()).then_some((gain, loss))
    }

    pub(crate) fn period(&self) -> usize {
        self.gain.period()
    }

    pub(crate) fn reset(&mut self) {
        self.gain.reset();
        self.loss.reset();
        self.prev = f64::NAN;
        self.has_prev = false;
    }

    pub(crate) fn validate(&self) -> Result<(), StreamingError> {
        self.gain.validate()?;
        self.loss.validate()?;
        if self.gain.period() != self.loss.period() {
            return Err(StreamingError(
                "gain and loss averages disagree on the period".to_string(),
            ));
        }
        Ok(())
    }
}

/// `rsi`'s value from the averages (flat window -> 0, clamped to `[0, 100]`).
pub(crate) fn rsi_value(gain: f64, loss: f64) -> f64 {
    let denom = gain + loss;
    if denom == 0.0 {
        0.0
    } else {
        (100.0 * gain / denom).clamp(0.0, 100.0)
    }
}

/// `cmo`'s value from the averages (flat window -> 0, clamped to `[-100, 100]`).
fn cmo_value(gain: f64, loss: f64) -> f64 {
    let denom = gain + loss;
    if denom == 0.0 {
        0.0
    } else {
        (100.0 * (gain - loss) / denom).clamp(-100.0, 100.0)
    }
}

// ---------------------------------------------------------------------------
// Native: CMO
// ---------------------------------------------------------------------------

/// Streaming CMO (`momentum::cmo`): first value at bar `timeperiod`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingCmo {
    averages: GainLoss,
}

impl StreamingCmo {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            averages: GainLoss::new(timeperiod)?,
        })
    }

    /// Commit one bar (NaN for the first `timeperiod` bars).
    pub fn update(&mut self, close: f64) -> f64 {
        self.averages
            .update(close)
            .map_or(f64::NAN, |(gain, loss)| cmo_value(gain, loss))
    }

    pub fn reset(&mut self) {
        self.averages.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.averages.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.averages.period()
    }
}

// ---------------------------------------------------------------------------
// Shared: the %D leg of the stochastics
// ---------------------------------------------------------------------------

/// Push `x` onto `ring`, keeping at most `cap` values.
fn push_capped(ring: &mut VecDeque<f64>, x: f64, cap: usize) {
    ring.push_back(x);
    if ring.len() > cap {
        ring.pop_front();
    }
}

/// `ring` is a full window with no NaN.
fn is_clean_window(ring: &VecDeque<f64>, period: usize) -> bool {
    ring.len() == period && ring.iter().all(|v| !v.is_nan())
}

// ---------------------------------------------------------------------------
// Native: StochF
// ---------------------------------------------------------------------------

/// Streaming StochF (`momentum::stochf`): fast %K from the rolling high / low
/// (flat range -> 0), %D = `MaStream(fastd_matype)` of %K. Both outputs are
/// NaN until %D is valid, at bar `fastk_period - 1 + ma_lookback(fastd)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingStochf {
    fastk_period: usize,
    fastd_matype: u8,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    fastd: MaStream,
    /// %K values fed to `fastd` (saturating).
    fed: usize,
}

impl StreamingStochf {
    pub fn new(
        fastk_period: usize,
        fastd_period: usize,
        fastd_matype: u8,
    ) -> Result<Self, StreamingError> {
        check_period(fastk_period, "fastk_period", 1)?;
        check_matype(fastd_matype, "fastd_matype")?;
        Ok(Self {
            fastk_period,
            fastd_matype,
            highs: VecDeque::new(),
            lows: VecDeque::new(),
            fastd: MaStream::new(fastd_period, fastd_matype)?,
            fed: 0,
        })
    }

    /// Commit one bar; returns `(fastk, fastd)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        push_capped(&mut self.highs, high, self.fastk_period);
        push_capped(&mut self.lows, low, self.fastk_period);
        if self.highs.len() < self.fastk_period {
            return NAN2;
        }
        // Strict comparisons from the oldest bar, as the batch scan.
        let mut hh = self.highs[0];
        let mut ll = self.lows[0];
        for (&h, &l) in self.highs.iter().zip(&self.lows) {
            if h > hh {
                hh = h;
            }
            if l < ll {
                ll = l;
            }
        }
        let range = hh - ll;
        let fastk = if range != 0.0 {
            100.0 * (close - ll) / range
        } else {
            0.0
        };
        let fastd = self.fastd.update(fastk);
        let index = self.fed;
        self.fed = self.fed.saturating_add(1);
        if index < ma_lookback(self.fastd.period(), self.fastd_matype) {
            return NAN2;
        }
        (fastk, fastd)
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.fastd.reset();
        self.fed = 0;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.fastk_period, "fastk_period", 1)?;
        check_matype(self.fastd_matype, "fastd_matype")?;
        if self.highs.len() != self.lows.len() || self.highs.len() > self.fastk_period {
            return Err(StreamingError(format!(
                "stochf windows hold {} highs and {} lows for fastk_period {}",
                self.highs.len(),
                self.lows.len(),
                self.fastk_period
            )));
        }
        if self.highs.len() < self.fastk_period && self.fed > 0 {
            return Err(StreamingError(
                "stochf fed %D before its window was full".to_string(),
            ));
        }
        self.fastd
            .validate_as(self.fastd.period(), self.fastd_matype)
    }

    pub fn fastk_period(&self) -> usize {
        self.fastk_period
    }

    pub fn fastd_period(&self) -> usize {
        self.fastd.period()
    }

    pub fn fastd_matype(&self) -> u8 {
        self.fastd_matype
    }
}

// ---------------------------------------------------------------------------
// Native: StochRSI
// ---------------------------------------------------------------------------

/// Streaming StochRSI (`momentum::stochrsi`): %K is the stochastic of the RSI
/// over `fastk_period` values (flat -> 0, NaN while the window holds a NaN);
/// %D is a per-window mean for SMA (`fastd_matype` 0, NaN-aware like the
/// batch) or a `MaStream`. Both outputs are NaN until %D is valid, at bar
/// `timeperiod + fastk_period - 1 + ma_lookback(fastd)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingStochrsi {
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: u8,
    averages: GainLoss,
    /// Latest RSI values (NaN while seeding).
    #[cfg_attr(feature = "serde", serde(with = "crate::streaming::snapshot::nan_seq"))]
    rsi_window: VecDeque<f64>,
    /// Latest %K values (SMA %D only).
    #[cfg_attr(feature = "serde", serde(with = "crate::streaming::snapshot::nan_seq"))]
    k_window: VecDeque<f64>,
    /// Typed %D (every `fastd_matype` but 0).
    fastd: Option<MaStream>,
    /// Committed bars (saturating).
    bars: usize,
}

impl StreamingStochrsi {
    pub fn new(
        timeperiod: usize,
        fastk_period: usize,
        fastd_period: usize,
        fastd_matype: u8,
    ) -> Result<Self, StreamingError> {
        check_period(fastk_period, "fastk_period", 1)?;
        check_period(fastd_period, "fastd_period", 1)?;
        check_matype(fastd_matype, "fastd_matype")?;
        Ok(Self {
            fastk_period,
            fastd_period,
            fastd_matype,
            averages: GainLoss::new(timeperiod)?,
            rsi_window: VecDeque::new(),
            k_window: VecDeque::new(),
            fastd: Self::typed_fastd(fastd_period, fastd_matype)?,
            bars: 0,
        })
    }

    fn typed_fastd(period: usize, matype: u8) -> Result<Option<MaStream>, StreamingError> {
        (matype != 0)
            .then(|| MaStream::new(period, matype))
            .transpose()
    }

    /// First bar at which %K exists.
    fn k_warmup(&self) -> usize {
        self.averages.period().saturating_add(self.fastk_period - 1)
    }

    /// Stochastic of the newest RSI against the window (NaN if it holds a NaN).
    fn fastk(&self) -> f64 {
        if !is_clean_window(&self.rsi_window, self.fastk_period) {
            return f64::NAN;
        }
        let newest = self.rsi_window.back().copied().unwrap_or(f64::NAN);
        let max = self
            .rsi_window
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let min = self
            .rsi_window
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if max != min {
            100.0 * (newest - min) / (max - min)
        } else {
            0.0
        }
    }

    /// Commit one bar; returns `(fastk, fastd)`.
    pub fn update(&mut self, close: f64) -> (f64, f64) {
        let index = self.bars;
        self.bars = self.bars.saturating_add(1);
        let rsi = self
            .averages
            .update(close)
            .map_or(f64::NAN, |(gain, loss)| rsi_value(gain, loss));
        push_capped(&mut self.rsi_window, rsi, self.fastk_period);
        if index < self.k_warmup() {
            return NAN2;
        }
        let fastk = self.fastk();
        let fastd = match &mut self.fastd {
            Some(ma) => ma.update(fastk),
            None => {
                push_capped(&mut self.k_window, fastk, self.fastd_period);
                if is_clean_window(&self.k_window, self.fastd_period) {
                    self.k_window.iter().sum::<f64>() / self.fastd_period as f64
                } else {
                    f64::NAN
                }
            }
        };
        let d_warmup = self
            .k_warmup()
            .saturating_add(ma_lookback(self.fastd_period, self.fastd_matype));
        if index < d_warmup {
            return NAN2;
        }
        (fastk, fastd)
    }

    pub fn reset(&mut self) {
        self.averages.reset();
        self.rsi_window.clear();
        self.k_window.clear();
        if let Some(ma) = &mut self.fastd {
            ma.reset();
        }
        self.bars = 0;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.fastk_period, "fastk_period", 1)?;
        check_period(self.fastd_period, "fastd_period", 1)?;
        check_matype(self.fastd_matype, "fastd_matype")?;
        self.averages.validate()?;
        if self.rsi_window.len() > self.fastk_period || self.k_window.len() > self.fastd_period {
            return Err(StreamingError(
                "stochrsi window exceeds its period".to_string(),
            ));
        }
        match &self.fastd {
            Some(ma) if self.fastd_matype != 0 => {
                if !self.k_window.is_empty() {
                    return Err(StreamingError(
                        "stochrsi holds an SMA window for a typed %D".to_string(),
                    ));
                }
                ma.validate_as(self.fastd_period, self.fastd_matype)
            }
            None if self.fastd_matype == 0 => Ok(()),
            _ => Err(StreamingError(
                "stochrsi %D state contradicts fastd_matype".to_string(),
            )),
        }
    }

    pub fn timeperiod(&self) -> usize {
        self.averages.period()
    }

    pub fn fastk_period(&self) -> usize {
        self.fastk_period
    }

    pub fn fastd_period(&self) -> usize {
        self.fastd_period
    }

    pub fn fastd_matype(&self) -> u8 {
        self.fastd_matype
    }
}
