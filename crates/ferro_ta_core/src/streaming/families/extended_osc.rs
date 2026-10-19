//! Extended oscillators: RWI, AO, AC, PO, DPO, KST, Vortex, Coppock
//! (windowed); RVI, DMI, Elder Ray, Fisher, Connors RSI, TSI and STC
//! (native).

use super::directional::DmState;
use super::extended_common::{check_fast_below_slow, finite2, Ring};
use super::momentum::{rsi_value, GainLoss};
use crate::streaming::primitives::EmaTa;
use crate::streaming::registry::check_period;
use crate::streaming::StreamingError;

const NAN2: (f64, f64) = (f64::NAN, f64::NAN);

/// Registry entries for the extended oscillators.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_extended_osc {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                rwi => StreamingRwi, doc "Streaming Random Walk Index: (high, low) maxima over lookbacks 2..=`timeperiod` of the range move per ATR * sqrt(lookback).",
                    params(timeperiod: period(2) = 14), inputs(high, low, close), outputs(out_rwi_high: f64, out_rwi_low: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::extended::rwi(high, low, close, timeperiod));
                ao => StreamingAo, doc "Streaming Awesome Oscillator: SMA(median price, fast) - SMA(median price, slow).",
                    params(fastperiod: period = 5, slowperiod: period = 34), inputs(high, low), outputs(out: f64),
                    windowed(lookback = slowperiod, batch = $crate::extended::ao(high, low, fastperiod, slowperiod), requires = fastperiod < slowperiod);
                ac => StreamingAc, doc "Streaming Accelerator Oscillator: AO - SMA(AO, `timeperiod`).",
                    params(fastperiod: period = 5, slowperiod: period = 34, timeperiod: period = 5), inputs(high, low), outputs(out: f64),
                    windowed(lookback = slowperiod + timeperiod - 1, batch = $crate::extended::ac(high, low, fastperiod, slowperiod, timeperiod), requires = fastperiod < slowperiod);
                po => StreamingPo, doc "Streaming price oscillator: SMA(close, fast) - SMA(close, slow).",
                    params(fastperiod: period = 10, slowperiod: period = 21), inputs(close), outputs(out: f64),
                    windowed(lookback = slowperiod, batch = $crate::extended::po(close, fastperiod, slowperiod), requires = fastperiod < slowperiod);
                dpo => StreamingDpo, doc "Streaming detrended price oscillator: close `timeperiod / 2 + 1` bars ago minus SMA(close, `timeperiod`).",
                    params(timeperiod: period = 20), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod.max(timeperiod / 2 + 2), batch = $crate::extended::dpo(close, timeperiod));
                rvi => StreamingRvi, doc "Streaming Relative Vigor Index and its 4-bar weighted signal (the batch running window sums are mirrored, so a flat window's zero denominator matches the batch function).",
                    params(timeperiod: period = 10), inputs(open, high, low, close), outputs(out_rvi: f64, out_signal: f64),
                    native(lookback = timeperiod + 6, batch = $crate::extended::rvi(open, high, low, close, timeperiod));
                kst => StreamingKst, doc "Streaming Know Sure Thing: weighted sum of four SMA-smoothed ROCs, plus an SMA signal.",
                    params(roc1: period = 10, roc2: period = 15, roc3: period = 20, roc4: period = 30, sma1: period = 10, sma2: period = 10, sma3: period = 10, sma4: period = 15, signalperiod: period = 9),
                    inputs(close), outputs(out_kst: f64, out_signal: f64),
                    windowed(lookback = (roc1 + sma1).max(roc2 + sma2).max(roc3 + sma3).max(roc4 + sma4) + signalperiod - 1,
                        batch = $crate::extended::kst(close, roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4, signalperiod));
                vortex => StreamingVortex, doc "Streaming Vortex indicator: (+VI, -VI) over `timeperiod` bars; a non-finite true range counts as 0.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out_plus_vi: f64, out_minus_vi: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::extended::vortex(high, low, close, timeperiod));
                coppock => StreamingCoppock, doc "Streaming Coppock curve: WMA(ROC(`roc1`) + ROC(`roc2`), `wma_period`).",
                    params(wma_period: period = 10, roc1: period = 14, roc2: period = 11), inputs(close), outputs(out: f64),
                    windowed(lookback = roc1.max(roc2) + wma_period, batch = $crate::extended::coppock(close, wma_period, roc1, roc2));
                dmi => StreamingDmi, doc "Streaming directional movement index: (+DI, -DI, ADX) from one Wilder state.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out_plus_di: f64, out_minus_di: f64, out_adx: f64),
                    native(lookback = 2 * timeperiod, batch = $crate::extended::dmi(high, low, close, timeperiod));
                elder_ray => StreamingElderRay, doc "Streaming Elder Ray: (bull, bear) = (high - EMA(close), low - EMA(close)).",
                    params(timeperiod: period = 13), inputs(high, low, close), outputs(out_bull: f64, out_bear: f64),
                    native(lookback = timeperiod, batch = $crate::extended::elder_ray(high, low, close, timeperiod));
                fisher => StreamingFisher, doc "Streaming Ehlers Fisher transform of the median price over `timeperiod` bars, plus the previous value as signal.",
                    params(timeperiod: period = 9), inputs(high, low), outputs(out_fisher: f64, out_signal: f64),
                    native(lookback = timeperiod + 1, batch = $crate::extended::fisher(high, low, timeperiod));
                crsi => StreamingCrsi, doc "Streaming Connors RSI: mean of RSI(close), RSI(up/down streak) and the percent rank of the 1-bar ROC.",
                    params(timeperiod: period = 3, streakperiod: period = 2, rankperiod: period = 100), inputs(close), outputs(out: f64),
                    native(lookback = (timeperiod + 1).max(streakperiod + 1).max(rankperiod + 2), batch = $crate::extended::crsi(close, timeperiod, streakperiod, rankperiod));
                tsi => StreamingTsi, doc "Streaming True Strength Index: 100 * double-EMA(momentum) / double-EMA(abs(momentum)), plus an EMA signal.",
                    params(longperiod: period = 25, shortperiod: period = 13, signalperiod: period = 13), inputs(close), outputs(out_tsi: f64, out_signal: f64),
                    native(lookback = longperiod + shortperiod + signalperiod - 1, batch = $crate::extended::tsi(close, longperiod, shortperiod, signalperiod));
                stc => StreamingStc, doc "Streaming Schaff Trend Cycle: stochastic of the EMA MACD, EMA-smoothed, stochastic again, EMA-smoothed, clamped to [0, 100].",
                    params(fastperiod: period = 23, slowperiod: period = 50, cycleperiod: period = 10, d1: period = 3, d2: period = 3), inputs(close), outputs(out: f64),
                    native(lookback = slowperiod + 2 * cycleperiod + d1 + d2 - 4, batch = $crate::extended::stc(close, fastperiod, slowperiod, cycleperiod, d1, d2));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_extended_osc! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Native: DMI
// ---------------------------------------------------------------------------

/// Streaming DMI (`extended::dmi` = `momentum::adx_all`'s +DI, -DI and ADX),
/// all three from one [`DmState`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingDmi {
    state: DmState,
}

impl StreamingDmi {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            state: DmState::new(timeperiod)?,
        })
    }

    /// Commit one bar: `(plus_di, minus_di, adx)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        let step = self.state.update(high, low, close);
        (step.plus_di, step.minus_di, step.adx)
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.state.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.state.period()
    }
}

// ---------------------------------------------------------------------------
// Native: ELDER_RAY
// ---------------------------------------------------------------------------

/// Streaming Elder Ray (`extended::elder_ray`): `overlap::ema` of close;
/// both powers NaN until it is finite.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingElderRay {
    ema: EmaTa,
}

impl StreamingElderRay {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            ema: EmaTa::new(timeperiod)?,
        })
    }

    /// Commit one bar: `(bull, bear)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        let ema = self.ema.update(close);
        if ema.is_finite() {
            (high - ema, low - ema)
        } else {
            NAN2
        }
    }

    pub fn reset(&mut self) {
        self.ema.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.ema.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.ema.period()
    }
}

// ---------------------------------------------------------------------------
// Native: FISHER
// ---------------------------------------------------------------------------

/// Streaming Fisher transform (`extended::fisher`): median price normalised
/// over `timeperiod` bars, smoothed `0.33 * norm + 0.67 * value` (clamped to
/// ±0.999), then `0.5 * ln((1 + v) / (1 - v)) + 0.5 * previous`; the signal
/// is the previous Fisher value.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingFisher {
    timeperiod: usize,
    medians: Ring,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    value: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_fish: f64,
    have_fish: bool,
}

impl StreamingFisher {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            medians: Ring::default(),
            value: 0.0,
            prev_fish: 0.0,
            have_fish: false,
        })
    }

    /// Commit one bar: `(fisher, signal)`.
    pub fn update(&mut self, high: f64, low: f64) -> (f64, f64) {
        let hl2 = (high + low) / 2.0;
        self.medians.push(hl2, self.timeperiod);
        if self.medians.len() < self.timeperiod {
            return NAN2;
        }
        let (max_h, min_l) = (self.medians.max(), self.medians.min());
        if !max_h.is_finite() || !min_l.is_finite() {
            return NAN2;
        }
        let range = max_h - min_l;
        let norm = if range > 0.0 {
            2.0 * ((hl2 - min_l) / range - 0.5)
        } else {
            0.0
        };
        self.value = (0.33 * norm + 0.67 * self.value).clamp(-0.999, 0.999);
        let next = 0.5 * ((1.0 + self.value) / (1.0 - self.value)).ln() + 0.5 * self.prev_fish;
        let signal = if self.have_fish {
            self.prev_fish
        } else {
            f64::NAN
        };
        self.prev_fish = next;
        self.have_fish = true;
        (next, signal)
    }

    pub fn reset(&mut self) {
        self.medians.clear();
        self.value = 0.0;
        self.prev_fish = 0.0;
        self.have_fish = false;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 1)?;
        self.medians.validate(self.timeperiod, "fisher window")
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }
}

// ---------------------------------------------------------------------------
// Native: CRSI
// ---------------------------------------------------------------------------

/// Streaming Connors RSI (`extended::crsi`): `rsi(close)`, `rsi(streak)`
/// (the up/down run length, 0 on bar 0 and on unchanged bars) and the percent
/// rank of the 1-bar ROC among the `rankperiod` ROCs *before* it; NaN unless
/// all three are finite.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingCrsi {
    rankperiod: usize,
    price: GainLoss,
    streak_rsi: GainLoss,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_close: f64,
    has_prev: bool,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    streak: f64,
    /// The `rankperiod` ROC values before the current bar.
    rocs: Ring,
}

impl StreamingCrsi {
    pub fn new(
        timeperiod: usize,
        streakperiod: usize,
        rankperiod: usize,
    ) -> Result<Self, StreamingError> {
        check_period(rankperiod, "rankperiod", 1)?;
        Ok(Self {
            rankperiod,
            price: GainLoss::new(timeperiod)?,
            streak_rsi: GainLoss::new(streakperiod)?,
            prev_close: f64::NAN,
            has_prev: false,
            streak: 0.0,
            rocs: Ring::default(),
        })
    }

    /// `up_down_streak`'s next value after `prev_close`.
    fn next_streak(&self, close: f64) -> f64 {
        let diff = close - self.prev_close;
        if diff > 0.0 {
            if self.streak > 0.0 {
                self.streak + 1.0
            } else {
                1.0
            }
        } else if diff < 0.0 {
            if self.streak < 0.0 {
                self.streak - 1.0
            } else {
                -1.0
            }
        } else {
            0.0
        }
    }

    /// Percent rank of `roc` among the stored ROCs (NaN unless all finite).
    fn percent_rank(&self, roc: f64) -> f64 {
        if !roc.is_finite()
            || self.rocs.len() != self.rankperiod
            || !self.rocs.iter().all(|v| v.is_finite())
        {
            return f64::NAN;
        }
        let count = self.rocs.iter().filter(|&&v| v <= roc).count();
        count as f64 * (100.0 / self.rankperiod as f64)
    }

    /// Commit one bar.
    pub fn update(&mut self, close: f64) -> f64 {
        let (streak, roc) = if self.has_prev {
            let prev = self.prev_close;
            let roc = if prev != 0.0 {
                (close - prev) / prev * 100.0
            } else {
                f64::NAN
            };
            (self.next_streak(close), roc)
        } else {
            (0.0, f64::NAN)
        };
        self.prev_close = close;
        self.has_prev = true;
        self.streak = streak;
        let a = self
            .price
            .update(close)
            .map_or(f64::NAN, |(g, l)| rsi_value(g, l));
        let b = self
            .streak_rsi
            .update(streak)
            .map_or(f64::NAN, |(g, l)| rsi_value(g, l));
        let c = self.percent_rank(roc);
        self.rocs.push(roc, self.rankperiod);
        if a.is_finite() && b.is_finite() && c.is_finite() {
            (a + b + c) / 3.0
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.price.reset();
        self.streak_rsi.reset();
        self.prev_close = f64::NAN;
        self.has_prev = false;
        self.streak = 0.0;
        self.rocs.clear();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.rankperiod, "rankperiod", 1)?;
        self.price.validate()?;
        self.streak_rsi.validate()?;
        self.rocs.validate(self.rankperiod, "crsi rank window")
    }

    pub fn timeperiod(&self) -> usize {
        self.price.period()
    }

    pub fn streakperiod(&self) -> usize {
        self.streak_rsi.period()
    }

    pub fn rankperiod(&self) -> usize {
        self.rankperiod
    }
}

// ---------------------------------------------------------------------------
// Native: TSI
// ---------------------------------------------------------------------------

/// Streaming TSI (`extended::tsi`): plain `overlap::ema` cascades that skip
/// the leading NaN (momentum is NaN on bar 0), `100 * m / a` where both are
/// finite and `a != 0`, then an `overlap::ema` signal of that line.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingTsi {
    mom_long: EmaTa,
    mom_short: EmaTa,
    abs_long: EmaTa,
    abs_short: EmaTa,
    signal: EmaTa,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_close: f64,
    has_prev: bool,
}

impl StreamingTsi {
    pub fn new(
        longperiod: usize,
        shortperiod: usize,
        signalperiod: usize,
    ) -> Result<Self, StreamingError> {
        Ok(Self {
            mom_long: EmaTa::new(longperiod)?,
            mom_short: EmaTa::new(shortperiod)?,
            abs_long: EmaTa::new(longperiod)?,
            abs_short: EmaTa::new(shortperiod)?,
            signal: EmaTa::new(signalperiod)?,
            prev_close: f64::NAN,
            has_prev: false,
        })
    }

    /// Commit one bar: `(tsi, signal)`.
    pub fn update(&mut self, close: f64) -> (f64, f64) {
        let prev = std::mem::replace(&mut self.prev_close, close);
        let mom = if std::mem::replace(&mut self.has_prev, true) {
            close - prev
        } else {
            f64::NAN
        };
        let num = self.mom_short.update(self.mom_long.update(mom));
        let den = self.abs_short.update(self.abs_long.update(mom.abs()));
        let tsi = if num.is_finite() && den.is_finite() && den != 0.0 {
            100.0 * num / den
        } else {
            f64::NAN
        };
        (tsi, self.signal.update(tsi))
    }

    pub fn reset(&mut self) {
        self.mom_long.reset();
        self.mom_short.reset();
        self.abs_long.reset();
        self.abs_short.reset();
        self.signal.reset();
        self.prev_close = f64::NAN;
        self.has_prev = false;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        for ema in [
            &self.mom_long,
            &self.mom_short,
            &self.abs_long,
            &self.abs_short,
            &self.signal,
        ] {
            ema.validate()?;
        }
        if self.mom_long.period() != self.abs_long.period()
            || self.mom_short.period() != self.abs_short.period()
        {
            return Err(StreamingError(
                "tsi numerator and denominator EMAs disagree on the period".to_string(),
            ));
        }
        Ok(())
    }

    pub fn longperiod(&self) -> usize {
        self.mom_long.period()
    }

    pub fn shortperiod(&self) -> usize {
        self.mom_short.period()
    }

    pub fn signalperiod(&self) -> usize {
        self.signal.period()
    }
}

// ---------------------------------------------------------------------------
// Native: STC
// ---------------------------------------------------------------------------

/// One `stochastic_of` bar: push `x`, then `100 * (x - min) / (max - min)`
/// (0 on a flat window) once the window is full and finite. On finite input
/// the batch deque's NaN surfaces exactly while a leading NaN is in the window.
fn stochastic_step(window: &mut Ring, x: f64, period: usize) -> f64 {
    window.push(x, period);
    if !x.is_finite() || window.len() < period || !window.iter().all(|v| v.is_finite()) {
        return f64::NAN;
    }
    let (high, low) = (window.max(), window.min());
    let span = high - low;
    if span != 0.0 {
        100.0 * (x - low) / span
    } else {
        0.0
    }
}

/// Streaming Schaff trend cycle (`extended::stc`): EMA MACD → stochastic →
/// EMA(`d1`) → stochastic → EMA(`d2`), clamped to `[0, 100]`; every EMA is
/// `overlap::ema` (skips the leading NaN of its upstream stage).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingStc {
    cycleperiod: usize,
    fast: EmaTa,
    slow: EmaTa,
    macd_window: Ring,
    smooth1: EmaTa,
    pf_window: Ring,
    smooth2: EmaTa,
}

impl StreamingStc {
    pub fn new(
        fastperiod: usize,
        slowperiod: usize,
        cycleperiod: usize,
        d1: usize,
        d2: usize,
    ) -> Result<Self, StreamingError> {
        check_period(cycleperiod, "cycleperiod", 1)?;
        let fast = EmaTa::new(fastperiod)?;
        let slow = EmaTa::new(slowperiod)?;
        check_fast_below_slow(fastperiod, slowperiod)?;
        Ok(Self {
            cycleperiod,
            fast,
            slow,
            macd_window: Ring::default(),
            smooth1: EmaTa::new(d1)?,
            pf_window: Ring::default(),
            smooth2: EmaTa::new(d2)?,
        })
    }

    /// Commit one bar.
    pub fn update(&mut self, close: f64) -> f64 {
        let macd = finite2(self.fast.update(close), self.slow.update(close), |f, s| {
            f - s
        });
        let stoch1 = stochastic_step(&mut self.macd_window, macd, self.cycleperiod);
        let pf = self.smooth1.update(stoch1);
        let stoch2 = stochastic_step(&mut self.pf_window, pf, self.cycleperiod);
        let v = self.smooth2.update(stoch2);
        if v.is_finite() {
            v.clamp(0.0, 100.0)
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.macd_window.clear();
        self.smooth1.reset();
        self.pf_window.clear();
        self.smooth2.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.cycleperiod, "cycleperiod", 1)?;
        for ema in [&self.fast, &self.slow, &self.smooth1, &self.smooth2] {
            ema.validate()?;
        }
        check_fast_below_slow(self.fast.period(), self.slow.period())?;
        self.macd_window
            .validate(self.cycleperiod, "stc macd window")?;
        self.pf_window
            .validate(self.cycleperiod, "stc smoothed window")
    }

    pub fn fastperiod(&self) -> usize {
        self.fast.period()
    }

    pub fn slowperiod(&self) -> usize {
        self.slow.period()
    }

    pub fn cycleperiod(&self) -> usize {
        self.cycleperiod
    }

    pub fn d1(&self) -> usize {
        self.smooth1.period()
    }

    pub fn d2(&self) -> usize {
        self.smooth2.period()
    }
}

// ---------------------------------------------------------------------------
// Native: RVI
// ---------------------------------------------------------------------------

/// `NanAwareSum` of `extended::oscillators`: a window mean whose running sum
/// keeps NaN out (counting them instead) and is never reseeded.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct NanAwareMean {
    terms: Ring,
    nan_count: usize,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    sum: f64,
}

impl NanAwareMean {
    fn advance(&mut self, x: f64, period: usize) -> f64 {
        if x.is_nan() {
            self.nan_count += 1;
        } else {
            self.sum += x;
        }
        if let Some(old) = self.terms.push(x, period) {
            if old.is_nan() {
                self.nan_count = self.nan_count.saturating_sub(1);
            } else {
                self.sum -= old;
            }
        }
        if self.terms.len() < period || self.nan_count > 0 || !self.sum.is_finite() {
            f64::NAN
        } else {
            self.sum / period as f64
        }
    }

    fn validate(&self, period: usize) -> Result<(), StreamingError> {
        self.terms.validate(period, "rvi window")?;
        let nans = self.terms.iter().filter(|v| v.is_nan()).count();
        if nans != self.nan_count {
            return Err(StreamingError(format!(
                "rvi window holds {nans} NaN terms but counts {}",
                self.nan_count
            )));
        }
        Ok(())
    }
}

/// `rvi_term` from the last four `a - b` differences (newest first).
fn rvi_term(diffs: &Ring) -> f64 {
    if diffs.len() < 4 {
        return f64::NAN;
    }
    diffs.ago(0) + 2.0 * diffs.ago(1) + 2.0 * diffs.ago(2) + diffs.ago(3)
}

/// Streaming RVI (`extended::rvi`): 4-bar weighted `(C - O)` and `(H - L)`
/// terms, their `NanAwareSum` window means (running sums, mirrored rather
/// than recomputed: on a flat window the batch denominator is a rounding
/// residue, not 0), the ratio where the denominator is non-zero, and the
/// `(x + 2 x1 + 2 x2 + x3) / 6` signal.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingRvi {
    timeperiod: usize,
    close_open: Ring,
    high_low: Ring,
    num: NanAwareMean,
    den: NanAwareMean,
    /// The previous three RVI values.
    lines: Ring,
}

impl StreamingRvi {
    const TERM_BARS: usize = 4;
    const SIGNAL_LAGS: usize = 3;

    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            close_open: Ring::default(),
            high_low: Ring::default(),
            num: NanAwareMean::default(),
            den: NanAwareMean::default(),
            lines: Ring::default(),
        })
    }

    /// Commit one bar: `(rvi, signal)`.
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> (f64, f64) {
        self.close_open.push(close - open, Self::TERM_BARS);
        self.high_low.push(high - low, Self::TERM_BARS);
        let num = self
            .num
            .advance(rvi_term(&self.close_open), self.timeperiod);
        let den = self.den.advance(rvi_term(&self.high_low), self.timeperiod);
        let line = if num.is_finite() && den.is_finite() && den != 0.0 {
            num / den
        } else {
            f64::NAN
        };
        let signal = if self.lines.len() == Self::SIGNAL_LAGS {
            let (b, c, d) = (self.lines.ago(0), self.lines.ago(1), self.lines.ago(2));
            if line.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() {
                (line + 2.0 * b + 2.0 * c + d) / 6.0
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        self.lines.push(line, Self::SIGNAL_LAGS);
        (line, signal)
    }

    pub fn reset(&mut self) {
        self.close_open.clear();
        self.high_low.clear();
        self.num = NanAwareMean::default();
        self.den = NanAwareMean::default();
        self.lines.clear();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 1)?;
        self.close_open
            .validate(Self::TERM_BARS, "rvi close-open terms")?;
        self.high_low
            .validate(Self::TERM_BARS, "rvi high-low terms")?;
        self.num.validate(self.timeperiod)?;
        self.den.validate(self.timeperiod)?;
        self.lines.validate(Self::SIGNAL_LAGS, "rvi signal lags")
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }
}
