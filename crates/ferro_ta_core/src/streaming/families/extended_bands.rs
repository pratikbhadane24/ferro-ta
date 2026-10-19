//! Extended channel, band and volatility streams: Donchian, Choppiness,
//! Bollinger %B / bandwidth, historical volatility, Ulcer, rolling median /
//! mode (windowed); Keltner, Chandelier, STARC, median bands, Chande Kroll,
//! MA envelopes, Chaikin volatility and the Mass index (native).

use super::extended_common::{finite2, Atr, Ring, RunSum};
use super::StreamingMa;
use crate::streaming::primitives::{EmaTa, MaStream};
use crate::streaming::registry::{check_float, check_period};
use crate::streaming::StreamingError;

const NAN2: (f64, f64) = (f64::NAN, f64::NAN);

/// Registry entries for the extended band / volatility streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_extended_bands {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                donchian => StreamingDonchian, doc "Streaming Donchian channels: (upper, middle, lower) = rolling highest high, midpoint, lowest low over `timeperiod` bars.",
                    params(timeperiod: period = 20), inputs(high, low), outputs(out_upper: f64, out_middle: f64, out_lower: f64),
                    windowed(lookback = timeperiod, batch = $crate::extended::donchian(high, low, timeperiod));
                choppiness_index => StreamingChoppinessIndex, doc "Streaming Choppiness Index over `timeperiod` bars (first value at bar `timeperiod`).",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::extended::choppiness_index(high, low, close, timeperiod));
                bbpercent => StreamingBbpercent, doc "Streaming Bollinger %B: (close - lower) / (upper - lower) of SMA bands (NaN on a zero-width band). Recomputes each window exactly; on long series the batch function accumulates rounding drift, so values can differ beyond 1e-9 (the stream is the exact one).",
                    params(timeperiod: period = 5, nbdevup: float = 2.0, nbdevdn: float = 2.0), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::extended::bbpercent(close, timeperiod, nbdevup, nbdevdn));
                bbwidth => StreamingBbwidth, doc "Streaming Bollinger bandwidth: (upper - lower) / middle of SMA bands (NaN on a zero middle).",
                    params(timeperiod: period = 5, nbdevup: float = 2.0, nbdevdn: float = 2.0), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::extended::bbwidth(close, timeperiod, nbdevup, nbdevdn));
                historical_volatility => StreamingHistoricalVolatility, doc "Streaming close-to-close historical volatility: population stddev of `timeperiod` log returns, annualised, in percent.",
                    params(timeperiod: period = 20, annual: float = 252.0), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::extended::historical_volatility(close, timeperiod, annual));
                ulcer_index => StreamingUlcerIndex, doc "Streaming Ulcer Index: RMS over `timeperiod` bars of the percent drawdown from the rolling `timeperiod`-bar high (the batch running sum is mirrored, so a rounding-negative mean square is NaN as in the batch function).",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    native(lookback = 2 * timeperiod - 1, batch = $crate::extended::ulcer_index(close, timeperiod));
                median => StreamingMedian, doc "Streaming rolling median over `timeperiod` bars (even windows average the two central values).",
                    params(timeperiod: period = 3), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::extended::median(real, timeperiod));
                mode => StreamingMode, doc "Streaming rolling mode: centre of the most populous of `bins` equal-width bins over `timeperiod` bars.",
                    params(timeperiod: period = 20, bins: period = 10), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::extended::mode(real, timeperiod, bins));
                keltner_channels => StreamingKeltnerChannels, doc "Streaming Keltner channels: (upper, middle, lower) = EMA(close) +/- `multiplier` * ATR(`atr_period`); the middle EMA is emitted on its own once seeded.",
                    params(timeperiod: period = 20, atr_period: period = 10, multiplier: float = 2.0), inputs(high, low, close),
                    outputs(out_upper: f64, out_middle: f64, out_lower: f64),
                    native(lookback = timeperiod.max(atr_period + 1), batch = $crate::extended::keltner_channels(high, low, close, timeperiod, atr_period, multiplier));
                chandelier_exit => StreamingChandelierExit, doc "Streaming Chandelier exit: (highest high - `multiplier` * ATR, lowest low + `multiplier` * ATR) over `timeperiod` bars.",
                    params(timeperiod: period = 22, multiplier: float = 3.0), inputs(high, low, close), outputs(out_long_exit: f64, out_short_exit: f64),
                    native(lookback = timeperiod + 1, batch = $crate::extended::chandelier_exit(high, low, close, timeperiod, multiplier));
                starc => StreamingStarc, doc "Streaming STARC bands: (upper, middle, lower) = SMA(close) +/- `multiplier` * ATR(`atr_period`); the middle SMA is emitted on its own once full.",
                    params(timeperiod: period = 15, atr_period: period = 15, multiplier: float = 2.0), inputs(high, low, close),
                    outputs(out_upper: f64, out_middle: f64, out_lower: f64),
                    native(lookback = timeperiod.max(atr_period + 1), batch = $crate::extended::starc(high, low, close, timeperiod, atr_period, multiplier));
                median_bands => StreamingMedianBands, doc "Streaming median bands: (median, upper, lower, median EMA) from the rolling median of (high + low) / 2, ATR envelopes and an EMA of the median.",
                    params(timeperiod: period = 3, atr_period: period = 14, multiplier: float = 2.0), inputs(high, low, close),
                    outputs(out_median: f64, out_upper: f64, out_lower: f64, out_median_ema: f64),
                    native(lookback = (2 * timeperiod - 1).max(atr_period + 1), batch = $crate::extended::median_bands(high, low, close, timeperiod, atr_period, multiplier));
                chande_kroll_stop => StreamingChandeKrollStop, doc "Streaming Chande Kroll stop: rolling `stop_period` extremes of the ATR-offset `timeperiod` highest high / lowest low.",
                    params(timeperiod: period = 10, multiplier: float = 1.0, stop_period: period = 9), inputs(high, low, close),
                    outputs(out_long_stop: f64, out_short_stop: f64),
                    native(lookback = timeperiod + stop_period, batch = $crate::extended::chande_kroll_stop(high, low, close, timeperiod, multiplier, stop_period));
                ma_envelopes => StreamingMaEnvelopes, doc "Streaming moving-average envelopes: (upper, middle, lower) = MA * (1 +/- `percent` / 100), MA of type `matype`.",
                    params(timeperiod: period = 20, percent: float = 2.5, matype: matype = 0), inputs(close),
                    outputs(out_upper: f64, out_middle: f64, out_lower: f64),
                    native(lookback = $crate::overlap::ma_lookback(timeperiod, matype) + 1, batch = $crate::extended::ma_envelopes(close, timeperiod, percent, matype));
                chaikin_vol => StreamingChaikinVol, doc "Streaming Chaikin volatility: `rocperiod`-bar percent rate of change of an EMA of the high-low range.",
                    params(timeperiod: period = 10, rocperiod: period = 10), inputs(high, low), outputs(out: f64),
                    native(lookback = timeperiod + rocperiod, batch = $crate::extended::chaikin_vol(high, low, timeperiod, rocperiod));
                mass => StreamingMass, doc "Streaming Mass index: `sumperiod`-bar sum of EMA(range) / EMA(EMA(range)); the sum restarts after a non-finite ratio.",
                    params(timeperiod: period = 9, sumperiod: period = 25), inputs(high, low), outputs(out: f64),
                    native(lookback = 2 * timeperiod + sumperiod - 2, batch = $crate::extended::mass(high, low, timeperiod, sumperiod));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_extended_bands! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

/// `mid ± multiplier * atr` when both are finite, else `(NaN, NaN)`.
fn envelope(mid: f64, atr: f64, multiplier: f64) -> (f64, f64) {
    if mid.is_finite() && atr.is_finite() {
        let band = multiplier * atr;
        (mid + band, mid - band)
    } else {
        NAN2
    }
}

// ---------------------------------------------------------------------------
// Native: KELTNER_CHANNELS / STARC (a middle line plus ATR bands)
// ---------------------------------------------------------------------------

/// Streaming Keltner channels (`extended::keltner_channels`): `overlap::ema`
/// middle (emitted alone while the ATR warms up), bands once both exist.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingKeltnerChannels {
    middle: EmaTa,
    atr: Atr,
    multiplier: f64,
}

impl StreamingKeltnerChannels {
    pub fn new(
        timeperiod: usize,
        atr_period: usize,
        multiplier: f64,
    ) -> Result<Self, StreamingError> {
        check_float(multiplier, "multiplier")?;
        Ok(Self {
            middle: EmaTa::new(timeperiod)?,
            atr: Atr::new(atr_period)?,
            multiplier,
        })
    }

    /// Commit one bar: `(upper, middle, lower)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        let middle = self.middle.update(close);
        let atr = self.atr.update(high, low, close);
        let (upper, lower) = envelope(middle, atr, self.multiplier);
        (upper, middle, lower)
    }

    pub fn reset(&mut self) {
        self.middle.reset();
        self.atr.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.multiplier, "multiplier")?;
        self.middle.validate()?;
        self.atr.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.middle.period()
    }

    pub fn atr_period(&self) -> usize {
        self.atr.period()
    }

    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }
}

/// Streaming STARC bands (`extended::starc`): `overlap::sma` middle (emitted
/// alone while the ATR warms up), bands once both exist.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingStarc {
    middle: MaStream,
    atr: Atr,
    multiplier: f64,
}

impl StreamingStarc {
    pub fn new(
        timeperiod: usize,
        atr_period: usize,
        multiplier: f64,
    ) -> Result<Self, StreamingError> {
        check_float(multiplier, "multiplier")?;
        Ok(Self {
            middle: MaStream::new(timeperiod, 0)?,
            atr: Atr::new(atr_period)?,
            multiplier,
        })
    }

    /// Commit one bar: `(upper, middle, lower)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        let middle = self.middle.update(close);
        let atr = self.atr.update(high, low, close);
        let (upper, lower) = envelope(middle, atr, self.multiplier);
        (upper, middle, lower)
    }

    pub fn reset(&mut self) {
        self.middle.reset();
        self.atr.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.multiplier, "multiplier")?;
        self.middle.validate_as(self.middle.period(), 0)?;
        self.atr.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.middle.period()
    }

    pub fn atr_period(&self) -> usize {
        self.atr.period()
    }

    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }
}

// ---------------------------------------------------------------------------
// Native: CHANDELIER_EXIT
// ---------------------------------------------------------------------------

/// Streaming Chandelier exit (`extended::chandelier_exit`): rolling
/// `timeperiod` high / low and ATR(`timeperiod`); first value at bar
/// `timeperiod`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingChandelierExit {
    highs: Ring,
    lows: Ring,
    atr: Atr,
    multiplier: f64,
}

impl StreamingChandelierExit {
    pub fn new(timeperiod: usize, multiplier: f64) -> Result<Self, StreamingError> {
        check_float(multiplier, "multiplier")?;
        Ok(Self {
            highs: Ring::default(),
            lows: Ring::default(),
            atr: Atr::new(timeperiod)?,
            multiplier,
        })
    }

    /// Commit one bar: `(long_exit, short_exit)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        let p = self.atr.period();
        self.highs.push(high, p);
        self.lows.push(low, p);
        let atr = self.atr.update(high, low, close);
        if self.highs.len() < p || atr.is_nan() {
            return NAN2;
        }
        let hh = self.highs.max();
        if hh.is_nan() {
            return NAN2;
        }
        let ll = self.lows.min();
        (hh - self.multiplier * atr, ll + self.multiplier * atr)
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.atr.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.multiplier, "multiplier")?;
        self.atr.validate()?;
        self.highs.validate(self.atr.period(), "chandelier highs")?;
        self.lows.validate(self.atr.period(), "chandelier lows")
    }

    pub fn timeperiod(&self) -> usize {
        self.atr.period()
    }

    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }
}

// ---------------------------------------------------------------------------
// Native: MEDIAN_BANDS
// ---------------------------------------------------------------------------

/// Median of `ring` (even counts average the two central values).
fn ring_median(ring: &Ring) -> f64 {
    let mut sorted: Vec<f64> = ring.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Streaming median bands (`extended::median_bands`): rolling median of
/// `0.5 * (high + low)` (NaN while the window is short or holds a non-finite
/// value), ATR bands around it, and an `overlap::ema` of the median.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMedianBands {
    sources: Ring,
    median_ema: EmaTa,
    atr: Atr,
    multiplier: f64,
}

impl StreamingMedianBands {
    pub fn new(
        timeperiod: usize,
        atr_period: usize,
        multiplier: f64,
    ) -> Result<Self, StreamingError> {
        check_float(multiplier, "multiplier")?;
        Ok(Self {
            sources: Ring::default(),
            median_ema: EmaTa::new(timeperiod)?,
            atr: Atr::new(atr_period)?,
            multiplier,
        })
    }

    /// Commit one bar: `(median, upper, lower, median_ema)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64, f64, f64) {
        let p = self.median_ema.period();
        self.sources.push(0.5 * (high + low), p);
        let full = self.sources.len() == p && self.sources.iter().all(|v| v.is_finite());
        let median = if full {
            ring_median(&self.sources)
        } else {
            f64::NAN
        };
        let atr = self.atr.update(high, low, close);
        let ema = self.median_ema.update(median);
        let (upper, lower) = envelope(median, atr, self.multiplier);
        (median, upper, lower, ema)
    }

    pub fn reset(&mut self) {
        self.sources.clear();
        self.median_ema.reset();
        self.atr.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.multiplier, "multiplier")?;
        self.median_ema.validate()?;
        self.atr.validate()?;
        self.sources
            .validate(self.median_ema.period(), "median window")
    }

    pub fn timeperiod(&self) -> usize {
        self.median_ema.period()
    }

    pub fn atr_period(&self) -> usize {
        self.atr.period()
    }

    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }
}

// ---------------------------------------------------------------------------
// Native: CHANDE_KROLL_STOP
// ---------------------------------------------------------------------------

/// Streaming Chande Kroll stop (`extended::chande_kroll_stop`): first stops
/// `HH(p) - x * ATR(p)` / `LL(p) + x * ATR(p)` from bar `p`, then their
/// rolling `stop_period` max / min over windows free of NaN.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingChandeKrollStop {
    stop_period: usize,
    multiplier: f64,
    highs: Ring,
    lows: Ring,
    atr: Atr,
    first_long: Ring,
    first_short: Ring,
}

impl StreamingChandeKrollStop {
    pub fn new(
        timeperiod: usize,
        multiplier: f64,
        stop_period: usize,
    ) -> Result<Self, StreamingError> {
        check_float(multiplier, "multiplier")?;
        check_period(stop_period, "stop_period", 1)?;
        Ok(Self {
            stop_period,
            multiplier,
            highs: Ring::default(),
            lows: Ring::default(),
            atr: Atr::new(timeperiod)?,
            first_long: Ring::default(),
            first_short: Ring::default(),
        })
    }

    /// Commit one bar: `(long_stop, short_stop)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        let p = self.atr.period();
        self.highs.push(high, p);
        self.lows.push(low, p);
        let atr = self.atr.update(high, low, close);
        let (long, short) = if self.highs.len() == p {
            let (hh, ll) = (self.highs.max(), self.lows.min());
            if hh.is_finite() && ll.is_finite() && atr.is_finite() {
                (hh - self.multiplier * atr, ll + self.multiplier * atr)
            } else {
                NAN2
            }
        } else {
            NAN2
        };
        let q = self.stop_period;
        self.first_long.push(long, q);
        self.first_short.push(short, q);
        let long_stop = if self.first_long.is_clean(q) {
            self.first_long.max()
        } else {
            f64::NAN
        };
        let short_stop = if self.first_short.is_clean(q) {
            self.first_short.min()
        } else {
            f64::NAN
        };
        (long_stop, short_stop)
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.atr.reset();
        self.first_long.clear();
        self.first_short.clear();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.multiplier, "multiplier")?;
        check_period(self.stop_period, "stop_period", 1)?;
        self.atr.validate()?;
        let p = self.atr.period();
        self.highs.validate(p, "chande kroll highs")?;
        self.lows.validate(p, "chande kroll lows")?;
        self.first_long
            .validate(self.stop_period, "chande kroll long stops")?;
        self.first_short
            .validate(self.stop_period, "chande kroll short stops")
    }

    pub fn timeperiod(&self) -> usize {
        self.atr.period()
    }

    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    pub fn stop_period(&self) -> usize {
        self.stop_period
    }
}

// ---------------------------------------------------------------------------
// Native: MA_ENVELOPES
// ---------------------------------------------------------------------------

/// Streaming MA envelopes (`extended::ma_envelopes`): `overlap::ma` middle,
/// `middle * (1 ± percent / 100)` where it is finite.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMaEnvelopes {
    middle: StreamingMa,
    percent: f64,
}

impl StreamingMaEnvelopes {
    pub fn new(timeperiod: usize, percent: f64, matype: u8) -> Result<Self, StreamingError> {
        check_float(percent, "percent")?;
        Ok(Self {
            middle: StreamingMa::new(timeperiod, matype)?,
            percent,
        })
    }

    /// Commit one bar: `(upper, middle, lower)`.
    pub fn update(&mut self, close: f64) -> (f64, f64, f64) {
        let m = self.middle.update(close);
        if !m.is_finite() {
            return (f64::NAN, m, f64::NAN);
        }
        let up = 1.0 + self.percent / 100.0;
        let dn = 1.0 - self.percent / 100.0;
        (m * up, m, m * dn)
    }

    pub fn reset(&mut self) {
        self.middle.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.percent, "percent")?;
        self.middle.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.middle.timeperiod()
    }

    pub fn percent(&self) -> f64 {
        self.percent
    }

    pub fn matype(&self) -> u8 {
        self.middle.matype()
    }
}

// ---------------------------------------------------------------------------
// Native: CHAIKIN_VOL
// ---------------------------------------------------------------------------

/// Streaming Chaikin volatility (`extended::chaikin_vol`): `overlap::ema` of
/// `high - low`, then `momentum::roc` of that EMA over `rocperiod` bars.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingChaikinVol {
    rocperiod: usize,
    ema: EmaTa,
    /// The last `rocperiod + 1` EMA values.
    history: Ring,
}

impl StreamingChaikinVol {
    pub fn new(timeperiod: usize, rocperiod: usize) -> Result<Self, StreamingError> {
        check_period(rocperiod, "rocperiod", 1)?;
        Ok(Self {
            rocperiod,
            ema: EmaTa::new(timeperiod)?,
            history: Ring::default(),
        })
    }

    /// Commit one bar (NaN until `rocperiod` bars after the EMA seeds).
    pub fn update(&mut self, high: f64, low: f64) -> f64 {
        let ema = self.ema.update(high - low);
        self.history.push(ema, self.rocperiod.saturating_add(1));
        if self.history.len() <= self.rocperiod {
            return f64::NAN;
        }
        let prev = self.history.front();
        if prev != 0.0 {
            (ema - prev) / prev * 100.0
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.ema.reset();
        self.history.clear();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.rocperiod, "rocperiod", 1)?;
        self.ema.validate()?;
        self.history
            .validate(self.rocperiod + 1, "chaikin volatility history")
    }

    pub fn timeperiod(&self) -> usize {
        self.ema.period()
    }

    pub fn rocperiod(&self) -> usize {
        self.rocperiod
    }
}

// ---------------------------------------------------------------------------
// Native: MASS
// ---------------------------------------------------------------------------

/// Streaming Mass index (`extended::mass`): `single = ema(range)`, `double =
/// ema(single)`, `ratio = single / double`, then `sum_finite` ([`RunSum`]),
/// which restarts on a non-finite ratio.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMass {
    sumperiod: usize,
    single: EmaTa,
    double: EmaTa,
    sum: RunSum,
}

impl StreamingMass {
    pub fn new(timeperiod: usize, sumperiod: usize) -> Result<Self, StreamingError> {
        check_period(sumperiod, "sumperiod", 1)?;
        Ok(Self {
            sumperiod,
            single: EmaTa::new(timeperiod)?,
            double: EmaTa::new(timeperiod)?,
            sum: RunSum::default(),
        })
    }

    /// Commit one bar.
    pub fn update(&mut self, high: f64, low: f64) -> f64 {
        let single = self.single.update(high - low);
        let double = self.double.update(single);
        let ratio = finite2(
            single,
            double,
            |s, d| if d != 0.0 { s / d } else { f64::NAN },
        );
        self.sum.update(ratio, self.sumperiod)
    }

    pub fn reset(&mut self) {
        self.single.reset();
        self.double.reset();
        self.sum.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.sumperiod, "sumperiod", 1)?;
        self.single.validate()?;
        self.double.validate()?;
        if self.single.period() != self.double.period() {
            return Err(StreamingError(
                "mass EMAs disagree on the period".to_string(),
            ));
        }
        self.sum.validate(self.sumperiod)
    }

    pub fn timeperiod(&self) -> usize {
        self.single.period()
    }

    pub fn sumperiod(&self) -> usize {
        self.sumperiod
    }
}

// ---------------------------------------------------------------------------
// Native: ULCER_INDEX
// ---------------------------------------------------------------------------

/// Streaming Ulcer Index (`extended::ulcer_index`): squared percent drawdown
/// from the rolling `timeperiod` high, `sum_finite` ([`RunSum`]) over
/// `timeperiod` of those, `sqrt(sum / p)`. Native, not windowed: the batch
/// sum runs from the first drawdown, and on a streak of new highs its
/// rounding residue can go negative, where `sqrt` yields NaN — a fresh window
/// sum would be exactly 0 instead.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingUlcerIndex {
    timeperiod: usize,
    closes: Ring,
    sum: RunSum,
}

impl StreamingUlcerIndex {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            closes: Ring::default(),
            sum: RunSum::default(),
        })
    }

    /// Commit one bar (NaN for the first `2 * timeperiod - 2` bars).
    pub fn update(&mut self, close: f64) -> f64 {
        let p = self.timeperiod;
        self.closes.push(close, p);
        let peak = if self.closes.len() == p {
            self.closes.max()
        } else {
            f64::NAN
        };
        let pd2 = if peak.is_finite() && peak != 0.0 && close.is_finite() {
            let pd = 100.0 * (close - peak) / peak;
            pd * pd
        } else {
            f64::NAN
        };
        let total = self.sum.update(pd2, p);
        if total.is_finite() {
            (total / p as f64).sqrt()
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.closes.clear();
        self.sum.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 1)?;
        self.closes.validate(self.timeperiod, "ulcer window")?;
        self.sum.validate(self.timeperiod)
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }
}
