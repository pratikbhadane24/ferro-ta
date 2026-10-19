//! Extended volume streams: EMV, VROC, VWMA (windowed); CMF, volume
//! oscillator, relative volume, Chaikin oscillator, smoothed OBV, Force
//! index, NVI / PVI (and their signals), Klinger and PVT (native).

use super::extended_common::{check_fast_below_slow, Ring};
use super::{StreamingAdosc, StreamingMa, StreamingObv};
use crate::streaming::primitives::EmaTa;
use crate::streaming::registry::check_period;
use crate::streaming::StreamingError;

/// Registry entries for the extended volume streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_extended_volume {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                cmf => StreamingCmf, doc "Streaming Chaikin money flow: `timeperiod`-bar sum of CLV * volume over the volume sum. Mirrors the batch running volume sums (not a fresh window sum), so an all-zero-volume window leaves the same residue and NaN mask as the batch function.",
                    params(timeperiod: period = 20), inputs(high, low, close, volume), outputs(out: f64),
                    native(lookback = timeperiod, batch = $crate::extended::cmf(high, low, close, volume, timeperiod));
                emv => StreamingEmv, doc "Streaming ease of movement: SMA over `timeperiod` bars of midpoint move * range * `scale` / volume (raw when `timeperiod` is 1).",
                    params(timeperiod: period = 14, scale: float = 10000.0), inputs(high, low, volume), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::extended::emv(high, low, volume, timeperiod, scale));
                volosc => StreamingVolosc, doc "Streaming volume oscillator: 100 * (SMA(volume, fast) - SMA(volume, slow)) / SMA(volume, slow). Mirrors the batch running volume sums (not a fresh window sum), so an all-zero-volume window leaves the same residue and NaN mask as the batch function.",
                    params(fastperiod: period = 5, slowperiod: period = 10), inputs(volume), outputs(out: f64),
                    native(lookback = slowperiod, batch = $crate::extended::volosc(volume, fastperiod, slowperiod));
                vroc => StreamingVroc, doc "Streaming volume rate of change over `timeperiod` bars, in percent.",
                    params(timeperiod: period = 25), inputs(volume), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::extended::vroc(volume, timeperiod));
                rvol => StreamingRvol, doc "Streaming relative volume: volume / SMA(volume, `timeperiod`). Mirrors the batch running volume sums (not a fresh window sum), so an all-zero-volume window leaves the same residue and NaN mask as the batch function.",
                    params(timeperiod: period = 20), inputs(volume), outputs(out: f64),
                    native(lookback = timeperiod, batch = $crate::extended::rvol(volume, timeperiod));
                vwma => StreamingVwma, doc "Streaming volume-weighted moving average over `timeperiod` bars.",
                    params(timeperiod: period = 20), inputs(close, volume), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::extended::vwma(close, volume, timeperiod));
                cho => StreamingCho, doc "Streaming Chaikin oscillator: fast EMA - slow EMA of the A/D line, both seeded from its first value (same as ADOSC).",
                    params(fastperiod: period = 3, slowperiod: period = 10), inputs(high, low, close, volume), outputs(out: f64),
                    native(lookback = slowperiod, batch = $crate::extended::cho(high, low, close, volume, fastperiod, slowperiod));
                obv_smoothed => StreamingObvSmoothed, doc "Streaming on-balance volume smoothed by a moving average of type `matype`.",
                    params(timeperiod: period = 20, matype: matype = 1), inputs(close, volume), outputs(out: f64),
                    native(lookback = $crate::overlap::ma_lookback(timeperiod, matype) + 1, batch = $crate::extended::obv_smoothed(close, volume, timeperiod, matype));
                force_index => StreamingForceIndex, doc "Streaming force index: EMA of (close - previous close) * volume (raw when `timeperiod` is 1).",
                    params(timeperiod: period = 13), inputs(close, volume), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::extended::force_index(close, volume, timeperiod));
                nvi => StreamingNvi, doc "Streaming negative volume index from 1000: moves with the close only on bars whose volume falls.",
                    params(), inputs(close, volume), outputs(out: f64),
                    native(lookback = 1, batch = $crate::extended::nvi(close, volume));
                pvi => StreamingPvi, doc "Streaming positive volume index from 1000: moves with the close only on bars whose volume rises.",
                    params(), inputs(close, volume), outputs(out: f64),
                    native(lookback = 1, batch = $crate::extended::pvi(close, volume));
                nvi_with_ema => StreamingNviWithEma, doc "Streaming negative volume index and an EMA signal of it.",
                    params(timeperiod: period = 255), inputs(close, volume), outputs(out_nvi: f64, out_signal: f64),
                    native(lookback = timeperiod, batch = $crate::extended::nvi_with_ema(close, volume, timeperiod));
                pvi_with_signal => StreamingPviWithSignal, doc "Streaming positive volume index and a moving-average signal of type `matype`.",
                    params(timeperiod: period = 255, matype: matype = 1), inputs(close, volume), outputs(out_pvi: f64, out_signal: f64),
                    native(lookback = $crate::overlap::ma_lookback(timeperiod, matype) + 1, batch = $crate::extended::pvi_with_signal(close, volume, timeperiod, matype));
                kvo => StreamingKvo, doc "Streaming Klinger volume oscillator: EMA(volume force, fast) - EMA(volume force, slow), plus an EMA signal.",
                    params(fastperiod: period = 34, slowperiod: period = 55, signalperiod: period = 13), inputs(high, low, close, volume), outputs(out_kvo: f64, out_signal: f64),
                    native(lookback = slowperiod + signalperiod, batch = $crate::extended::kvo(high, low, close, volume, fastperiod, slowperiod, signalperiod));
                pvt => StreamingPvt, doc "Streaming price volume trend: cumulative volume * (close - previous close) / previous close, from 0.",
                    params(), inputs(close, volume), outputs(out: f64),
                    native(lookback = 1, batch = $crate::extended::pvt(close, volume));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_extended_volume! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Native: CHO
// ---------------------------------------------------------------------------

/// Streaming Chaikin oscillator (`extended::cho` = `volume::adosc`), with the
/// C ABI's `fastperiod < slowperiod` rule.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingCho {
    inner: StreamingAdosc,
}

impl StreamingCho {
    pub fn new(fastperiod: usize, slowperiod: usize) -> Result<Self, StreamingError> {
        let inner = StreamingAdosc::new(fastperiod, slowperiod)?;
        check_fast_below_slow(fastperiod, slowperiod)?;
        Ok(Self { inner })
    }

    /// Commit one bar (NaN for the first `slowperiod - 1` bars).
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        self.inner.update(high, low, close, volume)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.inner.validate()?;
        check_fast_below_slow(self.inner.fastperiod(), self.inner.slowperiod())
    }

    pub fn fastperiod(&self) -> usize {
        self.inner.fastperiod()
    }

    pub fn slowperiod(&self) -> usize {
        self.inner.slowperiod()
    }
}

// ---------------------------------------------------------------------------
// Native: OBV_SMOOTHED
// ---------------------------------------------------------------------------

/// Streaming smoothed OBV (`extended::obv_smoothed`): `overlap::ma` of
/// `volume::obv`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingObvSmoothed {
    obv: StreamingObv,
    ma: StreamingMa,
}

impl StreamingObvSmoothed {
    pub fn new(timeperiod: usize, matype: u8) -> Result<Self, StreamingError> {
        Ok(Self {
            obv: StreamingObv::new()?,
            ma: StreamingMa::new(timeperiod, matype)?,
        })
    }

    /// Commit one bar (NaN during the `matype`'s warm-up).
    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        let obv = self.obv.update(close, volume);
        self.ma.update(obv)
    }

    pub fn reset(&mut self) {
        self.obv.reset();
        self.ma.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.obv.validate()?;
        self.ma.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.ma.timeperiod()
    }

    pub fn matype(&self) -> u8 {
        self.ma.matype()
    }
}

// ---------------------------------------------------------------------------
// Native: FORCE_INDEX
// ---------------------------------------------------------------------------

/// Streaming force index (`extended::force_index`): raw `(close - prev) *
/// volume` (NaN on bar 0), then `overlap::ema` (skipping that leading NaN)
/// unless `timeperiod <= 1`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingForceIndex {
    ema: EmaTa,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_close: f64,
    has_prev: bool,
}

impl StreamingForceIndex {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            ema: EmaTa::new(timeperiod)?,
            prev_close: f64::NAN,
            has_prev: false,
        })
    }

    /// Commit one bar.
    pub fn update(&mut self, close: f64, volume: f64) -> f64 {
        let prev = std::mem::replace(&mut self.prev_close, close);
        let raw = if std::mem::replace(&mut self.has_prev, true) {
            (close - prev) * volume
        } else {
            f64::NAN
        };
        if self.ema.period() <= 1 {
            raw
        } else {
            self.ema.update(raw)
        }
    }

    pub fn reset(&mut self) {
        self.ema.reset();
        self.prev_close = f64::NAN;
        self.has_prev = false;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.ema.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.ema.period()
    }
}

// ---------------------------------------------------------------------------
// Native: NVI / PVI and their signals
// ---------------------------------------------------------------------------

/// `extended::volume_index`: 1000 on bar 0, then `*= close / prev_close` on
/// bars whose volume falls (NVI) or rises (PVI) and whose previous close is
/// non-zero.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct VolumeIndex {
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    value: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_close: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_volume: f64,
    has_prev: bool,
}

impl VolumeIndex {
    const START: f64 = 1000.0;

    fn new() -> Self {
        Self {
            value: Self::START,
            prev_close: f64::NAN,
            prev_volume: f64::NAN,
            has_prev: false,
        }
    }

    fn update(&mut self, close: f64, volume: f64, positive: bool) -> f64 {
        let prev_close = std::mem::replace(&mut self.prev_close, close);
        let prev_volume = std::mem::replace(&mut self.prev_volume, volume);
        if !std::mem::replace(&mut self.has_prev, true) {
            return self.value;
        }
        let take = if positive {
            volume > prev_volume
        } else {
            volume < prev_volume
        };
        if take && prev_close != 0.0 {
            self.value *= close / prev_close;
        }
        self.value
    }
}

/// One volume-index stream: `$positive` picks PVI over NVI.
macro_rules! volume_index_stream {
    ($(#[$doc:meta])* $Ty:ident, $positive:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $Ty {
            index: VolumeIndex,
        }

        impl $Ty {
            pub fn new() -> Result<Self, StreamingError> {
                Ok(Self {
                    index: VolumeIndex::new(),
                })
            }

            /// Commit one bar; 1000 on the first.
            pub fn update(&mut self, close: f64, volume: f64) -> f64 {
                self.index.update(close, volume, $positive)
            }

            pub fn reset(&mut self) {
                self.index = VolumeIndex::new();
            }

            pub fn validate(&self) -> Result<(), StreamingError> {
                Ok(())
            }
        }
    };
}

volume_index_stream!(
    /// Streaming negative volume index (`extended::nvi`).
    StreamingNvi,
    false
);
volume_index_stream!(
    /// Streaming positive volume index (`extended::pvi`).
    StreamingPvi,
    true
);

/// Streaming NVI with an `overlap::ema` signal (`extended::nvi_with_ema`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingNviWithEma {
    index: VolumeIndex,
    signal: EmaTa,
}

impl StreamingNviWithEma {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            index: VolumeIndex::new(),
            signal: EmaTa::new(timeperiod)?,
        })
    }

    /// Commit one bar: `(nvi, signal)`.
    pub fn update(&mut self, close: f64, volume: f64) -> (f64, f64) {
        let nvi = self.index.update(close, volume, false);
        (nvi, self.signal.update(nvi))
    }

    pub fn reset(&mut self) {
        self.index = VolumeIndex::new();
        self.signal.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.signal.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.signal.period()
    }
}

/// Streaming PVI with an `overlap::ma` signal (`extended::pvi_with_signal`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingPviWithSignal {
    index: VolumeIndex,
    signal: StreamingMa,
}

impl StreamingPviWithSignal {
    pub fn new(timeperiod: usize, matype: u8) -> Result<Self, StreamingError> {
        Ok(Self {
            index: VolumeIndex::new(),
            signal: StreamingMa::new(timeperiod, matype)?,
        })
    }

    /// Commit one bar: `(pvi, signal)`.
    pub fn update(&mut self, close: f64, volume: f64) -> (f64, f64) {
        let pvi = self.index.update(close, volume, true);
        (pvi, self.signal.update(pvi))
    }

    pub fn reset(&mut self) {
        self.index = VolumeIndex::new();
        self.signal.reset();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.signal.validate()
    }

    pub fn timeperiod(&self) -> usize {
        self.signal.timeperiod()
    }

    pub fn matype(&self) -> u8 {
        self.signal.matype()
    }
}

// ---------------------------------------------------------------------------
// Native: KVO
// ---------------------------------------------------------------------------

/// Streaming Klinger oscillator (`extended::kvo`): the volume force from the
/// trend / cumulative-measurement recurrence (NaN on bar 0), `overlap::ema`
/// fast - slow where both are finite, and an `overlap::ema` signal of that.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingKvo {
    fast: EmaTa,
    slow: EmaTa,
    signal: EmaTa,
    /// Committed bars (saturating).
    bars: usize,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    trend: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    cm: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_dm: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_sv: f64,
}

impl StreamingKvo {
    pub fn new(
        fastperiod: usize,
        slowperiod: usize,
        signalperiod: usize,
    ) -> Result<Self, StreamingError> {
        let fast = EmaTa::new(fastperiod)?;
        let slow = EmaTa::new(slowperiod)?;
        check_fast_below_slow(fastperiod, slowperiod)?;
        Ok(Self {
            fast,
            slow,
            signal: EmaTa::new(signalperiod)?,
            bars: 0,
            trend: 0.0,
            cm: 0.0,
            prev_dm: 0.0,
            prev_sv: f64::NAN,
        })
    }

    /// This bar's volume force (NaN on bar 0).
    fn volume_force(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let bar = self.bars;
        self.bars = self.bars.saturating_add(1);
        let dm = high - low;
        let sv = high + low + close;
        let sv_prev = std::mem::replace(&mut self.prev_sv, sv);
        if bar == 0 {
            self.prev_dm = dm;
            return f64::NAN;
        }
        let next_trend = if sv > sv_prev {
            1.0
        } else if sv < sv_prev {
            -1.0
        } else {
            self.trend
        };
        if next_trend == self.trend && bar > 1 {
            self.cm += dm;
        } else {
            self.cm = dm + self.prev_dm;
        }
        let vf = if self.cm != 0.0 {
            volume * (2.0 * dm / self.cm - 1.0).abs() * next_trend * 100.0
        } else {
            0.0
        };
        self.trend = next_trend;
        self.prev_dm = dm;
        vf
    }

    /// Commit one bar: `(kvo, signal)`.
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> (f64, f64) {
        let vf = self.volume_force(high, low, close, volume);
        let (fast, slow) = (self.fast.update(vf), self.slow.update(vf));
        let kvo = if fast.is_finite() && slow.is_finite() {
            fast - slow
        } else {
            f64::NAN
        };
        (kvo, self.signal.update(kvo))
    }

    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.signal.reset();
        self.bars = 0;
        self.trend = 0.0;
        self.cm = 0.0;
        self.prev_dm = 0.0;
        self.prev_sv = f64::NAN;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.fast.validate()?;
        self.slow.validate()?;
        self.signal.validate()?;
        check_fast_below_slow(self.fast.period(), self.slow.period())
    }

    pub fn fastperiod(&self) -> usize {
        self.fast.period()
    }

    pub fn slowperiod(&self) -> usize {
        self.slow.period()
    }

    pub fn signalperiod(&self) -> usize {
        self.signal.period()
    }
}

// ---------------------------------------------------------------------------
// Native: PVT
// ---------------------------------------------------------------------------

/// Streaming price volume trend (`extended::pvt`): 0 on bar 0, then `+=
/// volume * (close - prev) / prev` (`+= 0` when `prev` is 0).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingPvt {
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

impl StreamingPvt {
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
        if std::mem::replace(&mut self.has_prev, true) {
            let delta = if prev != 0.0 {
                volume * (close - prev) / prev
            } else {
                0.0
            };
            self.total += delta;
        }
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
// Natives mirroring the batch running volume sums: CMF, VOLOSC, RVOL
// ---------------------------------------------------------------------------
//
// These kernels carry a window sum from the first full window (`sum += new
// - old`, never reseeded). With fractional volumes an all-zero-volume window
// leaves a rounding residue there, which a fresh window sum would make
// exactly 0 — flipping a `!= 0` test. So the streams keep the batch sums.

/// `overlap::sma`: `simd::sum` seed of the first `period` values (the same
/// reduction, so the same bits), then `sum += new - old`, `sum / p`.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct BatchSma {
    /// The last `period` values.
    window: Ring,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    sum: f64,
}

impl BatchSma {
    fn update(&mut self, x: f64, period: usize) -> f64 {
        let seeding = self.window.len() < period;
        let old = self.window.push(x, period);
        if seeding {
            if self.window.len() < period {
                return f64::NAN;
            }
            let seed: Vec<f64> = self.window.iter().copied().collect();
            self.sum = crate::simd::sum(&seed);
        } else {
            self.sum += x - old.unwrap_or(f64::NAN);
        }
        self.sum / period as f64
    }

    fn validate(&self, period: usize) -> Result<(), StreamingError> {
        self.window.validate(period, "sma window")
    }
}

/// Streaming CMF (`extended::cmf`): sequential seed sums of the first
/// `timeperiod` money-flow volumes and volumes, then `sum += new - old`;
/// `mfv_sum / vol_sum` where the volume sum is non-zero.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingCmf {
    timeperiod: usize,
    mfvs: Ring,
    volumes: Ring,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    mfv_sum: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    vol_sum: f64,
}

impl StreamingCmf {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            mfvs: Ring::default(),
            volumes: Ring::default(),
            mfv_sum: 0.0,
            vol_sum: 0.0,
        })
    }

    /// Commit one bar (NaN for the first `timeperiod - 1` bars).
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let hl = high - low;
        let clv = if hl != 0.0 {
            ((close - low) - (high - close)) / hl
        } else {
            0.0
        };
        let mfv = clv * volume;
        let p = self.timeperiod;
        let old_mfv = self.mfvs.push(mfv, p);
        let old_vol = self.volumes.push(volume, p);
        match (old_mfv, old_vol) {
            (Some(om), Some(ov)) => {
                self.mfv_sum += mfv - om;
                self.vol_sum += volume - ov;
            }
            _ if self.mfvs.len() == p => {
                self.mfv_sum = self.mfvs.iter().sum();
                self.vol_sum = self.volumes.iter().sum();
            }
            _ => return f64::NAN,
        }
        if self.vol_sum != 0.0 {
            self.mfv_sum / self.vol_sum
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.mfvs.clear();
        self.volumes.clear();
        self.mfv_sum = 0.0;
        self.vol_sum = 0.0;
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 1)?;
        self.mfvs
            .validate(self.timeperiod, "cmf money-flow window")?;
        self.volumes
            .validate(self.timeperiod, "cmf volume window")?;
        if self.mfvs.len() != self.volumes.len() {
            return Err(StreamingError(
                "cmf windows disagree on their length".to_string(),
            ));
        }
        Ok(())
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }
}

/// Streaming volume oscillator (`extended::volosc`): two batch-mirrored
/// `overlap::sma`s, `100 * (fast - slow) / slow` where both are finite and
/// `slow != 0`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingVolosc {
    fastperiod: usize,
    slowperiod: usize,
    fast: BatchSma,
    slow: BatchSma,
}

impl StreamingVolosc {
    pub fn new(fastperiod: usize, slowperiod: usize) -> Result<Self, StreamingError> {
        check_period(fastperiod, "fastperiod", 1)?;
        check_period(slowperiod, "slowperiod", 1)?;
        check_fast_below_slow(fastperiod, slowperiod)?;
        Ok(Self {
            fastperiod,
            slowperiod,
            fast: BatchSma::default(),
            slow: BatchSma::default(),
        })
    }

    /// Commit one bar (NaN for the first `slowperiod - 1` bars).
    pub fn update(&mut self, volume: f64) -> f64 {
        let fast = self.fast.update(volume, self.fastperiod);
        let slow = self.slow.update(volume, self.slowperiod);
        if fast.is_finite() && slow.is_finite() && slow != 0.0 {
            100.0 * (fast - slow) / slow
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.fast = BatchSma::default();
        self.slow = BatchSma::default();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.fastperiod, "fastperiod", 1)?;
        check_period(self.slowperiod, "slowperiod", 1)?;
        check_fast_below_slow(self.fastperiod, self.slowperiod)?;
        self.fast.validate(self.fastperiod)?;
        self.slow.validate(self.slowperiod)
    }

    pub fn fastperiod(&self) -> usize {
        self.fastperiod
    }

    pub fn slowperiod(&self) -> usize {
        self.slowperiod
    }
}

/// Streaming relative volume (`extended::rvol`): `volume / sma` with a
/// batch-mirrored `overlap::sma`, where the average is finite and non-zero.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingRvol {
    timeperiod: usize,
    avg: BatchSma,
}

impl StreamingRvol {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        check_period(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            avg: BatchSma::default(),
        })
    }

    /// Commit one bar (NaN for the first `timeperiod - 1` bars).
    pub fn update(&mut self, volume: f64) -> f64 {
        let avg = self.avg.update(volume, self.timeperiod);
        if avg.is_finite() && avg != 0.0 {
            volume / avg
        } else {
            f64::NAN
        }
    }

    pub fn reset(&mut self) {
        self.avg = BatchSma::default();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.timeperiod, "timeperiod", 1)?;
        self.avg.validate(self.timeperiod)
    }

    pub fn timeperiod(&self) -> usize {
        self.timeperiod
    }
}
