//! Hilbert-transform cycle streams (`ht_trendline`, `ht_dcperiod`,
//! `ht_dcphase`, `ht_phasor`, `ht_sine`, `ht_trendmode`, all driven by one
//! [`HtCore`]) and MAMA ([`MamaCore`]), mirroring `cycle::compute_ht_core`
//! and `overlap::mama` bar for bar.
//!
//! These are this crate's simplified Ehlers kernels, not TA-Lib's. The two
//! batch kernels do not share code: they run the same Hilbert front end
//! (4-bar smoothing, detrender, I1/Q1, jI/jQ, I2/Q2, Re/Im) but differ in the
//! period recursion — `compute_ht_core` starts from a zero period and clamps
//! relative to it only once it is positive, `mama` floors the previous period
//! at 1 — so only the front end ([`HilbertFront`]) is shared here and each
//! core keeps its own period step.

use std::f64::consts::PI;

use super::extended_common::Ring;
use crate::streaming::registry::check_float;
use crate::streaming::StreamingError;

/// Registry entries for the Hilbert-transform streams and MAMA.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_cycle {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                ht_trendline => StreamingHtTrendline, doc "Streaming Hilbert-transform instantaneous trendline: mean of the 4-bar smoothed price over the rounded dominant cycle period; first value at bar 63.",
                    params(), inputs(close), outputs(out: f64),
                    native(lookback = $crate::cycle::HT_LOOKBACK + 1, batch = $crate::cycle::ht_trendline(close));
                ht_dcperiod => StreamingHtDcperiod, doc "Streaming Hilbert-transform dominant cycle period in bars; first value at bar 63.",
                    params(), inputs(close), outputs(out: f64),
                    native(lookback = $crate::cycle::HT_LOOKBACK + 1, batch = $crate::cycle::ht_dcperiod(close));
                ht_dcphase => StreamingHtDcphase, doc "Streaming Hilbert-transform dominant cycle phase in degrees; first value at bar 63.",
                    params(), inputs(close), outputs(out: f64),
                    native(lookback = $crate::cycle::HT_LOOKBACK + 1, batch = $crate::cycle::ht_dcphase(close));
                ht_phasor => StreamingHtPhasor, doc "Streaming Hilbert-transform phasor components (inphase, quadrature); first value at bar 63.",
                    params(), inputs(close), outputs(out_inphase: f64, out_quadrature: f64),
                    native(lookback = $crate::cycle::HT_LOOKBACK + 1, batch = $crate::cycle::ht_phasor(close));
                ht_sine => StreamingHtSine, doc "Streaming Hilbert-transform sine wave (sine, leadsine = sine 45 degrees ahead); first value at bar 63.",
                    params(), inputs(close), outputs(out_sine: f64, out_leadsine: f64),
                    native(lookback = $crate::cycle::HT_LOOKBACK + 1, batch = $crate::cycle::ht_sine(close));
                ht_trendmode => StreamingHtTrendmode, doc "Streaming Hilbert-transform trend mode: 1 while the smoothed dominant cycle period is below 20 bars, else 0; 0 before bar 63.",
                    params(), inputs(close), outputs(out: i32),
                    native(lookback = $crate::cycle::HT_LOOKBACK + 1, batch = $crate::cycle::ht_trendmode(close));
                mama => StreamingMama, doc "Streaming MESA adaptive moving average (mama, fama); first value at bar 32. Requires `slowlimit <= fastlimit`.",
                    params(fastlimit: float = 0.5, slowlimit: float = 0.05), inputs(close), outputs(out_mama: f64, out_fama: f64),
                    native(lookback = 33, batch = $crate::overlap::mama(close, fastlimit, slowlimit));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_cycle! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

/// First bar `overlap::mama` emits (its private `lookback`; the entry's
/// test lookback is this + 1).
const MAMA_LOOKBACK: usize = 32;
/// Depth of the Hilbert FIR (`x[i]`, `x[i - 2]`, `x[i - 4]`, `x[i - 6]`).
const FIR_DEPTH: usize = 7;
/// Closes before the current one in the 4-bar smoothing.
const SMOOTH_TAPS: usize = 3;
/// Smoothed prices kept for the trendline mean. The smoothed period is an
/// EMA of values clamped to `[6, 50]`, so its rounding never exceeds 50.
const TRENDLINE_DEPTH: usize = 50;
/// First bar the Hilbert front end runs (it reads `smooth[i - 6]`).
const FRONT_START: usize = 6;

/// `x` lagged `k` bars, 0 before the series starts (the batch arrays are
/// zero-initialised).
fn lag(r: &Ring, k: usize) -> f64 {
    if k < r.len() {
        r.ago(k)
    } else {
        0.0
    }
}

/// The batch kernels' Hilbert FIR over the newest seven values of `r`.
fn hilbert(r: &Ring, alpha: f64) -> f64 {
    (0.0962 * lag(r, 0) + 0.5769 * lag(r, 2) - 0.5769 * lag(r, 4) - 0.0962 * lag(r, 6)) * alpha
}

/// Phase of `(i1, q1)` in degrees, as both batch kernels compute it.
fn phase_deg(i1: f64, q1: f64) -> f64 {
    if i1 != 0.0 {
        q1.atan2(i1) * 180.0 / PI
    } else if q1 > 0.0 {
        90.0
    } else if q1 < 0.0 {
        -90.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Shared Hilbert front end
// ---------------------------------------------------------------------------

/// One bar of the front end (from bar 6 on).
#[derive(Clone, Copy, Debug)]
struct Phasor {
    i1: f64,
    q1: f64,
    re: f64,
    im: f64,
}

/// The Hilbert front end both batch kernels run: smoothing, detrender, I1/Q1,
/// jI/jQ, I2/Q2 and Re/Im. The period-dependent `alpha` comes from the owner.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct HilbertFront {
    /// Committed bars (saturating).
    bars: usize,
    /// The last [`SMOOTH_TAPS`] closes.
    closes: Ring,
    smooth: Ring,
    detrender: Ring,
    i1: Ring,
    q1: Ring,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    i2: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    q2: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    re: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    im: f64,
}

impl HilbertFront {
    /// Bar index of the next [`HilbertFront::smooth_next`].
    fn bar(&self) -> usize {
        self.bars
    }

    /// Commit `close` to the smoother; returns `(bar, smooth)`.
    fn smooth_next(&mut self, close: f64) -> (usize, f64) {
        let bar = self.bars;
        self.bars = self.bars.saturating_add(1);
        let smooth = if bar >= SMOOTH_TAPS {
            (4.0 * close + 3.0 * self.closes.ago(0) + 2.0 * self.closes.ago(1) + self.closes.ago(2))
                / 10.0
        } else {
            close
        };
        self.closes.push(close, SMOOTH_TAPS);
        self.smooth.push(smooth, FIR_DEPTH);
        (bar, smooth)
    }

    /// The rest of the front end for bar `bar >= 6`.
    fn transform(&mut self, bar: usize, alpha: f64) -> Phasor {
        let detrender = hilbert(&self.smooth, alpha);
        self.detrender.push(detrender, FIR_DEPTH);
        let q1 = if bar >= 12 {
            hilbert(&self.detrender, alpha)
        } else {
            0.0
        };
        let i1 = if bar >= 9 {
            lag(&self.detrender, 3)
        } else {
            0.0
        };
        self.q1.push(q1, FIR_DEPTH);
        self.i1.push(i1, FIR_DEPTH);
        let ji = if bar >= 15 {
            hilbert(&self.i1, alpha)
        } else {
            0.0
        };
        let jq = if bar >= 18 {
            hilbert(&self.q1, alpha)
        } else {
            0.0
        };
        let (i2_prev, q2_prev) = (self.i2, self.q2);
        self.i2 = 0.2 * (i1 - jq) + 0.8 * i2_prev;
        self.q2 = 0.2 * (q1 + ji) + 0.8 * q2_prev;
        let re_raw = self.i2 * i2_prev + self.q2 * q2_prev;
        let im_raw = self.i2 * q2_prev - self.q2 * i2_prev;
        self.re = 0.2 * re_raw + 0.8 * self.re;
        self.im = 0.2 * im_raw + 0.8 * self.im;
        Phasor {
            i1,
            q1,
            re: self.re,
            im: self.im,
        }
    }

    fn validate(&self) -> Result<(), StreamingError> {
        self.closes.validate(SMOOTH_TAPS, "closes")?;
        for (ring, name) in [
            (&self.smooth, "smooth"),
            (&self.detrender, "detrender"),
            (&self.i1, "i1"),
            (&self.q1, "q1"),
        ] {
            ring.validate(FIR_DEPTH, name)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HT core
// ---------------------------------------------------------------------------

/// Every output of one [`HtCore::update`]; NaN (trend mode 0) before bar 63.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HtStep {
    pub trendline: f64,
    pub dc_period: f64,
    pub dc_phase: f64,
    pub inphase: f64,
    pub quadrature: f64,
    pub trend_mode: i32,
}

impl HtStep {
    const WARMUP: Self = Self {
        trendline: f64::NAN,
        dc_period: f64::NAN,
        dc_phase: f64::NAN,
        inphase: f64::NAN,
        quadrature: f64::NAN,
        trend_mode: 0,
    };
}

/// `cycle::compute_ht_core` as a stream: the [`HilbertFront`], the period
/// and smoothed-period recursions, and the last 50 smoothed prices for the
/// trendline.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct HtCore {
    front: HilbertFront,
    /// Smoothed prices for the trendline mean (newest last).
    trend_smooth: Ring,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    period: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    smooth_period: f64,
}

impl HtCore {
    pub(crate) fn update(&mut self, close: f64) -> HtStep {
        let (bar, smooth) = self.front.smooth_next(close);
        self.trend_smooth.push(smooth, TRENDLINE_DEPTH);
        if bar < FRONT_START {
            return HtStep::WARMUP;
        }
        let prev_period = self.period;
        let alpha = 0.075 * prev_period + 0.54;
        let ph = self.front.transform(bar, alpha);
        let mut p = if ph.re != 0.0 && ph.im != 0.0 && ph.re > 0.0 {
            2.0 * PI / (ph.im / ph.re).atan()
        } else {
            prev_period
        };
        if prev_period > 0.0 {
            if p > 1.5 * prev_period {
                p = 1.5 * prev_period;
            }
            if p < 0.67 * prev_period {
                p = 0.67 * prev_period;
            }
        }
        p = p.clamp(6.0, 50.0);
        self.period = 0.2 * p + 0.8 * prev_period;
        self.smooth_period = 0.33 * self.period + 0.67 * self.smooth_period;
        if bar < crate::cycle::HT_LOOKBACK {
            return HtStep::WARMUP;
        }
        let sp = self.smooth_period;
        HtStep {
            trendline: self.trendline(bar, sp),
            dc_period: sp,
            dc_phase: phase_deg(ph.i1, ph.q1),
            inphase: ph.i1,
            quadrature: ph.q1,
            trend_mode: if sp < 20.0 { 1 } else { 0 },
        }
    }

    /// Mean of the newest `round(sp)` smoothed prices (newest first, as the
    /// batch sums them). The ring bound never binds (see [`TRENDLINE_DEPTH`]).
    fn trendline(&self, bar: usize, sp: f64) -> f64 {
        let dc = (sp.round() as usize)
            .max(1)
            .min(bar + 1)
            .min(self.trend_smooth.len());
        let sum: f64 = self.trend_smooth.iter().rev().take(dc).copied().sum();
        sum / dc as f64
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn validate(&self) -> Result<(), StreamingError> {
        self.front.validate()?;
        self.trend_smooth.validate(TRENDLINE_DEPTH, "trend_smooth")
    }
}

/// A parameterless stream that emits a projection of [`HtStep`].
macro_rules! ht_stream {
    ($(#[$doc:meta])* $Ty:ident -> $Out:ty, |$s:ident| $out:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug, Default)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $Ty {
            core: HtCore,
        }

        impl $Ty {
            pub fn new() -> Result<Self, StreamingError> {
                Ok(Self::default())
            }

            /// Commit one close (warm-up value before bar 63).
            pub fn update(&mut self, close: f64) -> $Out {
                let $s = self.core.update(close);
                $out
            }

            pub fn reset(&mut self) {
                self.core.reset();
            }

            pub fn validate(&self) -> Result<(), StreamingError> {
                self.core.validate()
            }
        }
    };
}

ht_stream!(
    /// Streaming HT trendline (`cycle::ht_trendline`).
    StreamingHtTrendline -> f64, |s| s.trendline
);
ht_stream!(
    /// Streaming HT dominant cycle period (`cycle::ht_dcperiod`).
    StreamingHtDcperiod -> f64, |s| s.dc_period
);
ht_stream!(
    /// Streaming HT dominant cycle phase (`cycle::ht_dcphase`).
    StreamingHtDcphase -> f64, |s| s.dc_phase
);
ht_stream!(
    /// Streaming HT phasor (`cycle::ht_phasor`): `(inphase, quadrature)`.
    StreamingHtPhasor -> (f64, f64), |s| (s.inphase, s.quadrature)
);
ht_stream!(
    /// Streaming HT sine wave (`cycle::ht_sine`): `(sine, leadsine)`.
    StreamingHtSine -> (f64, f64), |s| sine_pair(s.dc_phase)
);
ht_stream!(
    /// Streaming HT trend mode (`cycle::ht_trendmode`).
    StreamingHtTrendmode -> i32, |s| s.trend_mode
);

/// `cycle::ht_sine` of one phase (degrees): `(sin, sin + 45 degrees)`.
fn sine_pair(dc_phase: f64) -> (f64, f64) {
    if dc_phase.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let phase_rad = dc_phase * PI / 180.0;
    (phase_rad.sin(), (phase_rad + PI / 4.0).sin())
}

// ---------------------------------------------------------------------------
// MAMA
// ---------------------------------------------------------------------------

/// `overlap::mama` as a stream: the [`HilbertFront`], its own period
/// recursion (previous period floored at 1), the previous phase, and the
/// MAMA / FAMA values (reset to the close until bar 32, then
/// `x * a + prev * (1 - a)`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingMama {
    fastlimit: f64,
    slowlimit: f64,
    front: HilbertFront,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    period: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    phase: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    mama: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    fama: f64,
}

/// `slowlimit <= fastlimit`: the batch kernel clamps the adaptive alpha to
/// `[slowlimit, fastlimit]`, which panics on an inverted range.
fn check_limits(fastlimit: f64, slowlimit: f64) -> Result<(), StreamingError> {
    check_float(fastlimit, "fastlimit")?;
    check_float(slowlimit, "slowlimit")?;
    if slowlimit > fastlimit {
        return Err(StreamingError(format!(
            "mama: slowlimit ({slowlimit}) must be <= fastlimit ({fastlimit})"
        )));
    }
    Ok(())
}

impl StreamingMama {
    pub fn new(fastlimit: f64, slowlimit: f64) -> Result<Self, StreamingError> {
        check_limits(fastlimit, slowlimit)?;
        Ok(Self::fresh(fastlimit, slowlimit))
    }

    fn fresh(fastlimit: f64, slowlimit: f64) -> Self {
        Self {
            fastlimit,
            slowlimit,
            front: HilbertFront::default(),
            period: 0.0,
            phase: 0.0,
            mama: f64::NAN,
            fama: f64::NAN,
        }
    }

    /// Commit one close; `(NaN, NaN)` before bar 32.
    pub fn update(&mut self, close: f64) -> (f64, f64) {
        if self.front.bar() == 0 {
            self.mama = close;
            self.fama = close;
        }
        let (bar, _) = self.front.smooth_next(close);
        if bar < FRONT_START {
            return (f64::NAN, f64::NAN);
        }
        let prev_period = self.period.max(1.0);
        let alpha = 0.075 * prev_period + 0.54;
        let ph = self.front.transform(bar, alpha);
        let mut p = if ph.re != 0.0 && ph.im != 0.0 && ph.re > 0.0 {
            PI * 2.0 / (ph.im / ph.re).atan()
        } else {
            prev_period
        };
        p = p
            .clamp(0.67 * prev_period, 1.5 * prev_period)
            .clamp(6.0, 50.0);
        self.period = 0.2 * p + 0.8 * prev_period;
        let phase = phase_deg(ph.i1, ph.q1);
        let mut delta_phase = self.phase - phase;
        self.phase = phase;
        if delta_phase < 1.0 {
            delta_phase = 1.0;
        }
        let adaptive = (self.fastlimit / delta_phase).clamp(self.slowlimit, self.fastlimit);
        if bar < MAMA_LOOKBACK {
            self.mama = close;
            self.fama = close;
            return (f64::NAN, f64::NAN);
        }
        self.mama = adaptive * close + (1.0 - adaptive) * self.mama;
        self.fama = 0.5 * adaptive * self.mama + (1.0 - 0.5 * adaptive) * self.fama;
        (self.mama, self.fama)
    }

    pub fn reset(&mut self) {
        *self = Self::fresh(self.fastlimit, self.slowlimit);
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        check_limits(self.fastlimit, self.slowlimit)?;
        self.front.validate()
    }

    pub fn fastlimit(&self) -> f64 {
        self.fastlimit
    }

    pub fn slowlimit(&self) -> f64 {
        self.slowlimit
    }
}
