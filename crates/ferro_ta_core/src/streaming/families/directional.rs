//! The ADX family: `plus_dm`, `minus_dm`, `plus_di`, `minus_di`, `dx`, `adx`
//! and `adxr`, all driven by one [`DmState`] that mirrors the batch kernels
//! `adx_inner` and `dm_only_inner`.

use std::collections::VecDeque;

use crate::streaming::registry::check_period;
use crate::streaming::StreamingError;

/// Registry entries for the directional-movement streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_directional {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                plus_dm => StreamingPlusDm, doc "Streaming +DM, Wilder smoothed; first value at bar `timeperiod - 1` (raw +DM when `timeperiod` is 1).",
                    params(timeperiod: period = 14), inputs(high, low), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::momentum::plus_dm(high, low, timeperiod));
                minus_dm => StreamingMinusDm, doc "Streaming -DM, Wilder smoothed; first value at bar `timeperiod - 1` (raw -DM when `timeperiod` is 1).",
                    params(timeperiod: period = 14), inputs(high, low), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::momentum::minus_dm(high, low, timeperiod));
                plus_di => StreamingPlusDi, doc "Streaming +DI = 100 * smoothed +DM / smoothed TR; first value at bar `timeperiod`.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::momentum::plus_di(high, low, close, timeperiod));
                minus_di => StreamingMinusDi, doc "Streaming -DI = 100 * smoothed -DM / smoothed TR; first value at bar `timeperiod`.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::momentum::minus_di(high, low, close, timeperiod));
                dx => StreamingDx, doc "Streaming directional movement index: 100 * abs(+DI - -DI) / (+DI + -DI); first value at bar `timeperiod`.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    native(lookback = timeperiod + 1, batch = $crate::momentum::dx(high, low, close, timeperiod));
                adx => StreamingAdx, doc "Streaming average directional index; first value at bar `2 * timeperiod - 1`.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    native(lookback = 2 * timeperiod, batch = $crate::momentum::adx(high, low, close, timeperiod));
                adxr => StreamingAdxr, doc "Streaming ADX rating: (ADX + ADX `timeperiod - 1` bars ago) / 2.",
                    params(timeperiod: period = 14), inputs(high, low, close), outputs(out: f64),
                    native(lookback = 3 * timeperiod - 1, batch = $crate::momentum::adxr(high, low, close, timeperiod));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_directional! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Every output of one [`DmState::update`]; NaN where not (yet) defined.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DmStep {
    /// Wilder-smoothed +DM (`momentum::plus_dm`: starts at bar `period - 1`).
    pub plus_dm: f64,
    /// Wilder-smoothed -DM (`momentum::minus_dm`).
    pub minus_dm: f64,
    /// `momentum::plus_di` (starts at bar `period`; NaN when the TR sum is 0).
    pub plus_di: f64,
    /// `momentum::minus_di`.
    pub minus_di: f64,
    /// `momentum::dx`.
    pub dx: f64,
    /// `momentum::adx` (starts at bar `2 * period - 1`).
    pub adx: f64,
}

impl DmStep {
    const NAN: Self = Self {
        plus_dm: f64::NAN,
        minus_dm: f64::NAN,
        plus_di: f64::NAN,
        minus_di: f64::NAN,
        dx: f64::NAN,
        adx: f64::NAN,
    };
}

/// Shared ADX-family state, mirroring the batch kernels bar for bar.
///
/// API for reuse (e.g. a `dmi` stream): [`DmState::new`], [`DmState::update`]
/// (one bar in, a [`DmStep`] with all six outputs out), [`DmState::reset`],
/// [`DmState::validate`], [`DmState::period`], [`DmState::bars`]. It is
/// `Clone + Debug` and serde, so a wrapper just embeds it as a field.
///
/// Two sums families run side by side because the batch kernels seed them
/// differently: the `plus_dm` / `minus_dm` sums seed on the first
/// `period - 1` movements (raw values for `period == 1`), the DI / DX / ADX
/// sums on the first `period`. Both then step as `s * decay + x` with
/// `decay = (p - 1) / p`. ADX seeds with the mean of the first `period` DX
/// values (never emitted at all if one is NaN) and steps as
/// `adx += (1 / p) * (dx - adx)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct DmState {
    period: usize,
    /// Committed bars (saturating).
    bars: usize,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_high: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_low: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_close: f64,
    // `plus_dm` / `minus_dm` smoothing (seeded on `period - 1` movements).
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    dm_plus: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    dm_minus: f64,
    // DI / DX / ADX smoothing (seeded on `period` movements).
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    tr_sum: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    pdm_sum: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    mdm_sum: f64,
    /// Running sum of the first `period` DX values (the ADX seed).
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    dx_sum: f64,
    /// A seed DX was NaN: the batch emits no ADX at all.
    adx_dead: bool,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    adx: f64,
}

impl DmState {
    pub(crate) fn new(period: usize) -> Result<Self, StreamingError> {
        check_period(period, "timeperiod", 1)?;
        Ok(Self::fresh(period))
    }

    /// Unseeded state for an already validated `period`.
    fn fresh(period: usize) -> Self {
        Self {
            period,
            bars: 0,
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            prev_close: f64::NAN,
            dm_plus: 0.0,
            dm_minus: 0.0,
            tr_sum: 0.0,
            pdm_sum: 0.0,
            mdm_sum: 0.0,
            dx_sum: 0.0,
            adx_dead: false,
            adx: f64::NAN,
        }
    }

    pub(crate) fn period(&self) -> usize {
        self.period
    }

    /// Committed bars (saturating).
    pub(crate) fn bars(&self) -> usize {
        self.bars
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::fresh(self.period);
    }

    pub(crate) fn validate(&self) -> Result<(), StreamingError> {
        check_period(self.period, "timeperiod", 1)
    }

    /// Commit one bar. DM-only callers may pass any finite `close`.
    pub(crate) fn update(&mut self, high: f64, low: f64, close: f64) -> DmStep {
        let bar = self.bars;
        self.bars = self.bars.saturating_add(1);
        let (prev_high, prev_low, prev_close) = (self.prev_high, self.prev_low, self.prev_close);
        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;
        if bar == 0 {
            return DmStep::NAN;
        }
        // Movement index `k` (the batch's `i`); this bar is `k + 1`.
        let k = bar - 1;
        let h_diff = high - prev_high;
        let l_diff = prev_low - low;
        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        let pdm = if h_diff > l_diff && h_diff > 0.0 {
            h_diff
        } else {
            0.0
        };
        let mdm = if l_diff > h_diff && l_diff > 0.0 {
            l_diff
        } else {
            0.0
        };
        let (plus_dm, minus_dm) = self.step_dm(k, pdm, mdm);
        let (plus_di, minus_di, dx) = self.step_di(k, tr, pdm, mdm);
        let adx = self.step_adx(bar, dx);
        DmStep {
            plus_dm,
            minus_dm,
            plus_di,
            minus_di,
            dx,
            adx,
        }
    }

    fn decay(&self) -> f64 {
        (self.period - 1) as f64 / self.period as f64
    }

    /// `dm_only_inner`: raw for `period == 1`, else seeded on `period - 1`.
    fn step_dm(&mut self, k: usize, pdm: f64, mdm: f64) -> (f64, f64) {
        if self.period == 1 {
            return (pdm, mdm);
        }
        let seed_len = self.period - 1;
        if k < seed_len {
            self.dm_plus += pdm;
            self.dm_minus += mdm;
            if k + 1 < seed_len {
                return (f64::NAN, f64::NAN);
            }
        } else {
            let decay = self.decay();
            self.dm_plus = self.dm_plus * decay + pdm;
            self.dm_minus = self.dm_minus * decay + mdm;
        }
        (self.dm_plus, self.dm_minus)
    }

    /// `adx_inner`'s DI / DX: seeded on `period` movements; nothing is
    /// written (NaN) while the TR sum is 0.
    fn step_di(&mut self, k: usize, tr: f64, pdm: f64, mdm: f64) -> (f64, f64, f64) {
        if k < self.period {
            self.tr_sum += tr;
            self.pdm_sum += pdm;
            self.mdm_sum += mdm;
            if k + 1 < self.period {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
        } else {
            let decay = self.decay();
            self.tr_sum = self.tr_sum * decay + tr;
            self.pdm_sum = self.pdm_sum * decay + pdm;
            self.mdm_sum = self.mdm_sum * decay + mdm;
        }
        if self.tr_sum == 0.0 {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let plus_di = 100.0 * self.pdm_sum / self.tr_sum;
        let minus_di = 100.0 * self.mdm_sum / self.tr_sum;
        let sum = plus_di + minus_di;
        let dx = if sum != 0.0 {
            100.0 * (plus_di - minus_di).abs() / sum
        } else {
            0.0
        };
        (plus_di, minus_di, dx)
    }

    /// ADX: mean of the DX values at bars `period ..= 2 * period - 1`, then
    /// `adx += (1 / p) * (dx - adx)`.
    fn step_adx(&mut self, bar: usize, dx: f64) -> f64 {
        let first_dx = self.period;
        let seed_bar = self.period.saturating_mul(2) - 1;
        if bar < first_dx {
            return f64::NAN;
        }
        if bar <= seed_bar {
            if dx.is_nan() {
                self.adx_dead = true;
            }
            self.dx_sum += dx;
            if bar < seed_bar || self.adx_dead {
                return f64::NAN;
            }
            self.adx = self.dx_sum / self.period as f64;
            return self.adx;
        }
        if self.adx_dead {
            return f64::NAN;
        }
        let alpha = 1.0 / self.period as f64;
        self.adx += alpha * (dx - self.adx);
        self.adx
    }
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

/// A stream that emits one field of [`DmStep`].
macro_rules! dm_stream {
    ($(#[$doc:meta])* $Ty:ident, $field:ident, |$($arg:ident),+| $bar:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $Ty {
            state: DmState,
        }

        impl $Ty {
            pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
                Ok(Self {
                    state: DmState::new(timeperiod)?,
                })
            }

            /// Commit one bar (NaN during warm-up).
            pub fn update(&mut self, $($arg: f64),+) -> f64 {
                let (high, low, close): (f64, f64, f64) = $bar;
                self.state.update(high, low, close).$field
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
    };
}

dm_stream!(
    /// Streaming +DM (`momentum::plus_dm`).
    StreamingPlusDm, plus_dm, |high, low| (high, low, 0.0)
);
dm_stream!(
    /// Streaming -DM (`momentum::minus_dm`).
    StreamingMinusDm, minus_dm, |high, low| (high, low, 0.0)
);
dm_stream!(
    /// Streaming +DI (`momentum::plus_di`).
    StreamingPlusDi, plus_di, |high, low, close| (high, low, close)
);
dm_stream!(
    /// Streaming -DI (`momentum::minus_di`).
    StreamingMinusDi, minus_di, |high, low, close| (high, low, close)
);
dm_stream!(
    /// Streaming DX (`momentum::dx`).
    StreamingDx, dx, |high, low, close| (high, low, close)
);
dm_stream!(
    /// Streaming ADX (`momentum::adx`).
    StreamingAdx, adx, |high, low, close| (high, low, close)
);

/// Streaming ADXR (`momentum::adxr`): `(adx + adx[timeperiod - 1 bars ago]) / 2`
/// where both exist.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingAdxr {
    state: DmState,
    /// The latest `timeperiod` ADX values (NaN while warming up).
    #[cfg_attr(feature = "serde", serde(with = "crate::streaming::snapshot::nan_seq"))]
    adx_ring: VecDeque<f64>,
}

impl StreamingAdxr {
    pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            state: DmState::new(timeperiod)?,
            adx_ring: VecDeque::new(),
        })
    }

    /// Commit one bar (NaN until ADX has been valid for `timeperiod - 1` bars).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let adx = self.state.update(high, low, close).adx;
        self.adx_ring.push_back(adx);
        if self.adx_ring.len() > self.state.period() {
            self.adx_ring.pop_front();
        }
        match self.adx_ring.front() {
            Some(&past) if self.adx_ring.len() == self.state.period() => {
                if adx.is_nan() || past.is_nan() {
                    f64::NAN
                } else {
                    (adx + past) / 2.0
                }
            }
            _ => f64::NAN,
        }
    }

    pub fn reset(&mut self) {
        self.state.reset();
        self.adx_ring.clear();
    }

    pub fn validate(&self) -> Result<(), StreamingError> {
        self.state.validate()?;
        let expected = self.state.bars().min(self.state.period());
        if self.adx_ring.len() != expected {
            return Err(StreamingError(format!(
                "adxr ring holds {} values after {} bars, expected {expected}",
                self.adx_ring.len(),
                self.state.bars()
            )));
        }
        Ok(())
    }

    pub fn timeperiod(&self) -> usize {
        self.state.period()
    }
}
