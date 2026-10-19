//! Streaming building blocks shared by the registry families.
//!
//! Each primitive mirrors one batch kernel's arithmetic *exactly* — the same
//! seed, recurrence form and NaN rule — so a stream composed from them tracks
//! its batch function to rounding (the seed sums are sequential here where the
//! batch kernels may use reassociating SIMD sums). The private `EmaState` in
//! `streaming.rs` predates these and does **not** match the batch EMA; it
//! stays only for the legacy streams.
//!
//! | primitive          | batch kernel                                   |
//! |--------------------|------------------------------------------------|
//! | [`EmaTa`]          | `overlap::ema`                                 |
//! | [`EmaFirstFinite`] | `overlap::ema_from_first_finite`               |
//! | [`EmaCascade`]     | stacked `ema_from_first_finite` (DEMA … TRIX)  |
//! | [`EmaMacd`]        | `overlap::macd` legs, `x*k + prev*(1-k)`       |
//! | [`Wilder`]         | RSI / ATR smoothing `(avg*(p-1)+x)*inv_p`      |
//! | [`Smma`]           | alligator's `smma`, `(prev*(p-1)+x)/p`         |
//! | [`MaStream`]       | `overlap::compute_ma_by_type` (`matype` 0..=8) |
//!
//! Every primitive is `Clone + Debug + serde`, has `update(&mut self, x) ->
//! f64` (NaN until it has a value), `reset` and `validate` (period cap and
//! internal consistency, run on snapshot restore).

use crate::streaming::registry::check_period;
use crate::streaming::StreamingError;

mod ma;
#[cfg(test)]
pub(crate) mod tests;

pub(crate) use ma::{window_ma, KamaCore, MaStream, T3};

/// EMA smoothing factor, as every batch EMA computes it.
#[inline(always)]
pub(crate) fn ema_k(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

/// Sequential sum of the first `period` values: the SMA seed of the
/// recursive averages.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct SeedSum {
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    sum: f64,
    len: usize,
}

impl SeedSum {
    /// Add `x`; returns the mean once the `period`-th value arrives.
    fn push(&mut self, x: f64, period: usize) -> Option<f64> {
        self.sum += x;
        self.len += 1;
        (self.len == period).then(|| self.sum / period as f64)
    }

    fn is_full(&self, period: usize) -> bool {
        self.len >= period
    }

    fn validate(&self, period: usize) -> Result<(), StreamingError> {
        if self.len > period {
            return Err(StreamingError(format!(
                "seed holds {} values but its period is {period}",
                self.len
            )));
        }
        Ok(())
    }
}

/// Shared shape of the SMA-seeded recursive averages: seed state plus the
/// current value (NaN until seeded).
macro_rules! seeded_average {
    ($(#[$doc:meta])* $Ty:ident { $($extra:ident: $ety:ty = $einit:expr),* }) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub(crate) struct $Ty {
            period: usize,
            seed: SeedSum,
            #[cfg_attr(feature = "serde", serde(with = "crate::streaming::snapshot::nan_as_null"))]
            value: f64,
            $($extra: $ety,)*
        }

        // A toolbox: not every primitive has a user for every method yet.
        #[allow(dead_code)]
        impl $Ty {
            pub(crate) fn new(period: usize) -> Result<Self, StreamingError> {
                check_period(period, "timeperiod", 1)?;
                Ok(Self {
                    period,
                    seed: SeedSum::default(),
                    value: f64::NAN,
                    $($extra: $einit,)*
                })
            }

            pub(crate) fn period(&self) -> usize {
                self.period
            }

            pub(crate) fn reset(&mut self) {
                *self = Self {
                    period: self.period,
                    seed: SeedSum::default(),
                    value: f64::NAN,
                    $($extra: $einit,)*
                };
            }

            pub(crate) fn validate(&self) -> Result<(), StreamingError> {
                check_period(self.period, "timeperiod", 1)?;
                self.seed.validate(self.period)
            }

            /// Seed phase: feed `x` to the SMA seed. `Ok(v)` is this bar's
            /// output when still seeding (NaN, or the seed itself); `Err(prev)`
            /// hands the seeded value to the recurrence.
            fn seed_or_prev(&mut self, x: f64) -> Result<f64, f64> {
                if self.seed.is_full(self.period) {
                    return Err(self.value);
                }
                if let Some(mean) = self.seed.push(x, self.period) {
                    self.value = mean;
                }
                Ok(self.value)
            }
        }
    };
}

seeded_average!(
    /// `overlap::ema`: a leading NaN prefix is skipped, the next `period`
    /// values (whatever they are) form the SMA seed, then
    /// `ema_step(prev, x, k)` (fused on native targets); a later NaN
    /// poisons the recurrence for good.
    EmaTa { started: bool = false }
);

impl EmaTa {
    pub(crate) fn update(&mut self, x: f64) -> f64 {
        if !self.started {
            if x.is_nan() {
                return f64::NAN;
            }
            self.started = true;
        }
        match self.seed_or_prev(x) {
            Ok(v) => v,
            Err(prev) => {
                self.value = crate::overlap::ema_step(prev, x, ema_k(self.period));
                self.value
            }
        }
    }
}

seeded_average!(
    /// `overlap::ema_from_first_finite`: seeds from the first run of
    /// `period` consecutive finite values — any non-finite value before the
    /// seed restarts the run — then `ema_step`; after the seed a NaN poisons
    /// the recurrence. Fed NaN while an upstream stage warms up, it reproduces
    /// the batch cascades (DEMA, TEMA, T3, TRIX, PPO signal) exactly.
    EmaFirstFinite {}
);

impl EmaFirstFinite {
    pub(crate) fn update(&mut self, x: f64) -> f64 {
        if !self.seed.is_full(self.period) && !x.is_finite() {
            self.seed = SeedSum::default();
            return f64::NAN;
        }
        match self.seed_or_prev(x) {
            Ok(v) => v,
            Err(prev) => {
                self.value = crate::overlap::ema_step(prev, x, ema_k(self.period));
                self.value
            }
        }
    }
}

seeded_average!(
    /// `overlap::macd`'s legs: SMA seed of the first `period` values, then
    /// the *unfused* `x * k + prev * (1 - k)` (MACD / MACDFIX).
    EmaMacd {}
);

impl EmaMacd {
    pub(crate) fn update(&mut self, x: f64) -> f64 {
        match self.seed_or_prev(x) {
            Ok(v) => v,
            Err(prev) => {
                let k = ema_k(self.period);
                self.value = x * k + prev * (1.0 - k);
                self.value
            }
        }
    }
}

seeded_average!(
    /// Wilder smoothing as in `rsi` / `atr`: SMA seed of the first `period`
    /// values (divided by `period`), then `(avg * (p - 1) + x) * (1 / p)`.
    /// The caller feeds the per-bar quantity (gain, loss, true range) from
    /// the bar the batch kernel starts at.
    Wilder {}
);

impl Wilder {
    pub(crate) fn update(&mut self, x: f64) -> f64 {
        match self.seed_or_prev(x) {
            Ok(v) => v,
            Err(prev) => {
                let p = self.period as f64;
                self.value = (prev * (p - 1.0) + x) * (1.0 / p);
                self.value
            }
        }
    }
}

seeded_average!(
    /// SMMA as in alligator's `smma`: SMA seed, then the *divide* form
    /// `(prev * (p - 1) + x) / p`; `period == 1` passes `x` through.
    #[allow(dead_code)] // consumed by the extended family
    Smma {}
);

#[allow(dead_code)] // consumed by the extended family
impl Smma {
    pub(crate) fn update(&mut self, x: f64) -> f64 {
        if self.period == 1 {
            return x;
        }
        match self.seed_or_prev(x) {
            Ok(v) => v,
            Err(prev) => {
                let p = self.period as f64;
                self.value = (prev * (p - 1.0) + x) / p;
                self.value
            }
        }
    }
}

/// `period` stages of [`EmaFirstFinite`] fed one into the next, as the batch
/// DEMA / TEMA / T3 / TRIX compose `ema_from_first_finite`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct EmaCascade {
    stages: Vec<EmaFirstFinite>,
}

impl EmaCascade {
    /// Most stages any kernel cascades (T3).
    pub(crate) const MAX_DEPTH: usize = 6;

    pub(crate) fn new(period: usize, depth: usize) -> Result<Self, StreamingError> {
        assert!(
            (1..=Self::MAX_DEPTH).contains(&depth),
            "cascade depth {depth}"
        );
        let stage = EmaFirstFinite::new(period)?;
        Ok(Self {
            stages: vec![stage; depth],
        })
    }

    /// The stages' period (`0` for a corrupt, stage-less snapshot, which
    /// `check_period` then rejects).
    pub(crate) fn period(&self) -> usize {
        self.stages.first().map_or(0, EmaFirstFinite::period)
    }

    /// Every stage's value for `x` (`out[k]` = stage `k + 1`; unused slots NaN).
    pub(crate) fn update(&mut self, x: f64) -> [f64; Self::MAX_DEPTH] {
        let mut out = [f64::NAN; Self::MAX_DEPTH];
        let mut v = x;
        for (slot, stage) in out.iter_mut().zip(&mut self.stages) {
            v = stage.update(v);
            *slot = v;
        }
        out
    }

    pub(crate) fn reset(&mut self) {
        self.stages.iter_mut().for_each(EmaFirstFinite::reset);
    }

    /// Stage count must be `depth`, every stage the same period.
    pub(crate) fn validate(&self, depth: usize) -> Result<(), StreamingError> {
        if self.stages.len() != depth {
            return Err(StreamingError(format!(
                "EMA cascade has {} stages, expected {depth}",
                self.stages.len()
            )));
        }
        let period = self.period();
        for stage in &self.stages {
            stage.validate()?;
            if stage.period() != period {
                return Err(StreamingError(
                    "EMA cascade stages disagree on the period".to_string(),
                ));
            }
        }
        Ok(())
    }
}
