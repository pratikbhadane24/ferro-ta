//! Building blocks shared by the `extended_*` families: a capped value ring
//! and the TA-Lib ATR every channel kernel calls.

use std::collections::VecDeque;

use crate::streaming::primitives::Wilder;
use crate::streaming::{validate_buffer_len, StreamingError};
use crate::window_sums::ZeroExactSum;

/// The last few values of a series, oldest first. The owner passes the cap on
/// every push (it is a parameter of the owner, so it is not stored twice).
#[derive(Clone, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub(crate) struct Ring(
    #[cfg_attr(feature = "serde", serde(with = "crate::streaming::snapshot::nan_seq"))]
    VecDeque<f64>,
);

impl Ring {
    /// Append `x`, keeping at most `cap` values; returns the evicted value.
    pub(crate) fn push(&mut self, x: f64, cap: usize) -> Option<f64> {
        self.0.push_back(x);
        if self.0.len() > cap {
            self.0.pop_front()
        } else {
            None
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// The value `k` bars before the latest (`0` = latest); NaN if absent.
    pub(crate) fn ago(&self, k: usize) -> f64 {
        self.0
            .len()
            .checked_sub(k + 1)
            .and_then(|i| self.0.get(i).copied())
            .unwrap_or(f64::NAN)
    }

    /// The oldest value (NaN when empty).
    pub(crate) fn front(&self) -> f64 {
        self.0.front().copied().unwrap_or(f64::NAN)
    }

    /// Holds `cap` values and none is NaN.
    pub(crate) fn is_clean(&self, cap: usize) -> bool {
        self.0.len() == cap && self.0.iter().all(|v| !v.is_nan())
    }

    /// Maximum of the newest `k` values (`-inf` when empty).
    pub(crate) fn max_last(&self, k: usize) -> f64 {
        self.0
            .iter()
            .rev()
            .take(k)
            .fold(f64::NEG_INFINITY, |m, &v| m.max(v))
    }

    /// Minimum of the newest `k` values (`+inf` when empty).
    pub(crate) fn min_last(&self, k: usize) -> f64 {
        self.0
            .iter()
            .rev()
            .take(k)
            .fold(f64::INFINITY, |m, &v| m.min(v))
    }

    pub(crate) fn max(&self) -> f64 {
        self.max_last(self.0.len())
    }

    pub(crate) fn min(&self) -> f64 {
        self.min_last(self.0.len())
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = &f64> {
        self.0.iter()
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    /// At most `cap` values.
    pub(crate) fn validate(&self, cap: usize, name: &str) -> Result<(), StreamingError> {
        validate_buffer_len(self.0.len(), cap, name)
    }
}

/// `volatility::atr`: Wilder smoothing of the true range from bar 1 (SMA
/// seed of `TR[1..=p]`), first value at bar `timeperiod`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Atr {
    wilder: Wilder,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    prev_close: f64,
    has_prev: bool,
}

impl Atr {
    pub(crate) fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            wilder: Wilder::new(timeperiod)?,
            prev_close: f64::NAN,
            has_prev: false,
        })
    }

    /// Commit one bar; NaN for the first `timeperiod` bars.
    pub(crate) fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let prev_close = std::mem::replace(&mut self.prev_close, close);
        if !std::mem::replace(&mut self.has_prev, true) {
            return f64::NAN;
        }
        self.wilder
            .update(crate::volatility::true_range(high, low, prev_close))
    }

    pub(crate) fn period(&self) -> usize {
        self.wilder.period()
    }

    pub(crate) fn reset(&mut self) {
        self.wilder.reset();
        self.prev_close = f64::NAN;
        self.has_prev = false;
    }

    pub(crate) fn validate(&self) -> Result<(), StreamingError> {
        self.wilder.validate()
    }
}

/// `extended`'s `sum_finite`: a running sum over the current run of finite
/// values (`+= x`, then `-= x[i - p]` once the run is longer than `p`),
/// emitted once the run reaches `p`; a non-finite value **restarts** it.
/// The batch kernel carries the sum (and its rounding) across the whole run,
/// so this mirrors it rather than recomputing each window.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct RunSum {
    /// The last `p` values of the current run.
    run: Ring,
    /// Length of the current run (saturating).
    run_len: usize,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    sum: f64,
}

impl RunSum {
    pub(crate) fn update(&mut self, x: f64, period: usize) -> f64 {
        if !x.is_finite() {
            *self = Self::default();
            return f64::NAN;
        }
        self.run_len = self.run_len.saturating_add(1);
        self.sum += x;
        let leaving = self.run.push(x, period);
        if self.run_len > period {
            self.sum -= leaving.unwrap_or(f64::NAN);
        }
        if self.run_len >= period {
            self.sum
        } else {
            f64::NAN
        }
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn validate(&self, period: usize) -> Result<(), StreamingError> {
        self.run.validate(period, "running-sum window")?;
        if self.run.len() != self.run_len.min(period) {
            return Err(StreamingError(format!(
                "running sum holds {} values but its run is {} long",
                self.run.len(),
                self.run_len
            )));
        }
        Ok(())
    }
}

/// `x op y` when both are finite, else NaN (the extended kernels' combine rule).
pub(crate) fn finite2(a: f64, b: f64, f: impl FnOnce(f64, f64) -> f64) -> f64 {
    if a.is_finite() && b.is_finite() {
        f(a, b)
    } else {
        f64::NAN
    }
}

/// `fastperiod < slowperiod`, as the fast/slow extended kernels' C ABI requires.
pub(crate) fn check_fast_below_slow(
    fastperiod: usize,
    slowperiod: usize,
) -> Result<(), StreamingError> {
    if fastperiod >= slowperiod {
        return Err(StreamingError(format!(
            "fastperiod must be < slowperiod, got {fastperiod} >= {slowperiod}"
        )));
    }
    Ok(())
}

/// Sequential (`iter().sum()`) seed of a [`ZeroExactSum`] over a full ring,
/// the batch CMF / MFI seed reduction.
pub(crate) fn seed_sum(ring: &Ring) -> ZeroExactSum {
    let window: Vec<f64> = ring.iter().copied().collect();
    ZeroExactSum::seeded(window.iter().sum(), &window)
}

/// Snapshot check for a [`ZeroExactSum`] mirroring the full `ring` (`cap`
/// values): its non-zero count must match the ring, and an all-zero ring
/// must carry an exact `0.0` sum. A ring still filling is not checked (its
/// sum is reseeded once full).
pub(crate) fn check_zero_exact(
    sum: &ZeroExactSum,
    ring: &Ring,
    cap: usize,
    name: &str,
) -> Result<(), StreamingError> {
    if ring.len() == cap && !sum.matches(ring.iter()) {
        return Err(StreamingError(format!(
            "{name} running sum disagrees with its window's zero values"
        )));
    }
    Ok(())
}
