//! [`MaStream`]: the streaming `overlap::compute_ma_by_type`, and the
//! window / adaptive averages behind it.

use std::collections::VecDeque;

use super::{EmaCascade, EmaTa};
use crate::rolling::RESEED_INTERVAL;
use crate::streaming::registry::{check_float, check_matype, check_period};
use crate::streaming::{validate_buffer_len, StreamingError};

/// The last `period` (or `period + 1`) values.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Window {
    period: usize,
    values: VecDeque<f64>,
}

impl Window {
    fn new(period: usize) -> Result<Self, StreamingError> {
        check_period(period, "timeperiod", 1)?;
        Ok(Self {
            period,
            values: VecDeque::new(),
        })
    }

    /// Append `x`, keeping at most `cap` values; returns the evicted one.
    fn push(&mut self, x: f64, cap: usize) -> Option<f64> {
        self.values.push_back(x);
        (self.values.len() > cap)
            .then(|| self.values.pop_front())
            .flatten()
    }

    fn validate(&self, cap: usize) -> Result<(), StreamingError> {
        check_period(self.period, "timeperiod", 1)?;
        validate_buffer_len(self.values.len(), cap, "moving-average window")
    }
}

/// `overlap::sma`: sequential seed sum, then `sum += new - old`, `sum / p`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Sma {
    window: Window,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    sum: f64,
}

impl Sma {
    fn update(&mut self, x: f64) -> f64 {
        let p = self.window.period;
        match self.window.push(x, p) {
            Some(old) => self.sum += x - old,
            None => self.sum += x,
        }
        if self.window.values.len() < p {
            return f64::NAN;
        }
        self.sum / p as f64
    }
}

/// `overlap::wma`: weighted sum `t` and plain sum `s` seeded over the first
/// window, then `t += p*x - s; s += x - old`, output `t * (1 / denom)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Wma {
    window: Window,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    t: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    s: f64,
}

impl Wma {
    fn update(&mut self, x: f64) -> f64 {
        let period = self.window.period;
        let p = period as f64;
        let seeding = self.window.values.len() < period;
        let old = self.window.push(x, period);
        if seeding {
            self.t += self.window.values.len() as f64 * x;
            self.s += x;
        } else {
            self.t += p * x - self.s;
            self.s += x - old.unwrap_or(f64::NAN);
        }
        if self.window.values.len() < period {
            return f64::NAN;
        }
        self.t * (1.0 / (period * (period + 1) / 2) as f64)
    }
}

/// `overlap::trima`: the exact triangular dot product over the window (the
/// batch kernel's seed / reseed formula), times `1 / weight_sum`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Trima {
    window: Window,
}

impl Trima {
    fn update(&mut self, x: f64) -> f64 {
        let p = self.window.period;
        self.window.push(x, p);
        if self.window.values.len() < p {
            return f64::NAN;
        }
        let half = p.div_ceil(2);
        let weight = |i: usize| (if i <= half { i } else { p + 1 - i }) as f64;
        let weight_sum: f64 = (1..=p).map(weight).sum();
        let dot: f64 = self
            .window
            .values
            .iter()
            .enumerate()
            .map(|(j, &v)| v * weight(j + 1))
            .sum();
        dot * (1.0 / weight_sum)
    }
}

/// `overlap::kama` (TA-Lib): rolling `|Δclose|` volatility over `p + 1`
/// bars with the subtract-then-add order, exact re-sum every
/// `RESEED_INTERVAL` bars or once a non-finite value leaves the window, the
/// 1e-14 band efficiency ratio on the *signed* `close - close[p]`; first
/// output at bar `p`, seeded from `close[p - 1]`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Kama {
    window: Window,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    value: f64,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    volatility: f64,
    since_reseed: usize,
    non_finite_inside: usize,
    contaminated: bool,
}

impl Kama {
    fn new(period: usize) -> Result<Self, StreamingError> {
        Ok(Self {
            window: Window::new(period)?,
            value: f64::NAN,
            volatility: 0.0,
            since_reseed: 0,
            non_finite_inside: 0,
            contaminated: false,
        })
    }

    fn abs_diff_sum(values: &VecDeque<f64>) -> f64 {
        values
            .iter()
            .zip(values.iter().skip(1))
            .map(|(a, b)| (b - a).abs())
            .sum()
    }

    fn update(&mut self, x: f64) -> f64 {
        let p = self.window.period;
        let full = self.window.values.len() == p + 1;
        if !full {
            self.window.push(x, p + 1);
            if !x.is_finite() {
                self.non_finite_inside += 1;
            }
            if self.window.values.len() <= p {
                return f64::NAN;
            }
            // Bar `p`: the first full `p + 1` window.
            self.volatility = Self::abs_diff_sum(&self.window.values);
            self.contaminated = self.non_finite_inside > 0;
            self.value = self.window.values[p - 1];
        } else {
            let (old, old_next) = (self.window.values[0], self.window.values[1]);
            let last = self.window.values[p];
            self.volatility -= (old - old_next).abs();
            self.volatility += (x - last).abs();
            self.since_reseed += 1;
            if !x.is_finite() {
                self.non_finite_inside += 1;
                self.contaminated = true;
            }
            if !old.is_finite() {
                self.non_finite_inside -= 1;
            }
            self.window.push(x, p + 1);
            if self.since_reseed >= RESEED_INTERVAL
                || (self.contaminated && self.non_finite_inside == 0)
            {
                self.volatility = Self::abs_diff_sum(&self.window.values);
                self.since_reseed = 0;
                self.contaminated = self.non_finite_inside > 0;
            }
        }
        let period_roc = x - self.window.values[0];
        let er = crate::overlap::efficiency_ratio(period_roc, self.volatility);
        let (fast_sc, slow_sc) = (2.0 / 3.0_f64, 2.0 / 31.0_f64);
        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);
        self.value += sc * (x - self.value);
        self.value
    }

    fn validate(&self) -> Result<(), StreamingError> {
        let p = self.window.period;
        self.window.validate(p + 1)?;
        let non_finite = self.window.values.iter().filter(|v| !v.is_finite()).count();
        if non_finite != self.non_finite_inside || self.since_reseed > RESEED_INTERVAL {
            return Err(StreamingError(
                "KAMA counters are inconsistent with its window".to_string(),
            ));
        }
        Ok(())
    }
}

/// `overlap::t3`: six cascaded first-finite EMAs combined as
/// `c1*e6 + c2*e5 + c3*e4 + c4*e3`, emitted once `e3..e6` are finite.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct T3 {
    cascade: EmaCascade,
    vfactor: f64,
}

impl T3 {
    pub(crate) fn new(period: usize, vfactor: f64) -> Result<Self, StreamingError> {
        check_float(vfactor, "vfactor")?;
        Ok(Self {
            cascade: EmaCascade::new(period, 6)?,
            vfactor,
        })
    }

    pub(crate) fn period(&self) -> usize {
        self.cascade.period()
    }

    pub(crate) fn vfactor(&self) -> f64 {
        self.vfactor
    }

    pub(crate) fn reset(&mut self) {
        self.cascade.reset();
    }

    pub(crate) fn update(&mut self, x: f64) -> f64 {
        let e = self.cascade.update(x);
        let v = self.vfactor;
        let c1 = -(v * v * v);
        let c2 = 3.0 * v * v + 3.0 * v * v * v;
        let c3 = -6.0 * v * v - 3.0 * v - 3.0 * v * v * v;
        let c4 = 1.0 + 3.0 * v + v * v * v + 3.0 * v * v;
        if e[2..6].iter().all(|v| v.is_finite()) {
            c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2]
        } else {
            f64::NAN
        }
    }

    pub(crate) fn validate(&self) -> Result<(), StreamingError> {
        check_float(self.vfactor, "vfactor")?;
        self.cascade.validate(6)
    }
}

/// Streaming `overlap::compute_ma_by_type`: 0 SMA, 1 EMA, 2 WMA, 3 DEMA,
/// 4 TEMA, 5 TRIMA, 6 KAMA, 7 and 8 T3 (`vfactor = 0.7`).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum MaStream {
    Sma(Sma),
    Ema(EmaTa),
    Wma(Wma),
    /// `2*e1 - e2` once both are non-NaN.
    Dema(EmaCascade),
    /// `3*e1 - 3*e2 + e3` once all are non-NaN.
    Tema(EmaCascade),
    Trima(Trima),
    Kama(Kama),
    T3(T3),
}

impl MaStream {
    /// T3 volume factor `compute_ma_by_type` uses for `matype` 7 / 8.
    const T3_VFACTOR: f64 = 0.7;

    pub(crate) fn new(period: usize, matype: u8) -> Result<Self, StreamingError> {
        check_period(period, "timeperiod", 1)?;
        check_matype(matype, "matype")?;
        Ok(match matype {
            0 => Self::Sma(Sma {
                window: Window::new(period)?,
                sum: 0.0,
            }),
            1 => Self::Ema(EmaTa::new(period)?),
            2 => Self::Wma(Wma {
                window: Window::new(period)?,
                t: 0.0,
                s: 0.0,
            }),
            3 => Self::Dema(EmaCascade::new(period, 2)?),
            4 => Self::Tema(EmaCascade::new(period, 3)?),
            5 => Self::Trima(Trima {
                window: Window::new(period)?,
            }),
            6 => Self::Kama(Kama::new(period)?),
            _ => Self::T3(T3::new(period, Self::T3_VFACTOR)?),
        })
    }

    pub(crate) fn update(&mut self, x: f64) -> f64 {
        match self {
            Self::Sma(s) => s.update(x),
            Self::Ema(s) => s.update(x),
            Self::Wma(s) => s.update(x),
            Self::Dema(c) => match c.update(x) {
                [e1, e2, ..] if !e1.is_nan() && !e2.is_nan() => 2.0 * e1 - e2,
                _ => f64::NAN,
            },
            Self::Tema(c) => match c.update(x) {
                [e1, e2, e3, ..] if !e1.is_nan() && !e2.is_nan() && !e3.is_nan() => {
                    3.0 * e1 - 3.0 * e2 + e3
                }
                _ => f64::NAN,
            },
            Self::Trima(s) => s.update(x),
            Self::Kama(s) => s.update(x),
            Self::T3(s) => s.update(x),
        }
    }

    pub(crate) fn period(&self) -> usize {
        match self {
            Self::Sma(Sma { window, .. })
            | Self::Wma(Wma { window, .. })
            | Self::Trima(Trima { window })
            | Self::Kama(Kama { window, .. }) => window.period,
            Self::Ema(s) => s.period(),
            Self::Dema(c) | Self::Tema(c) => c.period(),
            Self::T3(s) => s.period(),
        }
    }

    /// The `matype` this stream implements (`7` for T3).
    fn matype(&self) -> u8 {
        match self {
            Self::Sma(_) => 0,
            Self::Ema(_) => 1,
            Self::Wma(_) => 2,
            Self::Dema(_) => 3,
            Self::Tema(_) => 4,
            Self::Trima(_) => 5,
            Self::Kama(_) => 6,
            Self::T3(_) => 7,
        }
    }

    pub(crate) fn reset(&mut self) {
        // Rebuilding from the (validated) parameters cannot fail.
        if let Ok(fresh) = Self::new(self.period(), self.matype()) {
            *self = fresh;
        }
    }

    /// Internal consistency, plus that this is the stream
    /// `MaStream::new(period, matype)` would build — so a snapshot cannot
    /// swap the average type or period under its owner's parameters.
    pub(crate) fn validate_as(&self, period: usize, matype: u8) -> Result<(), StreamingError> {
        check_period(period, "timeperiod", 1)?;
        check_matype(matype, "matype")?;
        let same_type = self.matype() == matype || (self.matype() == 7 && matype == 8);
        if !same_type || self.period() != period {
            return Err(StreamingError(format!(
                "moving-average state does not match period {period}, matype {matype}"
            )));
        }
        match self {
            Self::Sma(Sma { window, .. })
            | Self::Wma(Wma { window, .. })
            | Self::Trima(Trima { window }) => window.validate(period),
            Self::Ema(s) => s.validate(),
            Self::Dema(c) => c.validate(2),
            Self::Tema(c) => c.validate(3),
            Self::Kama(s) => s.validate(),
            Self::T3(s) if s.vfactor == Self::T3_VFACTOR => s.validate(),
            Self::T3(_) => Err(StreamingError(
                "moving-average T3 must use vfactor 0.7".to_string(),
            )),
        }
    }
}
