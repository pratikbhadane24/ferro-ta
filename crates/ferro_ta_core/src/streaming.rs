//! Streaming / Incremental Indicators — bar-by-bar stateful structs.
//!
//! Pure Rust implementations with no PyO3 dependency.  Each struct:
//! - Accepts one value per call to `update()`.
//! - Returns `NaN` (or a NaN tuple) during the warm-up window.
//! - Exposes a `reset()` method to restart from scratch.
//! - Has a `period()` accessor (where applicable).

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Validation error for streaming indicator parameters.
#[derive(Debug, Clone)]
pub struct StreamingError(pub String);

impl std::fmt::Display for StreamingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StreamingError {}

fn validate_timeperiod(value: usize, name: &str, minimum: usize) -> Result<(), StreamingError> {
    if value < minimum {
        return Err(StreamingError(format!(
            "{} must be >= {}, got {}",
            name, minimum, value
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helper: EMA state (used inside composite classes)
// ---------------------------------------------------------------------------

/// SMA-seeded EMA state machine.  Not exposed directly — used by
/// `StreamingEMA`, `StreamingMACD`, etc.
pub(crate) struct EmaState {
    period: usize,
    alpha: f64,
    ema: f64,
    seed_buf: Vec<f64>,
    seeded: bool,
}

impl EmaState {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            alpha: 2.0 / (period as f64 + 1.0),
            ema: 0.0,
            seed_buf: Vec::with_capacity(period),
            seeded: false,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        if !self.seeded {
            self.seed_buf.push(value);
            if self.seed_buf.len() < self.period {
                return f64::NAN;
            }
            let seed = self.seed_buf.iter().sum::<f64>() / self.period as f64;
            self.ema = seed;
            self.seeded = true;
            return seed;
        }
        self.ema += self.alpha * (value - self.ema);
        self.ema
    }

    pub fn reset(&mut self) {
        self.ema = 0.0;
        self.seed_buf.clear();
        self.seeded = false;
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

// ---------------------------------------------------------------------------
// Internal helper: ATR state (Wilder smoothing)
// ---------------------------------------------------------------------------

/// Wilder-smoothed ATR state machine.  Used by `StreamingATR` and
/// `StreamingSupertrend`.
pub(crate) struct AtrState {
    period: usize,
    prev_close: f64,
    tr_buf: Vec<f64>,
    atr: f64,
    seeded: bool,
    has_prev: bool,
}

impl AtrState {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_close: 0.0,
            tr_buf: Vec::with_capacity(period),
            atr: 0.0,
            seeded: false,
            has_prev: false,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        // TA-Lib skips TR[0]; first ATR is the SMA of TR[1..=period] at bar `period`.
        if !self.has_prev {
            self.prev_close = close;
            self.has_prev = true;
            return f64::NAN;
        }
        let tr = crate::volatility::true_range(high, low, self.prev_close);
        self.prev_close = close;

        if !self.seeded {
            self.tr_buf.push(tr);
            if self.tr_buf.len() < self.period {
                return f64::NAN;
            }
            // Emit the seed itself at index `period`, matching batch ATR.
            self.atr = self.tr_buf.iter().sum::<f64>() / self.period as f64;
            self.seeded = true;
            return self.atr;
        }
        let p = self.period as f64;
        self.atr = (self.atr * (p - 1.0) + tr) / p;
        self.atr
    }

    pub fn reset(&mut self) {
        self.prev_close = 0.0;
        self.has_prev = false;
        self.tr_buf.clear();
        self.atr = 0.0;
        self.seeded = false;
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

// ---------------------------------------------------------------------------
// StreamingSMA
// ---------------------------------------------------------------------------

/// Simple Moving Average — O(1) per update via running sum.
///
/// Returns NaN during the first `period - 1` bars.
pub struct StreamingSMA {
    period: usize,
    buf: VecDeque<f64>,
    running_sum: f64,
    count: usize,
}

impl StreamingSMA {
    pub fn new(period: usize) -> Result<Self, StreamingError> {
        validate_timeperiod(period, "period", 1)?;
        Ok(Self {
            period,
            buf: VecDeque::with_capacity(period + 1),
            running_sum: 0.0,
            count: 0,
        })
    }

    /// Add a new bar and return the current SMA (NaN during warmup).
    pub fn update(&mut self, value: f64) -> f64 {
        if self.buf.len() == self.period {
            if let Some(old) = self.buf.pop_front() {
                self.running_sum -= old;
            }
        }
        self.buf.push_back(value);
        self.running_sum += value;
        self.count += 1;
        if self.count < self.period {
            f64::NAN
        } else {
            self.running_sum / self.period as f64
        }
    }

    /// Reset state to initial condition.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.running_sum = 0.0;
        self.count = 0;
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

// ---------------------------------------------------------------------------
// StreamingEMA
// ---------------------------------------------------------------------------

/// Exponential Moving Average with SMA seeding.
///
/// Uses a simple SMA for the first `period` bars to seed the EMA, then
/// switches to the standard EMA formula (alpha = 2 / (period + 1)).
/// Returns NaN during the warmup window.
pub struct StreamingEMA {
    inner: EmaState,
}

impl StreamingEMA {
    pub fn new(period: usize) -> Result<Self, StreamingError> {
        validate_timeperiod(period, "period", 1)?;
        Ok(Self {
            inner: EmaState::new(period),
        })
    }

    /// Add a new bar and return the current EMA (NaN during warmup).
    pub fn update(&mut self, value: f64) -> f64 {
        self.inner.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn period(&self) -> usize {
        self.inner.period()
    }
}

// ---------------------------------------------------------------------------
// StreamingRSI
// ---------------------------------------------------------------------------

/// Relative Strength Index with TA-Lib-compatible Wilder seeding.
///
/// Returns NaN during the first `period` bars.
pub struct StreamingRSI {
    period: usize,
    prev: f64,
    has_prev: bool,
    gains: Vec<f64>,
    losses: Vec<f64>,
    avg_gain: f64,
    avg_loss: f64,
    seeded: bool,
}

impl StreamingRSI {
    pub fn new(period: usize) -> Result<Self, StreamingError> {
        validate_timeperiod(period, "period", 1)?;
        Ok(Self {
            period,
            prev: 0.0,
            has_prev: false,
            gains: Vec::with_capacity(period),
            losses: Vec::with_capacity(period),
            avg_gain: 0.0,
            avg_loss: 0.0,
            seeded: false,
        })
    }

    /// Add a new close and return RSI in [0, 100] (NaN during warmup).
    pub fn update(&mut self, value: f64) -> f64 {
        if !self.has_prev {
            self.prev = value;
            self.has_prev = true;
            return f64::NAN;
        }
        let delta = value - self.prev;
        self.prev = value;
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };

        if !self.seeded {
            self.gains.push(gain);
            self.losses.push(loss);
            if self.gains.len() < self.period {
                return f64::NAN;
            }
            self.avg_gain = self.gains.iter().sum::<f64>() / self.period as f64;
            self.avg_loss = self.losses.iter().sum::<f64>() / self.period as f64;
            self.seeded = true;
        } else {
            let pf = (self.period - 1) as f64;
            self.avg_gain = (self.avg_gain * pf + gain) / self.period as f64;
            self.avg_loss = (self.avg_loss * pf + loss) / self.period as f64;
        }

        // TA-Lib convention (and batch `momentum::rsi`): flat window → 0.
        // Clamp for the same rounding reason as the batch version.
        let denom = self.avg_gain + self.avg_loss;
        if denom == 0.0 {
            return 0.0;
        }
        (100.0 * self.avg_gain / denom).clamp(0.0, 100.0)
    }

    pub fn reset(&mut self) {
        self.prev = 0.0;
        self.has_prev = false;
        self.gains.clear();
        self.losses.clear();
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.seeded = false;
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

// ---------------------------------------------------------------------------
// StreamingATR
// ---------------------------------------------------------------------------

/// Average True Range with TA-Lib-compatible Wilder seeding.
///
/// Accepts (high, low, close) per bar.
/// Returns NaN during the first `period` bars.
pub struct StreamingATR {
    inner: AtrState,
}

impl StreamingATR {
    pub fn new(period: usize) -> Result<Self, StreamingError> {
        validate_timeperiod(period, "period", 1)?;
        Ok(Self {
            inner: AtrState::new(period),
        })
    }

    /// Add a new bar (high, low, close) and return ATR (NaN during warmup).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        self.inner.update(high, low, close)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn period(&self) -> usize {
        self.inner.period()
    }
}

// ---------------------------------------------------------------------------
// StreamingBBands
// ---------------------------------------------------------------------------

/// Bollinger Bands — streaming variant using Welford's online algorithm.
///
/// Returns (upper, middle, lower).
/// NaN tuple during the warmup window.
pub struct StreamingBBands {
    period: usize,
    nbdevup: f64,
    nbdevdn: f64,
    buf: VecDeque<f64>,
    mean: f64,
    m2: f64,
}

impl StreamingBBands {
    pub fn new(period: usize, nbdevup: f64, nbdevdn: f64) -> Result<Self, StreamingError> {
        validate_timeperiod(period, "period", 2)?;
        Ok(Self {
            period,
            nbdevup,
            nbdevdn,
            buf: VecDeque::with_capacity(period + 1),
            mean: 0.0,
            m2: 0.0,
        })
    }

    /// Add a new bar; return (upper, middle, lower). NaN tuple during warmup.
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        let n = self.buf.len();

        if n == self.period {
            let x_old = self.buf.pop_front().unwrap();
            let count = self.period as f64;
            let delta_old = x_old - self.mean;
            self.mean -= delta_old / (count - 1.0);
            let delta2_old = x_old - self.mean;
            self.m2 -= delta_old * delta2_old;
        }

        self.buf.push_back(value);
        let count = self.buf.len() as f64;
        let delta_new = value - self.mean;
        self.mean += delta_new / count;
        let delta2_new = value - self.mean;
        self.m2 += delta_new * delta2_new;

        if self.m2 < 0.0 {
            self.m2 = 0.0;
        }

        if self.buf.len() < self.period {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        // Population variance (divide by N), matching batch `overlap::bbands`
        // and TA-Lib STDDEV.
        let variance = self.m2 / count;
        let std = variance.sqrt();
        (
            self.mean + self.nbdevup * std,
            self.mean,
            self.mean - self.nbdevdn * std,
        )
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.mean = 0.0;
        self.m2 = 0.0;
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

// ---------------------------------------------------------------------------
// StreamingMACD
// ---------------------------------------------------------------------------

/// MACD — fast EMA, slow EMA, signal EMA.
///
/// Returns (macd_line, signal_line, histogram).
/// NaN values during warmup.
pub struct StreamingMACD {
    fast: EmaState,
    slow: EmaState,
    signal: EmaState,
}

impl StreamingMACD {
    pub fn new(
        fastperiod: usize,
        slowperiod: usize,
        signalperiod: usize,
    ) -> Result<Self, StreamingError> {
        validate_timeperiod(fastperiod, "fastperiod", 1)?;
        validate_timeperiod(slowperiod, "slowperiod", 1)?;
        validate_timeperiod(signalperiod, "signalperiod", 1)?;
        if fastperiod >= slowperiod {
            return Err(StreamingError(
                "fastperiod must be < slowperiod".to_string(),
            ));
        }
        Ok(Self {
            fast: EmaState::new(fastperiod),
            slow: EmaState::new(slowperiod),
            signal: EmaState::new(signalperiod),
        })
    }

    /// Add a new close; return (macd_line, signal_line, histogram).
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        let fast_val = self.fast.update(value);
        let slow_val = self.slow.update(value);

        if slow_val.is_nan() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let macd = fast_val - slow_val;
        let signal = self.signal.update(macd);
        if signal.is_nan() {
            // Batch `overlap::macd` (and TA-Lib) pad the MACD line until the
            // signal line is valid — suppress the early MACD values too.
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        (macd, signal, macd - signal)
    }

    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.signal.reset();
    }

    pub fn fast_period(&self) -> usize {
        self.fast.period()
    }

    pub fn slow_period(&self) -> usize {
        self.slow.period()
    }

    pub fn signal_period(&self) -> usize {
        self.signal.period()
    }
}

// ---------------------------------------------------------------------------
// StreamingStoch
// ---------------------------------------------------------------------------

/// Slow Stochastic (SMA-smoothed).
///
/// Returns (slowk, slowd).
/// NaN tuple during warmup.
pub struct StreamingStoch {
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
    high_buf: VecDeque<f64>,
    low_buf: VecDeque<f64>,
    fastk_buf: VecDeque<f64>,
    slowk_buf: VecDeque<f64>,
}

impl StreamingStoch {
    pub fn new(
        fastk_period: usize,
        slowk_period: usize,
        slowd_period: usize,
    ) -> Result<Self, StreamingError> {
        validate_timeperiod(fastk_period, "fastk_period", 1)?;
        validate_timeperiod(slowk_period, "slowk_period", 1)?;
        validate_timeperiod(slowd_period, "slowd_period", 1)?;
        Ok(Self {
            fastk_period,
            slowk_period,
            slowd_period,
            high_buf: VecDeque::with_capacity(fastk_period + 1),
            low_buf: VecDeque::with_capacity(fastk_period + 1),
            fastk_buf: VecDeque::with_capacity(slowk_period + 1),
            slowk_buf: VecDeque::with_capacity(slowd_period + 1),
        })
    }

    /// Add a new bar (high, low, close); return (slowk, slowd).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        if self.high_buf.len() == self.fastk_period {
            self.high_buf.pop_front();
            self.low_buf.pop_front();
        }
        self.high_buf.push_back(high);
        self.low_buf.push_back(low);

        if self.high_buf.len() < self.fastk_period {
            return (f64::NAN, f64::NAN);
        }

        let max_h = self
            .high_buf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_l = self.low_buf.iter().cloned().fold(f64::INFINITY, f64::min);

        let fastk = if max_h != min_l {
            100.0 * (close - min_l) / (max_h - min_l)
        } else {
            0.0
        };

        if self.fastk_buf.len() == self.slowk_period {
            self.fastk_buf.pop_front();
        }
        self.fastk_buf.push_back(fastk);
        if self.fastk_buf.len() < self.slowk_period {
            return (f64::NAN, f64::NAN);
        }

        let slowk = self.fastk_buf.iter().sum::<f64>() / self.slowk_period as f64;

        if self.slowk_buf.len() == self.slowd_period {
            self.slowk_buf.pop_front();
        }
        self.slowk_buf.push_back(slowk);
        if self.slowk_buf.len() < self.slowd_period {
            // Batch `momentum::stoch` (and TA-Lib) pad slowk until slowd is
            // valid — suppress the early slowk values too.
            return (f64::NAN, f64::NAN);
        }

        let slowd = self.slowk_buf.iter().sum::<f64>() / self.slowd_period as f64;
        (slowk, slowd)
    }

    pub fn reset(&mut self) {
        self.high_buf.clear();
        self.low_buf.clear();
        self.fastk_buf.clear();
        self.slowk_buf.clear();
    }

    pub fn period(&self) -> usize {
        self.fastk_period
    }
}

// ---------------------------------------------------------------------------
// StreamingVWAP
// ---------------------------------------------------------------------------

/// Cumulative Volume Weighted Average Price.
///
/// Resets automatically when `reset()` is called (e.g. at session open).
/// Accepts (high, low, close, volume) per bar.
#[derive(Default)]
pub struct StreamingVWAP {
    cum_tpv: f64,
    cum_vol: f64,
}

impl StreamingVWAP {
    pub fn new() -> Self {
        Self {
            cum_tpv: 0.0,
            cum_vol: 0.0,
        }
    }

    /// Add a new bar (high, low, close, volume) and return cumulative VWAP.
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let tp = (high + low + close) / 3.0;
        self.cum_tpv += tp * volume;
        self.cum_vol += volume;
        if self.cum_vol == 0.0 {
            f64::NAN
        } else {
            self.cum_tpv / self.cum_vol
        }
    }

    /// Reset for a new session.
    pub fn reset(&mut self) {
        self.cum_tpv = 0.0;
        self.cum_vol = 0.0;
    }
}

// ---------------------------------------------------------------------------
// StreamingSupertrend
// ---------------------------------------------------------------------------

/// ATR-based Supertrend — streaming variant.
///
/// Accepts (high, low, close) per bar.
/// Returns (supertrend_line, direction).
/// direction: 1 = uptrend, -1 = downtrend, 0 = warmup.
pub struct StreamingSupertrend {
    period: usize,
    multiplier: f64,
    atr: AtrState,
    upper_band: f64,
    lower_band: f64,
    has_bands: bool,
    direction: i8,
    prev_close: f64,
    has_prev: bool,
}

impl StreamingSupertrend {
    pub fn new(period: usize, multiplier: f64) -> Result<Self, StreamingError> {
        validate_timeperiod(period, "period", 1)?;
        Ok(Self {
            period,
            multiplier,
            atr: AtrState::new(period),
            upper_band: 0.0,
            lower_band: 0.0,
            has_bands: false,
            direction: 0,
            prev_close: 0.0,
            has_prev: false,
        })
    }

    /// Add a new bar (high, low, close); return (supertrend_line, direction).
    ///
    /// First ATR-valid bar is `timeperiod`; emit using unadjusted bands and
    /// the same close-vs-upper rule as batch `extended::supertrend`.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, i8) {
        let atr = self.atr.update(high, low, close);
        if atr.is_nan() {
            self.prev_close = close;
            self.has_prev = true;
            return (f64::NAN, 0);
        }

        let hl2 = (high + low) / 2.0;
        let upper_basic = hl2 + self.multiplier * atr;
        let lower_basic = hl2 - self.multiplier * atr;

        if !self.has_bands {
            self.upper_band = upper_basic;
            self.lower_band = lower_basic;
            self.has_bands = true;
            self.direction = if close > self.upper_band { 1 } else { -1 };
            self.prev_close = close;
            self.has_prev = true;
            let line = if self.direction == 1 {
                self.lower_band
            } else {
                self.upper_band
            };
            return (line, self.direction);
        }

        let prev_close = self.prev_close;

        let new_lower = if lower_basic > self.lower_band || prev_close < self.lower_band {
            lower_basic
        } else {
            self.lower_band
        };
        let new_upper = if upper_basic < self.upper_band || prev_close > self.upper_band {
            upper_basic
        } else {
            self.upper_band
        };

        self.lower_band = new_lower;
        self.upper_band = new_upper;

        self.direction = if self.direction == -1 {
            if close > new_upper {
                1
            } else {
                -1
            }
        } else if close < new_lower {
            -1
        } else {
            1
        };

        self.prev_close = close;
        let line = if self.direction == 1 {
            new_lower
        } else {
            new_upper
        };
        (line, self.direction)
    }

    pub fn reset(&mut self) {
        self.atr.reset();
        self.upper_band = 0.0;
        self.lower_band = 0.0;
        self.has_bands = false;
        self.direction = 0;
        self.prev_close = 0.0;
        self.has_prev = false;
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compare two f64 values, treating NaN == NaN as true.
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < tol
    }

    #[test]
    fn test_sma_basic() {
        let mut sma = StreamingSMA::new(3).unwrap();
        assert!(sma.update(1.0).is_nan());
        assert!(sma.update(2.0).is_nan());
        let v = sma.update(3.0);
        assert!(approx_eq(v, 2.0, 1e-10));
        let v = sma.update(4.0);
        assert!(approx_eq(v, 3.0, 1e-10));
        let v = sma.update(5.0);
        assert!(approx_eq(v, 4.0, 1e-10));
        assert_eq!(sma.period(), 3);
    }

    #[test]
    fn test_sma_reset() {
        let mut sma = StreamingSMA::new(2).unwrap();
        sma.update(10.0);
        sma.update(20.0);
        sma.reset();
        assert!(sma.update(5.0).is_nan());
        let v = sma.update(7.0);
        assert!(approx_eq(v, 6.0, 1e-10));
    }

    #[test]
    fn test_ema_warmup_and_decay() {
        let mut ema = StreamingEMA::new(3).unwrap();
        assert!(ema.update(2.0).is_nan());
        assert!(ema.update(4.0).is_nan());
        // Third bar: SMA seed = (2+4+6)/3 = 4.0
        let v = ema.update(6.0);
        assert!(approx_eq(v, 4.0, 1e-10));
        // Fourth bar: alpha = 0.5, ema = 4.0 + 0.5*(8.0-4.0) = 6.0
        let v = ema.update(8.0);
        assert!(approx_eq(v, 6.0, 1e-10));
    }

    #[test]
    fn test_rsi_warmup() {
        let mut rsi = StreamingRSI::new(3).unwrap();
        // First bar: no prev
        assert!(rsi.update(44.0).is_nan());
        // Bars 2-4: collecting gains/losses
        assert!(rsi.update(44.5).is_nan());
        assert!(rsi.update(43.5).is_nan());
        // Bar 5: seeded
        let v = rsi.update(44.5);
        assert!(!v.is_nan());
        assert!((0.0..=100.0).contains(&v));
    }

    #[test]
    fn test_atr_warmup() {
        let mut atr = StreamingATR::new(3).unwrap();
        // First 3 bars return NaN (TR[0] skipped; seed at bar `period`)
        assert!(atr.update(10.0, 9.0, 9.5).is_nan());
        assert!(atr.update(11.0, 9.5, 10.5).is_nan());
        assert!(atr.update(10.5, 9.0, 9.5).is_nan());
        // Bar 4 (index 3): SMA of TR[1..=3]
        let v = atr.update(11.0, 10.0, 10.5);
        assert!(!v.is_nan());
        assert!(v > 0.0);
    }

    #[test]
    fn streaming_atr_matches_batch_bar_for_bar() {
        let h = vec![20.0, 12.0, 13.0, 14.0, 15.0, 16.0, 15.5, 17.0];
        let l = vec![5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 15.0];
        let c = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 16.0];
        let period = 3;
        let batch = crate::volatility::atr(&h, &l, &c, period);
        let mut stream = StreamingATR::new(period).unwrap();
        for i in 0..h.len() {
            let v = stream.update(h[i], l[i], c[i]);
            assert!(
                approx_eq(v, batch[i], 1e-10),
                "bar {i}: streaming={v} batch={}",
                batch[i]
            );
        }
    }

    #[test]
    fn test_bbands_warmup() {
        let mut bb = StreamingBBands::new(3, 2.0, 2.0).unwrap();
        let (u, m, l) = bb.update(10.0);
        assert!(u.is_nan() && m.is_nan() && l.is_nan());
        let (u, m, l) = bb.update(11.0);
        assert!(u.is_nan() && m.is_nan() && l.is_nan());
        let (u, m, l) = bb.update(12.0);
        assert!(!u.is_nan() && !m.is_nan() && !l.is_nan());
        assert!(approx_eq(m, 11.0, 1e-10));
        assert!(u > m && l < m);
    }

    #[test]
    fn streaming_bbands_uses_population_std() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (bu, bm, bl) = crate::overlap::bbands(&prices, 3, 2.0, 2.0, 0);
        let mut bb = StreamingBBands::new(3, 2.0, 2.0).unwrap();
        for (i, &p) in prices.iter().enumerate() {
            let (u, m, l) = bb.update(p);
            assert!(approx_eq(u, bu[i], 1e-10), "upper[{i}]: {u} vs {}", bu[i]);
            assert!(approx_eq(m, bm[i], 1e-10), "middle[{i}]: {m} vs {}", bm[i]);
            assert!(approx_eq(l, bl[i], 1e-10), "lower[{i}]: {l} vs {}", bl[i]);
        }
        // Window [1,2,3]: pop std = sqrt(2/3), not sample std = 1.
        let pop_std = (2.0_f64 / 3.0).sqrt();
        assert!(approx_eq(bu[2], 2.0 + 2.0 * pop_std, 1e-12));
    }

    #[test]
    fn test_macd_basic() {
        let mut macd = StreamingMACD::new(3, 5, 2).unwrap();
        // sig_start = (5-1)+(2-1) = 5; MACD itself is NaN-padded until then
        for i in 0..5 {
            let (m, s, _h) = macd.update(100.0 + i as f64);
            assert!(m.is_nan(), "macd should be NaN at bar {i}");
            assert!(s.is_nan());
        }
        let (m, s, _h) = macd.update(105.0);
        assert!(!m.is_nan());
        assert!(!s.is_nan());
    }

    #[test]
    fn streaming_macd_matches_batch_including_nan_pad() {
        let close: Vec<f64> = (0..40).map(|i| 100.0 + i as f64).collect();
        let (bm, bs, bh) = crate::overlap::macd(&close, 12, 26, 9);
        let mut stream = StreamingMACD::new(12, 26, 9).unwrap();
        for i in 0..close.len() {
            let (m, s, h) = stream.update(close[i]);
            assert!(approx_eq(m, bm[i], 1e-10), "macd[{i}]: {m} vs {}", bm[i]);
            assert!(approx_eq(s, bs[i], 1e-10), "signal[{i}]: {s} vs {}", bs[i]);
            assert!(approx_eq(h, bh[i], 1e-10), "hist[{i}]: {h} vs {}", bh[i]);
        }
    }

    #[test]
    fn test_macd_matches_batch() {
        let closes: Vec<f64> = (0..40)
            .map(|i| 100.0 + (i as f64 * 0.7).sin() * 5.0)
            .collect();
        let (b_macd, b_signal, b_hist) = crate::overlap::macd(&closes, 3, 5, 2);

        let mut macd = StreamingMACD::new(3, 5, 2).unwrap();
        for (i, &c) in closes.iter().enumerate() {
            let (m, s, h) = macd.update(c);
            assert!(approx_eq(m, b_macd[i], 1e-9), "macd bar {i}");
            assert!(approx_eq(s, b_signal[i], 1e-9), "signal bar {i}");
            assert!(approx_eq(h, b_hist[i], 1e-9), "hist bar {i}");
        }
    }

    #[test]
    fn test_macd_fast_ge_slow_rejected() {
        assert!(StreamingMACD::new(5, 3, 2).is_err());
        assert!(StreamingMACD::new(5, 5, 2).is_err());
    }

    #[test]
    fn test_stoch_matches_batch() {
        let n = 40;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let (b_k, b_d) = crate::momentum::stoch(&high, &low, &close, 3, 2, 0, 2, 0);

        let mut stoch = StreamingStoch::new(3, 2, 2).unwrap();
        for i in 0..n {
            let (k, d) = stoch.update(high[i], low[i], close[i]);
            assert!(approx_eq(k, b_k[i], 1e-9), "slowk bar {i}");
            assert!(approx_eq(d, b_d[i], 1e-9), "slowd bar {i}");
        }
    }

    #[test]
    fn streaming_stoch_matches_batch_including_nan_pad() {
        let h: Vec<f64> = (0..30).map(|i| 12.0 + i as f64).collect();
        let l: Vec<f64> = (0..30).map(|i| 8.0 + i as f64).collect();
        let c: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let (bk, bd) = crate::momentum::stoch(&h, &l, &c, 5, 3, 0, 3, 0);
        let mut stream = StreamingStoch::new(5, 3, 3).unwrap();
        for i in 0..h.len() {
            let (k, d) = stream.update(h[i], l[i], c[i]);
            assert!(approx_eq(k, bk[i], 1e-10), "slowk[{i}]: {k} vs {}", bk[i]);
            assert!(approx_eq(d, bd[i], 1e-10), "slowd[{i}]: {d} vs {}", bd[i]);
        }
    }

    #[test]
    fn test_vwap_basic() {
        let mut vwap = StreamingVWAP::new();
        let v = vwap.update(10.0, 8.0, 9.0, 100.0);
        // tp = (10+8+9)/3 = 9.0, vwap = 9.0*100/100 = 9.0
        assert!(approx_eq(v, 9.0, 1e-10));
        let v = vwap.update(12.0, 10.0, 11.0, 200.0);
        // tp2 = 11.0, cum_tpv = 900+2200=3100, cum_vol=300, vwap=10.333..
        assert!(approx_eq(v, 3100.0 / 300.0, 1e-10));
    }

    #[test]
    fn test_vwap_zero_volume() {
        let mut vwap = StreamingVWAP::new();
        let v = vwap.update(10.0, 8.0, 9.0, 0.0);
        assert!(v.is_nan());
    }

    #[test]
    fn test_supertrend_warmup() {
        let mut st = StreamingSupertrend::new(3, 2.0).unwrap();
        let (line, dir) = st.update(10.0, 9.0, 9.5);
        assert!(line.is_nan() && dir == 0);
        let (line, dir) = st.update(11.0, 9.5, 10.5);
        assert!(line.is_nan() && dir == 0);
        let (line, dir) = st.update(10.5, 9.0, 9.5);
        assert!(line.is_nan() && dir == 0);
        // Bar 4 (index 3 = timeperiod): first real value
        let (line, dir) = st.update(11.0, 10.0, 10.5);
        assert!(!line.is_nan());
        assert!(dir == 1 || dir == -1);
    }

    /// Deterministic OHLC fixture for streaming/batch parity tests.
    fn parity_ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.4).sin() * 6.0 + (i as f64 * 0.11).cos() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.25).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.1).collect();
        (high, low, close)
    }

    #[test]
    fn test_atr_matches_batch() {
        let (high, low, close) = parity_ohlc(40);
        let batch = crate::volatility::atr(&high, &low, &close, 5);
        let mut atr = StreamingATR::new(5).unwrap();
        for i in 0..close.len() {
            let v = atr.update(high[i], low[i], close[i]);
            assert!(approx_eq(v, batch[i], 1e-9), "atr bar {i}");
        }
    }

    #[test]
    fn test_rsi_matches_batch() {
        let (_, _, close) = parity_ohlc(40);
        let batch = crate::momentum::rsi(&close, 5);
        let mut rsi = StreamingRSI::new(5).unwrap();
        for (i, &c) in close.iter().enumerate() {
            let v = rsi.update(c);
            assert!(approx_eq(v, batch[i], 1e-9), "rsi bar {i}");
        }
    }

    #[test]
    fn test_rsi_flat_series_is_zero() {
        // TA-Lib convention: no gains and no losses → 0, not 100.
        let mut rsi = StreamingRSI::new(3).unwrap();
        let mut last = f64::NAN;
        for _ in 0..8 {
            last = rsi.update(50.0);
        }
        assert!(approx_eq(last, 0.0, 1e-10));
        assert!(approx_eq(
            crate::momentum::rsi(&[50.0; 8], 3)[7],
            0.0,
            1e-10
        ));
    }

    #[test]
    fn test_bbands_matches_batch() {
        let (_, _, close) = parity_ohlc(40);
        let (b_up, b_mid, b_lo) = crate::overlap::bbands(&close, 5, 2.0, 2.0, 0);
        let mut bb = StreamingBBands::new(5, 2.0, 2.0).unwrap();
        for (i, &c) in close.iter().enumerate() {
            let (u, m, l) = bb.update(c);
            assert!(approx_eq(u, b_up[i], 1e-9), "upper bar {i}");
            assert!(approx_eq(m, b_mid[i], 1e-9), "middle bar {i}");
            assert!(approx_eq(l, b_lo[i], 1e-9), "lower bar {i}");
        }
    }

    #[test]
    fn test_supertrend_matches_batch() {
        let (high, low, close) = parity_ohlc(50);
        let (b_line, b_dir) = crate::extended::supertrend(&high, &low, &close, 5, 2.0);
        let mut st = StreamingSupertrend::new(5, 2.0).unwrap();
        for i in 0..close.len() {
            let (line, dir) = st.update(high[i], low[i], close[i]);
            assert!(approx_eq(line, b_line[i], 1e-9), "line bar {i}");
            assert_eq!(dir, b_dir[i], "direction bar {i}");
        }
    }

    #[test]
    fn streaming_supertrend_matches_batch() {
        let h = vec![20.0, 12.0, 13.0, 14.0, 15.0, 14.5, 15.5, 16.0, 15.0, 17.0];
        let l = vec![5.0, 10.0, 11.0, 12.0, 13.0, 12.5, 13.5, 14.0, 13.0, 14.5];
        let c = vec![10.0, 11.0, 12.0, 13.0, 14.0, 13.5, 14.5, 15.0, 14.0, 16.0];
        let (bl, bd) = crate::extended::supertrend(&h, &l, &c, 3, 2.0);
        let mut stream = StreamingSupertrend::new(3, 2.0).unwrap();
        for i in 0..h.len() {
            let (line, dir) = stream.update(h[i], l[i], c[i]);
            assert!(
                approx_eq(line, bl[i], 1e-10),
                "line[{i}]: {line} vs {}",
                bl[i]
            );
            assert_eq!(dir, bd[i], "dir[{i}]: {dir} vs {}", bd[i]);
        }
    }

    #[test]
    fn test_streaming_sma_matches_batch() {
        // Compare streaming SMA against a simple batch computation
        let data = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
        let period = 3;
        let mut sma = StreamingSMA::new(period).unwrap();
        let streaming: Vec<f64> = data.iter().map(|&v| sma.update(v)).collect();

        // Batch SMA
        for i in 0..data.len() {
            if i + 1 < period {
                assert!(streaming[i].is_nan(), "bar {} should be NaN", i);
            } else {
                let batch: f64 = data[i + 1 - period..=i].iter().sum::<f64>() / period as f64;
                assert!(
                    approx_eq(streaming[i], batch, 1e-10),
                    "bar {}: streaming={} batch={}",
                    i,
                    streaming[i],
                    batch
                );
            }
        }
    }
}
