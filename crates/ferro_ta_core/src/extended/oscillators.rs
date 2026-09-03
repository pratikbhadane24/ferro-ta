//! Oscillator extended indicators (AO, AC, KST, TSI, STC, and related).
//!
//! Every kernel here allocates its outputs with `vec![f64::NAN; n]` and writes
//! by index, leaving warm-up bars at their initialized `NaN`. An earlier
//! revision used `Vec::with_capacity` + `push` to dodge the `NaN` prologue;
//! that was measured as a regression (the per-bar capacity check and vec-header
//! reload cost more than the one-time prologue saves) and reverted — see the
//! rationale on the ROC family in `crate::momentum::roc`.

use crate::momentum;
use crate::overlap;
use crate::price_transform;
use crate::rolling;
use crate::volatility as vol;
use crate::volume;

/// Running window sum with `math_ops::rolling_sum`'s exact semantics,
/// usable *inside* another kernel's traversal.
///
/// `rolling_sum` keeps `NaN`s out of the accumulator and counts them instead,
/// so a single `NaN` corrupts exactly the `timeperiod` outputs whose window
/// contains it and the sum then recovers on its own. This replicates that
/// recurrence — same operation order, same non-finite handling, no reseed — so
/// a kernel driving this is bit-identical to one that called `rolling_sum` on
/// a materialized intermediate, minus the intermediate.
///
/// `rolling::RollingSum` is the general-purpose accumulator, but its
/// periodic exact reseed changes the accumulated value in the last ulp, which
/// would break these kernels' published goldens.
struct NanAwareSum {
    sum: f64,
    nan_count: usize,
    timeperiod: usize,
    period_f64: f64,
}

impl NanAwareSum {
    fn new(timeperiod: usize) -> Self {
        Self {
            sum: 0.0,
            nan_count: 0,
            timeperiod,
            period_f64: timeperiod as f64,
        }
    }

    /// Slide to bar `i`: `x_new` enters the window and `x_old` leaves it
    /// (`x_old` is only read once `i >= timeperiod`).
    ///
    /// Returns the window mean, or `NaN` during warmup, while the window holds
    /// a `NaN`, or when the sum itself is not finite.
    #[inline]
    fn advance(&mut self, i: usize, x_new: f64, x_old: f64) -> f64 {
        if x_new.is_nan() {
            self.nan_count += 1;
        } else {
            self.sum += x_new;
        }
        if i >= self.timeperiod {
            if x_old.is_nan() {
                self.nan_count -= 1;
            } else {
                self.sum -= x_old;
            }
        }
        if i + 1 < self.timeperiod || self.nan_count > 0 {
            return f64::NAN;
        }
        // Kept as `/ period` rather than `* (1 / period)`: the division is off
        // the loop-carried path (only `sum` is carried), so hoisting it would
        // buy nothing and would change the last ulp of every output.
        if self.sum.is_finite() {
            self.sum / self.period_f64
        } else {
            f64::NAN
        }
    }
}

/// `momentum::roc(close, timeperiod)[i]`, without materializing the series.
#[inline]
fn roc_at(close: &[f64], i: usize, timeperiod: usize) -> f64 {
    if i < timeperiod {
        return f64::NAN;
    }
    let prev = close[i - timeperiod];
    if prev != 0.0 {
        (close[i] - prev) / prev * 100.0
    } else {
        f64::NAN
    }
}

fn wma_from_first_finite(src: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    if timeperiod < 1 {
        return out;
    }
    let Some(start) = src.iter().position(|v| v.is_finite()) else {
        return out;
    };
    let tail = overlap::wma(&src[start..], timeperiod);
    out[start..].copy_from_slice(&tail);
    out
}

/// Awesome Oscillator: `SMA(median, fast) − SMA(median, slow)`.
pub fn ao(high: &[f64], low: &[f64], fastperiod: usize, slowperiod: usize) -> Vec<f64> {
    let n = high.len();
    if fastperiod < 1 || slowperiod < 1 || n == 0 || low.len() != n {
        return vec![f64::NAN; n];
    }
    let median = price_transform::medprice(high, low);
    // The difference lands in the fast buffer: reusing it drops a full-length
    // NaN-filled output allocation.
    let mut fast = overlap::sma(&median, fastperiod);
    let slow = overlap::sma(&median, slowperiod);
    for (f, &s) in fast.iter_mut().zip(slow.iter()) {
        *f = if f.is_finite() && s.is_finite() {
            *f - s
        } else {
            f64::NAN
        };
    }
    fast
}

/// Accelerator Oscillator: `AO − SMA(AO, timeperiod)`.
pub fn ac(
    high: &[f64],
    low: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    timeperiod: usize,
) -> Vec<f64> {
    let n = high.len();
    if timeperiod < 1 {
        return vec![f64::NAN; n];
    }
    let awesome = ao(high, low, fastperiod, slowperiod);
    let mut result = vec![f64::NAN; n];
    let mut smoother = NanAwareSum::new(timeperiod);
    for i in 0..n {
        let x = awesome[i];
        let old = if i >= timeperiod {
            awesome[i - timeperiod]
        } else {
            f64::NAN
        };
        let mean = smoother.advance(i, x, old);
        if x.is_finite() && mean.is_finite() {
            result[i] = x - mean;
        }
    }
    result
}

/// Price Oscillator (SMA): `SMA(close, fast) − SMA(close, slow)`.
pub fn po(close: &[f64], fastperiod: usize, slowperiod: usize) -> Vec<f64> {
    let n = close.len();
    if fastperiod < 1 || slowperiod < 1 || n == 0 {
        return vec![f64::NAN; n];
    }
    let mut fast = overlap::sma(close, fastperiod);
    let slow = overlap::sma(close, slowperiod);
    for (f, &s) in fast.iter_mut().zip(slow.iter()) {
        *f = if f.is_finite() && s.is_finite() {
            *f - s
        } else {
            f64::NAN
        };
    }
    fast
}

/// Detrended Price Oscillator: `close[i − shift] − SMA(close, timeperiod)`,
/// where `shift = timeperiod / 2 + 1`.
pub fn dpo(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 || n == 0 {
        return vec![f64::NAN; n];
    }
    let shift = timeperiod / 2 + 1;
    let mut sma = overlap::sma(close, timeperiod);
    for i in 0..n {
        sma[i] = if i >= shift && sma[i].is_finite() {
            close[i - shift] - sma[i]
        } else {
            f64::NAN
        };
    }
    sma
}

/// One `(C−O)`-style 4-bar weighted RVI term at bar `i`, or `NaN` before the
/// series is long enough. `a`/`b` are `(close, open)` or `(high, low)`.
#[inline]
fn rvi_term(a: &[f64], b: &[f64], i: usize) -> f64 {
    if i < 3 {
        return f64::NAN;
    }
    (a[i] - b[i])
        + 2.0 * (a[i - 1] - b[i - 1])
        + 2.0 * (a[i - 2] - b[i - 2])
        + (a[i - 3] - b[i - 3])
}

/// Relative Vigor Index and its 4-bar weighted signal.
///
/// `num = (C−O) + 2(C1−O1) + 2(C2−O2) + (C3−O3)`,
/// `den = (H−L) + 2(H1−L1) + 2(H2−L2) + (H3−L3)`,
/// `RVI = SMA(num) / SMA(den)`,
/// `signal = (RVI + 2 RVI1 + 2 RVI2 + RVI3) / 6`.
///
/// Single traversal: the two weighted term series live in
/// `timeperiod`-slot rings rather than full-length buffers, so each term is
/// computed exactly once and the kernel allocates only its two outputs plus
/// 2 × `timeperiod` scratch words that stay resident in L1.
pub fn rvi(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    if timeperiod < 1 || n < 4 || open.len() != n || high.len() != n || low.len() != n {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }
    let mut rvi_out = vec![f64::NAN; n];
    let mut signal = vec![f64::NAN; n];
    let mut num_sum = NanAwareSum::new(timeperiod);
    let mut den_sum = NanAwareSum::new(timeperiod);
    let mut ring_num = vec![f64::NAN; timeperiod];
    let mut ring_den = vec![f64::NAN; timeperiod];
    let mut pos = 0usize;
    for i in 0..n {
        let num_in = rvi_term(close, open, i);
        let den_in = rvi_term(high, low, i);
        // Slot `i % timeperiod` was last written at bar `i - timeperiod`,
        // which is exactly the term leaving the window.
        let (num_old, den_old) = (ring_num[pos], ring_den[pos]);
        ring_num[pos] = num_in;
        ring_den[pos] = den_in;
        pos += 1;
        if pos == timeperiod {
            pos = 0;
        }
        let num_s = num_sum.advance(i, num_in, num_old);
        let den_s = den_sum.advance(i, den_in, den_old);
        let line = if num_s.is_finite() && den_s.is_finite() && den_s != 0.0 {
            num_s / den_s
        } else {
            f64::NAN
        };
        rvi_out[i] = line;
        if i >= 3 {
            let b = rvi_out[i - 1];
            let c = rvi_out[i - 2];
            let d = rvi_out[i - 3];
            if line.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() {
                signal[i] = (line + 2.0 * b + 2.0 * c + d) / 6.0;
            }
        }
    }
    (rvi_out, signal)
}

/// Chaikin Oscillator — same math as [`volume::adosc`].
pub fn cho(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fastperiod: usize,
    slowperiod: usize,
) -> Vec<f64> {
    volume::adosc(high, low, close, volume, fastperiod, slowperiod)
}

/// One `SMA(ROC(close, roc_period), sma_period)` stream of KST, advanced one
/// bar at a time.
///
/// The ROC value leaving the SMA window is held in a `sma_period`-slot ring
/// instead of being recomputed, so the fused loop pays exactly one division
/// per stream per bar — the same count the two-pass version paid.
struct RocSmaStream {
    roc_period: usize,
    ring: Box<[f64]>,
    pos: usize,
    smoother: NanAwareSum,
}

impl RocSmaStream {
    fn new(roc_period: usize, sma_period: usize) -> Self {
        Self {
            roc_period,
            ring: vec![f64::NAN; sma_period].into_boxed_slice(),
            pos: 0,
            smoother: NanAwareSum::new(sma_period),
        }
    }

    #[inline]
    fn advance(&mut self, i: usize, close: &[f64]) -> f64 {
        let x_new = roc_at(close, i, self.roc_period);
        // Slot `i % sma_period` was last written at bar `i - sma_period`,
        // which is exactly the value leaving the window.
        let x_old = self.ring[self.pos];
        self.ring[self.pos] = x_new;
        self.pos += 1;
        if self.pos == self.ring.len() {
            self.pos = 0;
        }
        self.smoother.advance(i, x_new, x_old)
    }
}

/// Know Sure Thing: weighted sum of four ROC SMAs, plus a signal SMA.
///
/// Defaults match the classic 10/15/20/30 ROC windows with 10/10/10/15
/// smoothers and a 9-bar signal.
///
/// All nine moving pieces run in one traversal, so only the two outputs are
/// allocated — the four ROC series, four SMA series and the signal's rolling
/// sum never touch memory.
#[allow(clippy::too_many_arguments)]
pub fn kst(
    close: &[f64],
    roc1: usize,
    roc2: usize,
    roc3: usize,
    roc4: usize,
    sma1: usize,
    sma2: usize,
    sma3: usize,
    sma4: usize,
    signalperiod: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    if n == 0
        || roc1 < 1
        || roc2 < 1
        || roc3 < 1
        || roc4 < 1
        || sma1 < 1
        || sma2 < 1
        || sma3 < 1
        || sma4 < 1
        || signalperiod < 1
    {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }
    let mut s1 = RocSmaStream::new(roc1, sma1);
    let mut s2 = RocSmaStream::new(roc2, sma2);
    let mut s3 = RocSmaStream::new(roc3, sma3);
    let mut s4 = RocSmaStream::new(roc4, sma4);
    let mut signal_sum = NanAwareSum::new(signalperiod);
    let mut kst_out = vec![f64::NAN; n];
    let mut signal = vec![f64::NAN; n];
    for i in 0..n {
        let a = s1.advance(i, close);
        let b = s2.advance(i, close);
        let c = s3.advance(i, close);
        let d = s4.advance(i, close);
        let line = if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() {
            a + 2.0 * b + 3.0 * c + 4.0 * d
        } else {
            f64::NAN
        };
        kst_out[i] = line;
        let old = if i >= signalperiod {
            kst_out[i - signalperiod]
        } else {
            f64::NAN
        };
        signal[i] = signal_sum.advance(i, line, old);
    }
    (kst_out, signal)
}

/// True Strength Index and an EMA signal of that series.
pub fn tsi(
    close: &[f64],
    longperiod: usize,
    shortperiod: usize,
    signalperiod: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    if longperiod < 1 || shortperiod < 1 || signalperiod < 1 || n == 0 {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }
    // Bar 0 keeps the initialized `NaN`; every later bar is written by index.
    let mut mom = vec![f64::NAN; n];
    let mut abs_mom = vec![f64::NAN; n];
    for i in 1..n {
        let d = close[i] - close[i - 1];
        mom[i] = d;
        abs_mom[i] = d.abs();
    }
    // The ratio lands in the numerator buffer, which is also what the signal
    // EMA reads and what the caller gets back.
    let mut result = overlap::ema(&overlap::ema(&mom, longperiod), shortperiod);
    let den = overlap::ema(&overlap::ema(&abs_mom, longperiod), shortperiod);
    for (r, &dv) in result.iter_mut().zip(den.iter()) {
        *r = if r.is_finite() && dv.is_finite() && dv != 0.0 {
            100.0 * *r / dv
        } else {
            f64::NAN
        };
    }
    let signal = overlap::ema(&result, signalperiod);
    (result, signal)
}

/// True range at bar `j >= 1`, with non-finite values folded to zero the way
/// the materialized version did before summing.
#[inline]
fn vortex_tr_at(high: &[f64], low: &[f64], close: &[f64], j: usize) -> f64 {
    let v = vol::true_range(high[j], low[j], close[j - 1]);
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Vortex Indicator: `+VI` and `−VI` over `timeperiod`.
///
/// The `+VM`, `−VM` and true-range series are held in three
/// `timeperiod`-slot rings instead of three full-length buffers, so each term
/// is computed exactly once and the kernel allocates only its two outputs
/// plus 3 × `timeperiod` scratch words that stay resident in L1.
pub fn vortex(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    if timeperiod < 1 || n < 2 || high.len() != n || low.len() != n || n <= timeperiod {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }
    let mut ring_plus = vec![0.0_f64; timeperiod];
    let mut ring_minus = vec![0.0_f64; timeperiod];
    let mut ring_tr = vec![0.0_f64; timeperiod];
    // Seeded in the same left-to-right order as the previous `iter().sum()`
    // over `plus_vm[1..=timeperiod]`, so each accumulated value is identical.
    let mut p_sum = 0.0_f64;
    let mut m_sum = 0.0_f64;
    let mut t_sum = 0.0_f64;
    for j in 1..=timeperiod {
        let pv = (high[j] - low[j - 1]).abs();
        let mv = (low[j] - high[j - 1]).abs();
        let tv = vortex_tr_at(high, low, close, j);
        p_sum += pv;
        m_sum += mv;
        t_sum += tv;
        ring_plus[j - 1] = pv;
        ring_minus[j - 1] = mv;
        ring_tr[j - 1] = tv;
    }
    // The first `timeperiod` bars keep their initialized `NaN`.
    let mut plus = vec![f64::NAN; n];
    let mut minus = vec![f64::NAN; n];
    if t_sum != 0.0 {
        plus[timeperiod] = p_sum / t_sum;
        minus[timeperiod] = m_sum / t_sum;
    }
    // Bar `i` occupies slot `(i - 1) % timeperiod`, which is also the slot of
    // the bar `i - timeperiod` leaving the window.
    let mut pos = 0usize;
    for i in (timeperiod + 1)..n {
        let pv = (high[i] - low[i - 1]).abs();
        let mv = (low[i] - high[i - 1]).abs();
        let tv = vortex_tr_at(high, low, close, i);
        p_sum += pv - ring_plus[pos];
        m_sum += mv - ring_minus[pos];
        t_sum += tv - ring_tr[pos];
        ring_plus[pos] = pv;
        ring_minus[pos] = mv;
        ring_tr[pos] = tv;
        pos += 1;
        if pos == timeperiod {
            pos = 0;
        }
        if t_sum != 0.0 {
            plus[i] = p_sum / t_sum;
            minus[i] = m_sum / t_sum;
        }
    }
    (plus, minus)
}

/// Schaff Trend Cycle: stochastic of MACD, double-smoothed (`d1`, `d2`).
pub fn stc(
    close: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    cycleperiod: usize,
    d1: usize,
    d2: usize,
) -> Vec<f64> {
    let n = close.len();
    if fastperiod < 1 || slowperiod < 1 || cycleperiod < 1 || d1 < 1 || d2 < 1 || n == 0 {
        return vec![f64::NAN; n];
    }
    // MACD lands in the fast-EMA buffer, and the final clamp is applied in
    // place to the last EMA: two full-length NaN-filled buffers fewer.
    let mut macd = overlap::ema(close, fastperiod);
    let slow = overlap::ema(close, slowperiod);
    for (m, &s) in macd.iter_mut().zip(slow.iter()) {
        *m = if m.is_finite() && s.is_finite() {
            *m - s
        } else {
            f64::NAN
        };
    }
    let stoch1 = stochastic_of(&macd, cycleperiod);
    let pf = overlap::ema(&stoch1, d1);
    let stoch2 = stochastic_of(&pf, cycleperiod);
    let mut result = overlap::ema(&stoch2, d2);
    for v in result.iter_mut() {
        *v = if v.is_finite() {
            v.clamp(0.0, 100.0)
        } else {
            f64::NAN
        };
    }
    result
}

/// Rolling stochastic of `src`: `100 (src − min) / (max − min)`.
///
/// One monotonic-deque traversal produces both extremes, and the result is
/// written back over the max buffer.
fn stochastic_of(src: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = src.len();
    if timeperiod < 1 {
        return vec![f64::NAN; n];
    }
    // Zero-initialized rather than NaN-filled: `sliding_min_max_into` writes
    // every slot, warmup included, so the fill is pure overhead.
    let mut hh = vec![0.0; n];
    let mut ll = vec![0.0; n];
    rolling::sliding_min_max_into(src, src, timeperiod, &mut hh, &mut ll);
    for i in 0..n {
        let high = hh[i];
        let low = ll[i];
        hh[i] = if !src[i].is_finite() || !high.is_finite() || !low.is_finite() {
            f64::NAN
        } else {
            let span = high - low;
            if span != 0.0 {
                100.0 * (src[i] - low) / span
            } else {
                0.0
            }
        };
    }
    hh
}

/// Gator Oscillator from the Alligator jaw / teeth / lips.
///
/// Returns `( |jaw − teeth|, −|teeth − lips| )`.
#[allow(clippy::too_many_arguments)]
pub fn gator(
    high: &[f64],
    low: &[f64],
    jaw_period: usize,
    jaw_shift: usize,
    teeth_period: usize,
    teeth_shift: usize,
    lips_period: usize,
    lips_shift: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (mut jaw, mut teeth, lips) = super::alligator(
        high,
        low,
        jaw_period,
        jaw_shift,
        teeth_period,
        teeth_shift,
        lips_period,
        lips_shift,
    );
    // Outputs land in the jaw and teeth buffers, snapshotting each bar's three
    // inputs first, so no extra full-length allocation is needed.
    for i in 0..jaw.len() {
        let (j, t, l) = (jaw[i], teeth[i], lips[i]);
        jaw[i] = if j.is_finite() && t.is_finite() {
            (j - t).abs()
        } else {
            f64::NAN
        };
        teeth[i] = if t.is_finite() && l.is_finite() {
            -(t - l).abs()
        } else {
            f64::NAN
        };
    }
    (jaw, teeth)
}

/// Coppock Curve: `WMA(ROC(roc1) + ROC(roc2), wma_period)`.
pub fn coppock(close: &[f64], wma_period: usize, roc1: usize, roc2: usize) -> Vec<f64> {
    let n = close.len();
    if wma_period < 1 || roc1 < 1 || roc2 < 1 {
        return vec![f64::NAN; n];
    }
    let mut sum = momentum::roc(close, roc1);
    let short_roc = momentum::roc(close, roc2);
    for (s, &r) in sum.iter_mut().zip(short_roc.iter()) {
        *s = if s.is_finite() && r.is_finite() {
            *s + r
        } else {
            f64::NAN
        };
    }
    wma_from_first_finite(&sum, wma_period)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;
    use crate::math_ops;

    fn linear_hl(n: usize) -> (Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (1..=n).map(|i| i as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=n).map(|i| i as f64 - 1.0).collect();
        (high, low)
    }

    /// Deterministic OHLC series with enough curvature that every branch of
    /// the fused kernels (rising, falling, flat spans) is exercised.
    fn synthetic(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                100.0 + 10.0 * (t * 0.07).sin() + 3.0 * (t * 0.31).cos()
            })
            .collect();
        let open: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - 0.25 + 0.1 * ((i % 7) as f64))
            .collect();
        let high: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 0.5 + 0.2 * ((i % 5) as f64))
            .collect();
        let low: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - 0.5 - 0.2 * ((i % 3) as f64))
            .collect();
        (open, high, low, close)
    }

    #[track_caller]
    fn assert_bits(got: &[f64], want: &[f64], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}[{i}]: got {g} ({:#x}) want {w} ({:#x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Pre-optimization implementations, kept verbatim as the bit-for-bit
    // oracle for the fused rewrites above.
    // -----------------------------------------------------------------------

    fn reference_sma_nan_aware(src: &[f64], timeperiod: usize) -> Vec<f64> {
        if timeperiod < 1 {
            return vec![f64::NAN; src.len()];
        }
        let sums = math_ops::rolling_sum(src, timeperiod);
        let p = timeperiod as f64;
        sums.into_iter()
            .map(|s| if s.is_finite() { s / p } else { f64::NAN })
            .collect()
    }

    fn reference_ao(high: &[f64], low: &[f64], fastperiod: usize, slowperiod: usize) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![f64::NAN; n];
        if fastperiod < 1 || slowperiod < 1 || n == 0 || low.len() != n {
            return result;
        }
        let median = price_transform::medprice(high, low);
        let fast = overlap::sma(&median, fastperiod);
        let slow = overlap::sma(&median, slowperiod);
        for i in 0..n {
            if fast[i].is_finite() && slow[i].is_finite() {
                result[i] = fast[i] - slow[i];
            }
        }
        result
    }

    fn reference_ac(
        high: &[f64],
        low: &[f64],
        fastperiod: usize,
        slowperiod: usize,
        timeperiod: usize,
    ) -> Vec<f64> {
        let n = high.len();
        if timeperiod < 1 {
            return vec![f64::NAN; n];
        }
        let awesome = reference_ao(high, low, fastperiod, slowperiod);
        let smooth = reference_sma_nan_aware(&awesome, timeperiod);
        let mut result = vec![f64::NAN; n];
        for i in 0..n {
            if awesome[i].is_finite() && smooth[i].is_finite() {
                result[i] = awesome[i] - smooth[i];
            }
        }
        result
    }

    fn reference_po(close: &[f64], fastperiod: usize, slowperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if fastperiod < 1 || slowperiod < 1 || n == 0 {
            return result;
        }
        let fast = overlap::sma(close, fastperiod);
        let slow = overlap::sma(close, slowperiod);
        for i in 0..n {
            if fast[i].is_finite() && slow[i].is_finite() {
                result[i] = fast[i] - slow[i];
            }
        }
        result
    }

    fn reference_dpo(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n == 0 {
            return result;
        }
        let shift = timeperiod / 2 + 1;
        let sma = overlap::sma(close, timeperiod);
        for i in 0..n {
            if i >= shift && sma[i].is_finite() {
                result[i] = close[i - shift] - sma[i];
            }
        }
        result
    }

    fn reference_rvi(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        timeperiod: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut rvi_out = vec![f64::NAN; n];
        let mut signal = vec![f64::NAN; n];
        if timeperiod < 1 || n < 4 || open.len() != n || high.len() != n || low.len() != n {
            return (rvi_out, signal);
        }
        let mut num = vec![f64::NAN; n];
        let mut den = vec![f64::NAN; n];
        for i in 3..n {
            num[i] = (close[i] - open[i])
                + 2.0 * (close[i - 1] - open[i - 1])
                + 2.0 * (close[i - 2] - open[i - 2])
                + (close[i - 3] - open[i - 3]);
            den[i] = (high[i] - low[i])
                + 2.0 * (high[i - 1] - low[i - 1])
                + 2.0 * (high[i - 2] - low[i - 2])
                + (high[i - 3] - low[i - 3]);
        }
        let num_s = reference_sma_nan_aware(&num, timeperiod);
        let den_s = reference_sma_nan_aware(&den, timeperiod);
        for i in 0..n {
            if num_s[i].is_finite() && den_s[i].is_finite() && den_s[i] != 0.0 {
                rvi_out[i] = num_s[i] / den_s[i];
            }
        }
        for i in 3..n {
            let a = rvi_out[i];
            let b = rvi_out[i - 1];
            let c = rvi_out[i - 2];
            let d = rvi_out[i - 3];
            if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() {
                signal[i] = (a + 2.0 * b + 2.0 * c + d) / 6.0;
            }
        }
        (rvi_out, signal)
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_kst(
        close: &[f64],
        roc1: usize,
        roc2: usize,
        roc3: usize,
        roc4: usize,
        sma1: usize,
        sma2: usize,
        sma3: usize,
        sma4: usize,
        signalperiod: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let nan = vec![f64::NAN; n];
        if n == 0
            || roc1 < 1
            || roc2 < 1
            || roc3 < 1
            || roc4 < 1
            || sma1 < 1
            || sma2 < 1
            || sma3 < 1
            || sma4 < 1
            || signalperiod < 1
        {
            return (nan.clone(), nan);
        }
        let r1 = reference_sma_nan_aware(&momentum::roc(close, roc1), sma1);
        let r2 = reference_sma_nan_aware(&momentum::roc(close, roc2), sma2);
        let r3 = reference_sma_nan_aware(&momentum::roc(close, roc3), sma3);
        let r4 = reference_sma_nan_aware(&momentum::roc(close, roc4), sma4);
        let mut kst_out = vec![f64::NAN; n];
        for i in 0..n {
            if r1[i].is_finite() && r2[i].is_finite() && r3[i].is_finite() && r4[i].is_finite() {
                kst_out[i] = r1[i] + 2.0 * r2[i] + 3.0 * r3[i] + 4.0 * r4[i];
            }
        }
        let signal = reference_sma_nan_aware(&kst_out, signalperiod);
        (kst_out, signal)
    }

    fn reference_tsi(
        close: &[f64],
        longperiod: usize,
        shortperiod: usize,
        signalperiod: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if longperiod < 1 || shortperiod < 1 || signalperiod < 1 || n == 0 {
            return (result, vec![f64::NAN; n]);
        }
        let mut mom = vec![f64::NAN; n];
        let mut abs_mom = vec![f64::NAN; n];
        for i in 1..n {
            let d = close[i] - close[i - 1];
            mom[i] = d;
            abs_mom[i] = d.abs();
        }
        let num = overlap::ema(&overlap::ema(&mom, longperiod), shortperiod);
        let den = overlap::ema(&overlap::ema(&abs_mom, longperiod), shortperiod);
        for i in 0..n {
            if num[i].is_finite() && den[i].is_finite() && den[i] != 0.0 {
                result[i] = 100.0 * num[i] / den[i];
            }
        }
        let signal = overlap::ema(&result, signalperiod);
        (result, signal)
    }

    fn reference_vortex(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        timeperiod: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut plus = vec![f64::NAN; n];
        let mut minus = vec![f64::NAN; n];
        if timeperiod < 1 || n < 2 || high.len() != n || low.len() != n {
            return (plus, minus);
        }
        let mut plus_vm = vec![0.0; n];
        let mut minus_vm = vec![0.0; n];
        let tr = vol::trange(high, low, close);
        for i in 1..n {
            plus_vm[i] = (high[i] - low[i - 1]).abs();
            minus_vm[i] = (low[i] - high[i - 1]).abs();
        }
        if n <= timeperiod {
            return (plus, minus);
        }
        let mut p_sum: f64 = plus_vm[1..=timeperiod].iter().sum();
        let mut m_sum: f64 = minus_vm[1..=timeperiod].iter().sum();
        let mut t_sum: f64 = tr[1..=timeperiod]
            .iter()
            .map(|v| if v.is_finite() { *v } else { 0.0 })
            .sum();
        if t_sum != 0.0 {
            plus[timeperiod] = p_sum / t_sum;
            minus[timeperiod] = m_sum / t_sum;
        }
        for i in (timeperiod + 1)..n {
            p_sum += plus_vm[i] - plus_vm[i - timeperiod];
            m_sum += minus_vm[i] - minus_vm[i - timeperiod];
            let add = if tr[i].is_finite() { tr[i] } else { 0.0 };
            let sub = if tr[i - timeperiod].is_finite() {
                tr[i - timeperiod]
            } else {
                0.0
            };
            t_sum += add - sub;
            if t_sum != 0.0 {
                plus[i] = p_sum / t_sum;
                minus[i] = m_sum / t_sum;
            }
        }
        (plus, minus)
    }

    fn reference_stochastic_of(src: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = src.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 {
            return result;
        }
        let hh = math::max(src, timeperiod);
        let ll = math::min(src, timeperiod);
        for i in 0..n {
            if !src[i].is_finite() || !hh[i].is_finite() || !ll[i].is_finite() {
                continue;
            }
            let span = hh[i] - ll[i];
            result[i] = if span != 0.0 {
                100.0 * (src[i] - ll[i]) / span
            } else {
                0.0
            };
        }
        result
    }

    fn reference_stc(
        close: &[f64],
        fastperiod: usize,
        slowperiod: usize,
        cycleperiod: usize,
        d1: usize,
        d2: usize,
    ) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if fastperiod < 1 || slowperiod < 1 || cycleperiod < 1 || d1 < 1 || d2 < 1 || n == 0 {
            return result;
        }
        let fast = overlap::ema(close, fastperiod);
        let slow = overlap::ema(close, slowperiod);
        let mut macd = vec![f64::NAN; n];
        for i in 0..n {
            if fast[i].is_finite() && slow[i].is_finite() {
                macd[i] = fast[i] - slow[i];
            }
        }
        let stoch1 = reference_stochastic_of(&macd, cycleperiod);
        let pf = overlap::ema(&stoch1, d1);
        let stoch2 = reference_stochastic_of(&pf, cycleperiod);
        let stc_out = overlap::ema(&stoch2, d2);
        for i in 0..n {
            if stc_out[i].is_finite() {
                result[i] = stc_out[i].clamp(0.0, 100.0);
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_gator(
        high: &[f64],
        low: &[f64],
        jaw_period: usize,
        jaw_shift: usize,
        teeth_period: usize,
        teeth_shift: usize,
        lips_period: usize,
        lips_shift: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let (jaw, teeth, lips) = super::super::alligator(
            high,
            low,
            jaw_period,
            jaw_shift,
            teeth_period,
            teeth_shift,
            lips_period,
            lips_shift,
        );
        let n = jaw.len();
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        for i in 0..n {
            if jaw[i].is_finite() && teeth[i].is_finite() {
                upper[i] = (jaw[i] - teeth[i]).abs();
            }
            if teeth[i].is_finite() && lips[i].is_finite() {
                lower[i] = -(teeth[i] - lips[i]).abs();
            }
        }
        (upper, lower)
    }

    fn reference_coppock(close: &[f64], wma_period: usize, roc1: usize, roc2: usize) -> Vec<f64> {
        let n = close.len();
        if wma_period < 1 || roc1 < 1 || roc2 < 1 {
            return vec![f64::NAN; n];
        }
        let long_roc = momentum::roc(close, roc1);
        let short_roc = momentum::roc(close, roc2);
        let mut sum = vec![f64::NAN; n];
        for i in 0..n {
            if long_roc[i].is_finite() && short_roc[i].is_finite() {
                sum[i] = long_roc[i] + short_roc[i];
            }
        }
        wma_from_first_finite(&sum, wma_period)
    }

    // -----------------------------------------------------------------------
    // Bit-for-bit equivalence of the fused kernels with the two-pass originals
    // -----------------------------------------------------------------------

    #[test]
    fn fused_kernels_are_bit_identical_to_two_pass_originals() {
        for n in [0usize, 1, 3, 4, 9, 41, 137, 512] {
            let (o, h, l, c) = synthetic(n);

            assert_bits(&ao(&h, &l, 5, 34), &reference_ao(&h, &l, 5, 34), "ao");
            assert_bits(&ac(&h, &l, 5, 34, 5), &reference_ac(&h, &l, 5, 34, 5), "ac");
            assert_bits(&po(&c, 10, 21), &reference_po(&c, 10, 21), "po");
            assert_bits(&dpo(&c, 20), &reference_dpo(&c, 20), "dpo");
            assert_bits(
                &coppock(&c, 10, 14, 11),
                &reference_coppock(&c, 10, 14, 11),
                "coppock",
            );
            assert_bits(
                &stc(&c, 23, 50, 10, 3, 3),
                &reference_stc(&c, 23, 50, 10, 3, 3),
                "stc",
            );

            let (got, want) = (rvi(&o, &h, &l, &c, 10), reference_rvi(&o, &h, &l, &c, 10));
            assert_bits(&got.0, &want.0, "rvi line");
            assert_bits(&got.1, &want.1, "rvi signal");

            let (got, want) = (
                kst(&c, 10, 15, 20, 30, 10, 10, 10, 15, 9),
                reference_kst(&c, 10, 15, 20, 30, 10, 10, 10, 15, 9),
            );
            assert_bits(&got.0, &want.0, "kst line");
            assert_bits(&got.1, &want.1, "kst signal");

            let (got, want) = (tsi(&c, 25, 13, 13), reference_tsi(&c, 25, 13, 13));
            assert_bits(&got.0, &want.0, "tsi line");
            assert_bits(&got.1, &want.1, "tsi signal");

            let (got, want) = (vortex(&h, &l, &c, 14), reference_vortex(&h, &l, &c, 14));
            assert_bits(&got.0, &want.0, "vortex plus");
            assert_bits(&got.1, &want.1, "vortex minus");

            let (got, want) = (
                gator(&h, &l, 13, 8, 8, 5, 5, 3),
                reference_gator(&h, &l, 13, 8, 8, 5, 5, 3),
            );
            assert_bits(&got.0, &want.0, "gator upper");
            assert_bits(&got.1, &want.1, "gator lower");
        }
    }

    /// A 20k-bar series crosses `rolling::RESEED_INTERVAL` (8192) twice. The
    /// fused accumulators deliberately mirror `rolling_sum`'s non-reseeding
    /// recurrence, so there must be no divergence anywhere along it.
    #[test]
    fn fused_kernels_stay_bit_identical_across_20k_bars() {
        let n = 20_000;
        let (o, h, l, c) = synthetic(n);

        let (got, want) = (
            kst(&c, 10, 15, 20, 30, 10, 10, 10, 15, 9),
            reference_kst(&c, 10, 15, 20, 30, 10, 10, 10, 15, 9),
        );
        assert_bits(&got.0, &want.0, "kst line 20k");
        assert_bits(&got.1, &want.1, "kst signal 20k");

        let (got, want) = (rvi(&o, &h, &l, &c, 10), reference_rvi(&o, &h, &l, &c, 10));
        assert_bits(&got.0, &want.0, "rvi line 20k");
        assert_bits(&got.1, &want.1, "rvi signal 20k");

        let (got, want) = (vortex(&h, &l, &c, 14), reference_vortex(&h, &l, &c, 14));
        assert_bits(&got.0, &want.0, "vortex plus 20k");
        assert_bits(&got.1, &want.1, "vortex minus 20k");

        assert_bits(
            &ac(&h, &l, 5, 34, 5),
            &reference_ac(&h, &l, 5, 34, 5),
            "ac 20k",
        );
        assert_bits(
            &stc(&c, 23, 50, 10, 3, 3),
            &reference_stc(&c, 23, 50, 10, 3, 3),
            "stc 20k",
        );
    }

    // -----------------------------------------------------------------------
    // Localized-NaN semantics of the fused rolling accumulators
    // -----------------------------------------------------------------------

    /// One `NaN` in `open` poisons the four weighted `num` terms at
    /// `j..=j+3`, hence exactly `timeperiod + 3` RVI outputs, and the kernel
    /// then recovers — the accumulator is not permanently contaminated.
    #[test]
    fn rvi_nan_corrupts_exactly_timeperiod_plus_three_then_recovers() {
        let p = 10usize;
        let n = 200usize;
        let j = 60usize;
        let (mut o, h, l, c) = synthetic(n);
        o[j] = f64::NAN;

        let (line, signal) = rvi(&o, &h, &l, &c, p);
        let warmup = 3 + p - 1;
        let corrupted: Vec<usize> = (warmup..n).filter(|&i| line[i].is_nan()).collect();
        assert_eq!(
            corrupted.len(),
            p + 3,
            "expected exactly {} corrupted RVI outputs, got {:?}",
            p + 3,
            corrupted
        );
        assert_eq!(corrupted.first().copied(), Some(j));
        assert_eq!(corrupted.last().copied(), Some(j + p + 2));
        assert!(line[j + p + 3].is_finite(), "RVI did not recover");
        assert!(signal[j + p + 6].is_finite(), "RVI signal did not recover");
        assert!(line[n - 1].is_finite());

        // Same NaN, same outputs as the two-pass original.
        let want = reference_rvi(&o, &h, &l, &c, p);
        assert_bits(&line, &want.0, "rvi line with NaN");
        assert_bits(&signal, &want.1, "rvi signal with NaN");
    }

    /// With all four ROC/SMA pairs equal, one `NaN` in `close` reaches the
    /// stream twice (as `close[i]` and as `close[i - roc]`), so it corrupts
    /// exactly `2 * sma_period` outputs in two disjoint runs of `sma_period`.
    #[test]
    fn kst_nan_corrupts_exactly_two_windows_then_recovers() {
        let (r, p) = (20usize, 5usize);
        let n = 200usize;
        let j = 60usize;
        let (_, _, _, mut c) = synthetic(n);
        c[j] = f64::NAN;

        let (line, signal) = kst(&c, r, r, r, r, p, p, p, p, 1);
        let warmup = r + p - 1;
        let corrupted: Vec<usize> = (warmup..n).filter(|&i| line[i].is_nan()).collect();
        assert_eq!(
            corrupted,
            (j..j + p).chain(j + r..j + r + p).collect::<Vec<_>>(),
            "KST NaN did not stay localized"
        );
        assert!(line[j + p].is_finite(), "KST did not recover after run 1");
        assert!(
            line[j + r + p].is_finite(),
            "KST did not recover after run 2"
        );
        assert!(line[n - 1].is_finite());
        // signalperiod == 1: the signal tracks the line to within the drift of
        // a one-slot running sum, and recovers on the same bars.
        assert!(signal[j + p].is_finite() && signal[j + r + p].is_finite());
        assert!((signal[n - 1] - line[n - 1]).abs() < 1e-9);

        let want = reference_kst(&c, r, r, r, r, p, p, p, p, 1);
        assert_bits(&line, &want.0, "kst line with NaN");
        assert_bits(&signal, &want.1, "kst signal with NaN");
    }

    /// `ac` smooths the AO series with the same fused accumulator; a `NaN`
    /// reaching it must not poison the tail.
    #[test]
    fn ac_nan_smoother_recovers() {
        let n = 200usize;
        let (_, h, l, _) = synthetic(n);
        let mut awesome = ao(&h, &l, 5, 34);
        awesome[100] = f64::NAN;
        // Drive the smoother directly: `ao` itself is NaN-poisoned by
        // `overlap::sma`, which is out of scope for this file.
        let p = 5usize;
        let mut smoother = NanAwareSum::new(p);
        let means: Vec<f64> = (0..n)
            .map(|i| {
                let old = if i >= p { awesome[i - p] } else { f64::NAN };
                smoother.advance(i, awesome[i], old)
            })
            .collect();
        let want = reference_sma_nan_aware(&awesome, p);
        assert_bits(&means, &want, "ac smoother with NaN");
        let corrupted = (38..n).filter(|&i| means[i].is_nan()).count();
        assert_eq!(corrupted, p, "expected exactly {p} corrupted means");
        assert!(means[n - 1].is_finite(), "smoother did not recover");
    }

    // -----------------------------------------------------------------------
    // Pre-existing behavioural tests
    // -----------------------------------------------------------------------

    #[test]
    fn ao_equals_sma_difference() {
        let (h, l) = linear_hl(40);
        let got = ao(&h, &l, 5, 34);
        let med = price_transform::medprice(&h, &l);
        let exp_f = overlap::sma(&med, 5);
        let exp_s = overlap::sma(&med, 34);
        assert!(got[32].is_nan());
        assert!((got[33] - (exp_f[33] - exp_s[33])).abs() < 1e-12);
    }

    #[test]
    fn cho_matches_adosc() {
        let n = 30;
        let h: Vec<f64> = (1..=n).map(|i| i as f64 + 1.0).collect();
        let l: Vec<f64> = (1..=n).map(|i| i as f64 - 1.0).collect();
        let c: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let v = vec![1000.0; n];
        let a = cho(&h, &l, &c, &v, 3, 10);
        let b = volume::adosc(&h, &l, &c, &v, 3, 10);
        for i in 0..n {
            assert!((a[i].is_nan() && b[i].is_nan()) || (a[i] - b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn po_zero_when_periods_equal() {
        let c: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let result = po(&c, 3, 3);
        for &v in result.iter().filter(|v| v.is_finite()) {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn dpo_shift_identity() {
        let c: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let result = dpo(&c, 4);
        // shift = 4/2+1 = 3; SMA(4)[3] = 2.5; close[0] - 2.5 = -1.5
        assert!(result[2].is_nan());
        assert!((result[3] - (1.0 - 2.5)).abs() < 1e-12);
    }

    #[test]
    fn tsi_constant_change_is_100() {
        let c: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let (result, _signal) = tsi(&c, 5, 3, 3);
        let last = *result.iter().rev().find(|v| v.is_finite()).unwrap();
        assert!((last - 100.0).abs() < 1e-8, "{last}");
    }

    #[test]
    fn coppock_empty() {
        assert!(coppock(&[], 10, 14, 11).is_empty());
    }

    #[test]
    fn ac_kst_rvi_coppock_are_finite_after_warmup() {
        let n = 80;
        let (h, l) = linear_hl(n);
        let c: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let o: Vec<f64> = (1..=n).map(|i| i as f64 - 0.25).collect();
        let ac_line = ac(&h, &l, 5, 34, 5);
        assert!(
            ac_line.iter().any(|v| v.is_finite()),
            "AC poisoned by NaN SMA"
        );
        let (kst_line, kst_sig) = kst(&c, 5, 8, 10, 12, 3, 3, 3, 4, 3);
        assert!(
            kst_line.iter().any(|v| v.is_finite()),
            "KST poisoned by NaN SMA"
        );
        assert!(kst_sig.iter().any(|v| v.is_finite()));
        let (rvi_line, rvi_sig) = rvi(&o, &h, &l, &c, 4);
        assert!(
            rvi_line.iter().any(|v| v.is_finite()),
            "RVI poisoned by NaN SMA"
        );
        assert!(rvi_sig.iter().any(|v| v.is_finite()));
        let copp = coppock(&c, 4, 5, 3);
        assert!(
            copp.iter().any(|v| v.is_finite()),
            "Coppock poisoned by NaN WMA"
        );
    }

    #[test]
    fn ao_length_mismatch_returns_nan() {
        let (h, _) = linear_hl(40);
        let short = vec![1.0; 10];
        let long = vec![1.0; 50];
        for l in [short, long] {
            let result = ao(&h, &l, 5, 34);
            assert_eq!(result.len(), h.len());
            assert!(result.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn rvi_length_mismatch_returns_nan() {
        let n = 20;
        let (h, l) = linear_hl(n);
        let c: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let o: Vec<f64> = c.iter().map(|v| v - 0.25).collect();
        let short = vec![1.0; 5];
        let long = vec![1.0; n + 3];
        for other in [short, long] {
            for (line, signal) in [
                rvi(&other, &h, &l, &c, 4),
                rvi(&o, &other, &l, &c, 4),
                rvi(&o, &h, &other, &c, 4),
            ] {
                assert_eq!(line.len(), n);
                assert_eq!(signal.len(), n);
                assert!(line.iter().all(|v| v.is_nan()));
                assert!(signal.iter().all(|v| v.is_nan()));
            }
        }
    }

    #[test]
    fn vortex_length_mismatch_returns_nan() {
        let n = 20;
        let (h, l) = linear_hl(n);
        let c: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let short = vec![1.0; 5];
        let long = vec![1.0; n + 3];
        for other in [short, long] {
            for (plus, minus) in [vortex(&other, &l, &c, 5), vortex(&h, &other, &c, 5)] {
                assert_eq!(plus.len(), n);
                assert_eq!(minus.len(), n);
                assert!(plus.iter().all(|v| v.is_nan()));
                assert!(minus.iter().all(|v| v.is_nan()));
            }
        }
    }
}
