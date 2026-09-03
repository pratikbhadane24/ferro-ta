//! Trend / overlap extended indicators.
//!
//! Published textbook / journal formulas. Leading warmup bars are `NaN`.

use crate::math;
use crate::momentum;
use crate::overlap;
use crate::volatility;

// ---------------------------------------------------------------------------
// ALMA — Arnaud Legoux Moving Average
// ---------------------------------------------------------------------------

/// Arnaud Legoux Moving Average.
///
/// Gaussian weights over a `timeperiod` window. The peak sits at
/// `floor(offset * (timeperiod - 1))` from the oldest bar, and the width is
/// `timeperiod / sigma`. Output is the weighted sum divided by the weight
/// sum. The first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `close` — price series.
/// * `timeperiod` — window length (typically 21).
/// * `offset` — Gaussian peak location in `[0, 1]` (typically 0.85).
/// * `sigma` — Gaussian width control (typically 6). Must be `> 0`.
pub fn alma(close: &[f64], timeperiod: usize, offset: f64, sigma: f64) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod || !sigma.is_finite() || sigma <= 0.0 {
        return result;
    }

    let m = (offset * (timeperiod - 1) as f64).floor();
    let s = timeperiod as f64 / sigma;
    let two_s2 = 2.0 * s * s;
    let mut weights = vec![0.0; timeperiod];
    let mut wsum = 0.0;
    for (k, w) in weights.iter_mut().enumerate() {
        let d = k as f64 - m;
        *w = (-d * d / two_s2).exp();
        wsum += *w;
    }
    if wsum == 0.0 || !wsum.is_finite() {
        return result;
    }

    for i in (timeperiod - 1)..n {
        let start = i + 1 - timeperiod;
        let mut acc = 0.0;
        for (k, &w) in weights.iter().enumerate() {
            acc += close[start + k] * w;
        }
        result[i] = acc / wsum;
    }
    result
}

// ---------------------------------------------------------------------------
// ZLEMA — Zero-Lag EMA
// ---------------------------------------------------------------------------

/// Zero-lag exponential moving average.
///
/// `lag = floor((timeperiod - 1) / 2)`, then
/// `ZLEMA = EMA(close + (close - close[lag]), timeperiod)`.
/// The EMA is SMA-seeded from the first window of finite de-lagged values.
pub fn zlema(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 {
        return vec![f64::NAN; n];
    }
    let lag = (timeperiod - 1) / 2;
    let mut adjusted = vec![f64::NAN; n];
    for i in lag..n {
        adjusted[i] = close[i] + (close[i] - close[i - lag]);
    }
    overlap::ema_from_first_finite(&adjusted, timeperiod)
}

// ---------------------------------------------------------------------------
// FRAMA — Fractal Adaptive Moving Average
// ---------------------------------------------------------------------------

/// Fractal Adaptive Moving Average (Ehlers).
///
/// Fractal dimension from the normalized ranges of the two halves of the
/// window versus the full window:
///
/// `dim = (ln(N1 + N2) - ln(N3)) / ln(2)`
///
/// then `alpha = exp(-4.6 * (dim - 1))` clipped to `[0.01, 1]`, and
/// `FRAMA = alpha * close + (1 - alpha) * FRAMA[1]`.
/// Seeded with `close` at the first full window (`timeperiod - 1`).
/// `timeperiod` should be even; odd periods use `floor(timeperiod / 2)` halves.
pub fn frama(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 2 || n < timeperiod {
        return result;
    }

    let half = (timeperiod / 2).max(1);
    let hh_half = math::max(close, half);
    let ll_half = math::min(close, half);
    let hh_full = math::max(close, timeperiod);
    let ll_full = math::min(close, timeperiod);
    let half_f = half as f64;
    let period_f = timeperiod as f64;
    let ln2 = std::f64::consts::LN_2;

    let mut seeded = false;
    for i in (timeperiod - 1)..n {
        let n1 = (hh_half[i] - ll_half[i]) / half_f;
        let n2 = if i >= half {
            (hh_half[i - half] - ll_half[i - half]) / half_f
        } else {
            f64::NAN
        };
        let n3 = (hh_full[i] - ll_full[i]) / period_f;
        let alpha = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
            let dim = ((n1 + n2).ln() - n3.ln()) / ln2;
            (-4.6 * (dim - 1.0)).exp().clamp(0.01, 1.0)
        } else {
            1.0
        };

        if !seeded {
            result[i] = close[i];
            seeded = true;
        } else {
            result[i] = alpha * close[i] + (1.0 - alpha) * result[i - 1];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// MCGINLEY — McGinley Dynamic
// ---------------------------------------------------------------------------

/// McGinley Dynamic.
///
/// `MD = MD[1] + (close - MD[1]) / (timeperiod * (close / MD[1])^4)`.
/// Seeded with the SMA of the first `timeperiod` bars (first `timeperiod - 1`
/// outputs are `NaN`).
pub fn mcginley(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }

    let p = timeperiod as f64;
    let seed: f64 = close[..timeperiod].iter().sum::<f64>() / p;
    result[timeperiod - 1] = seed;
    let mut md = seed;
    for i in timeperiod..n {
        md = mcginley_step(md, close[i], p);
        result[i] = md;
    }
    result
}

#[inline]
fn mcginley_step(md: f64, price: f64, period: f64) -> f64 {
    if !md.is_finite() || md == 0.0 || !price.is_finite() {
        return price;
    }
    let ratio = price / md;
    let denom = period * ratio.powi(4);
    if !denom.is_finite() || denom == 0.0 {
        return price;
    }
    md + (price - md) / denom
}

// ---------------------------------------------------------------------------
// VIDYA — Variable Index Dynamic Average
// ---------------------------------------------------------------------------

/// Variable Index Dynamic Average (Chande).
///
/// Adaptive EMA whose smoothing is scaled by `|CMO(cmo_period)| / 100`:
///
/// `k = 2 / (timeperiod + 1)`,
/// `VIDYA = k * |CMO| * close + (1 - k * |CMO|) * VIDYA[1]`.
///
/// Seeded with `close` on the first finite CMO bar.
pub fn vidya(close: &[f64], timeperiod: usize, cmo_period: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || cmo_period < 1 {
        return result;
    }

    let cmo = momentum::cmo(close, cmo_period);
    let k = 2.0 / (timeperiod as f64 + 1.0);
    let mut prev = f64::NAN;
    for i in 0..n {
        let c = cmo[i];
        if !c.is_finite() {
            continue;
        }
        let sc = k * (c.abs() / 100.0);
        if !prev.is_finite() {
            prev = close[i];
            result[i] = prev;
        } else {
            prev = sc * close[i] + (1.0 - sc) * prev;
            result[i] = prev;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// ALLIGATOR — Bill Williams
// ---------------------------------------------------------------------------

/// Bill Williams Alligator (jaw / teeth / lips).
///
/// Each line is an SMMA of the median price `(high + low) / 2`, then shifted
/// forward by the corresponding offset (`jaw[i] = SMMA[i - jaw_shift]`).
///
/// Defaults in the Python wrapper are 13/8, 8/5, 5/3.
pub fn alligator(
    high: &[f64],
    low: &[f64],
    jaw_period: usize,
    jaw_shift: usize,
    teeth_period: usize,
    teeth_shift: usize,
    lips_period: usize,
    lips_shift: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    if n == 0 || n != low.len() || jaw_period < 1 || teeth_period < 1 || lips_period < 1 {
        return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    }

    let mut median = vec![0.0; n];
    for i in 0..n {
        median[i] = (high[i] + low[i]) * 0.5;
    }

    let jaw = shift_forward(&smma(&median, jaw_period), jaw_shift);
    let teeth = shift_forward(&smma(&median, teeth_period), teeth_shift);
    let lips = shift_forward(&smma(&median, lips_period), lips_shift);
    (jaw, teeth, lips)
}

/// Wilder / SMMA: SMA seed, then `(prev * (period - 1) + x) / period`.
fn smma(src: &[f64], period: usize) -> Vec<f64> {
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    if period < 1 || n < period {
        return out;
    }
    if period == 1 {
        return src.to_vec();
    }
    let p = period as f64;
    let seed: f64 = src[..period].iter().sum::<f64>() / p;
    out[period - 1] = seed;
    for i in period..n {
        out[i] = (out[i - 1] * (p - 1.0) + src[i]) / p;
    }
    out
}

/// Shift a series forward: `out[i] = src[i - shift]` (leading `shift` NaNs).
fn shift_forward(src: &[f64], shift: usize) -> Vec<f64> {
    let n = src.len();
    if shift == 0 {
        return src.to_vec();
    }
    let mut out = vec![f64::NAN; n];
    if shift < n {
        out[shift..].copy_from_slice(&src[..n - shift]);
    }
    out
}

// ---------------------------------------------------------------------------
// MA_ENVELOPES
// ---------------------------------------------------------------------------

/// Moving-average envelopes: `MA * (1 ± percent / 100)`.
///
/// `matype` matches [`overlap::ma`] (0=SMA … 7=T3). Returns
/// `(upper, middle, lower)`.
pub fn ma_envelopes(
    close: &[f64],
    timeperiod: usize,
    percent: f64,
    matype: u8,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let middle = overlap::ma(close, timeperiod, matype);
    let n = middle.len();
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    let up = 1.0 + percent / 100.0;
    let dn = 1.0 - percent / 100.0;
    for i in 0..n {
        let m = middle[i];
        if m.is_finite() {
            upper[i] = m * up;
            lower[i] = m * dn;
        }
    }
    (upper, middle, lower)
}

// ---------------------------------------------------------------------------
// CHANDE_KROLL_STOP
// ---------------------------------------------------------------------------

/// Chande Kroll Stop.
///
/// First stops: `HH(high, p) - x * ATR(p)` and `LL(low, p) + x * ATR(p)`.
/// Final stops: `HH(first_long, q)` and `LL(first_short, q)`.
///
/// `timeperiod` is `p` (HH/LL and ATR). `stop_period` is `q`. `multiplier` is `x`.
pub fn chande_kroll_stop(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
    multiplier: f64,
    stop_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    if timeperiod < 1 || stop_period < 1 || n == 0 {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }

    let hh = math::max(high, timeperiod);
    let ll = math::min(low, timeperiod);
    let atr = volatility::atr(high, low, close, timeperiod);

    let mut first_long = vec![f64::NAN; n];
    let mut first_short = vec![f64::NAN; n];
    let mut first_valid = None;
    for i in 0..n {
        if hh[i].is_finite() && ll[i].is_finite() && atr[i].is_finite() {
            first_long[i] = hh[i] - multiplier * atr[i];
            first_short[i] = ll[i] + multiplier * atr[i];
            if first_valid.is_none() {
                first_valid = Some(i);
            }
        }
    }

    let Some(start) = first_valid else {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    };
    let long_stop = rolling_max_from(&first_long, stop_period, start);
    let short_stop = rolling_min_from(&first_short, stop_period, start);
    (long_stop, short_stop)
}

fn rolling_max_from(src: &[f64], period: usize, start: usize) -> Vec<f64> {
    rolling_extremum_from(src, period, start, true)
}

fn rolling_min_from(src: &[f64], period: usize, start: usize) -> Vec<f64> {
    rolling_extremum_from(src, period, start, false)
}

fn rolling_extremum_from(src: &[f64], period: usize, start: usize, is_max: bool) -> Vec<f64> {
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    if period < 1 || start >= n {
        return out;
    }
    let first = start + period - 1;
    if first >= n {
        return out;
    }
    for i in first..n {
        let lo = i + 1 - period;
        let window = &src[lo..=i];
        if window.iter().any(|v| v.is_nan()) {
            continue;
        }
        out[i] = if is_max {
            window.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        } else {
            window.iter().copied().fold(f64::INFINITY, f64::min)
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn linear(n: usize) -> Vec<f64> {
        (1..=n).map(|i| i as f64).collect()
    }

    fn assert_nan_prefix(xs: &[f64], n: usize) {
        for (i, v) in xs.iter().take(n).enumerate() {
            assert!(v.is_nan(), "expected NaN at {i}, got {v}");
        }
    }

    #[test]
    fn alma_linear_symmetric_period3() {
        // period=3, offset=0.85 → m=floor(1.7)=1, weights symmetric → midpoint.
        let close = linear(5);
        let result = alma(&close, 3, 0.85, 6.0);
        assert_nan_prefix(&result, 2);
        assert!((result[2] - 2.0).abs() < TOL);
        assert!((result[3] - 3.0).abs() < TOL);
        assert!((result[4] - 4.0).abs() < TOL);
    }

    #[test]
    fn alma_constant_is_constant() {
        let close = vec![7.0; 10];
        let result = alma(&close, 5, 0.85, 6.0);
        assert_nan_prefix(&result, 4);
        for v in result.iter().skip(4) {
            assert!((v - 7.0).abs() < TOL);
        }
    }

    #[test]
    fn alma_offset_peaks_recent_vs_old() {
        let close = vec![10.0, 10.0, 10.0, 100.0];
        let recent = alma(&close, 4, 1.0, 6.0);
        let oldest = alma(&close, 4, 0.0, 6.0);
        assert!(recent[3] > oldest[3]);
        assert!(recent[3] > 40.0);
        assert!(oldest[3] < 40.0);
    }

    #[test]
    fn zlema_linear_tracks_close() {
        // lag=1, adjusted = close + 1 on a unit-slope series; SMA-seeded EMA
        // of that series equals close after warmup lag + timeperiod - 1.
        let close = linear(10);
        let result = zlema(&close, 3);
        assert_nan_prefix(&result, 3);
        for i in 3..10 {
            assert!(
                (result[i] - close[i]).abs() < TOL,
                "ZLEMA[{i}]={}, close={}",
                result[i],
                close[i]
            );
        }
    }

    #[test]
    fn frama_linear_alpha_clamps_to_one() {
        let close = linear(8);
        let result = frama(&close, 4);
        assert_nan_prefix(&result, 3);
        for i in 3..8 {
            assert!(
                (result[i] - close[i]).abs() < TOL,
                "FRAMA[{i}]={}, close={}",
                result[i],
                close[i]
            );
        }
    }

    #[test]
    fn frama_oscillating_dim2() {
        let close = vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let result = frama(&close, 4);
        assert_nan_prefix(&result, 3);
        assert!((result[3] - 3.0).abs() < TOL);
        let alpha = (-4.6_f64).exp();
        let expected = alpha * 1.0 + (1.0 - alpha) * 3.0;
        assert!((result[4] - expected).abs() < 1e-12);
    }

    #[test]
    fn mcginley_constant_is_constant() {
        let close = vec![10.0; 8];
        let result = mcginley(&close, 3);
        assert_nan_prefix(&result, 2);
        for v in result.iter().skip(2) {
            assert!((v - 10.0).abs() < TOL);
        }
    }

    #[test]
    fn mcginley_first_step() {
        let close = linear(8);
        let result = mcginley(&close, 3);
        assert_nan_prefix(&result, 2);
        assert!((result[2] - 2.0).abs() < TOL);
        // MD = 2 + (4-2) / (3 * (4/2)^4) = 2 + 2/48 = 49/24
        assert!((result[3] - 49.0 / 24.0).abs() < TOL);
    }

    #[test]
    fn vidya_cmo_period3_golden() {
        // Same close / CMO goldens as momentum::cmo_wilder_golden_period3.
        let close = vec![1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0];
        let result = vidya(&close, 3, 3);
        assert_nan_prefix(&result, 3);
        assert!((result[3] - 2.0).abs() < TOL);
        // k=0.5, |CMO[4]|=200/3 → sc = 0.5 * (2/3) = 1/3
        // VIDYA[4] = (1/3)*4 + (2/3)*2 = 8/3
        assert!((result[4] - 8.0 / 3.0).abs() < TOL);
    }

    #[test]
    fn alligator_smma_shift() {
        let high: Vec<f64> = (0..8).map(|i| 11.0 + i as f64).collect();
        let low: Vec<f64> = (0..8).map(|i| 9.0 + i as f64).collect();
        // median = 10,11,12,13,14,15,16,17
        let (jaw, teeth, lips) = alligator(&high, &low, 3, 2, 2, 1, 2, 1);
        // SMMA3 seed at i=2: (10+11+12)/3=11, then shift 2 → jaw[4]=11
        assert_nan_prefix(&jaw, 4);
        assert!((jaw[4] - 11.0).abs() < TOL);
        // SMMA3[3] = (11*2 + 13)/3 = 35/3, jaw[5]=35/3
        assert!((jaw[5] - 35.0 / 3.0).abs() < TOL);
        // SMMA2 seed at i=1: 10.5, shift 1 → teeth[2]=10.5
        assert!((teeth[2] - 10.5).abs() < TOL);
        assert!((lips[2] - 10.5).abs() < TOL);
        assert_eq!(jaw.len(), 8);
    }

    #[test]
    fn ma_envelopes_sma_percent() {
        let close = linear(5);
        let (upper, middle, lower) = ma_envelopes(&close, 3, 10.0, 0);
        assert_nan_prefix(&middle, 2);
        assert!((middle[2] - 2.0).abs() < TOL);
        assert!((upper[2] - 2.2).abs() < TOL);
        assert!((lower[2] - 1.8).abs() < TOL);
        assert!((middle[4] - 4.0).abs() < TOL);
        assert!((upper[4] - 4.4).abs() < TOL);
        assert!((lower[4] - 3.6).abs() < TOL);
    }

    #[test]
    fn chande_kroll_stop_p3_q2() {
        let high = vec![10.0, 12.0, 11.0, 13.0, 14.0, 15.0];
        let low = vec![8.0, 9.0, 8.0, 10.0, 11.0, 12.0];
        let close = vec![9.0, 11.0, 10.0, 12.0, 13.0, 14.0];
        let (long_stop, short_stop) = chande_kroll_stop(&high, &low, &close, 3, 1.0, 2);
        // ATR(3) first finite at 3; q=2 → first output at 4.
        assert_nan_prefix(&long_stop, 4);
        assert_nan_prefix(&short_stop, 4);
        assert!((long_stop[4] - 11.0).abs() < TOL);
        assert!((long_stop[5] - 12.0).abs() < TOL);
        assert!((short_stop[4] - 11.0).abs() < TOL);
        assert!((short_stop[5] - 11.0).abs() < TOL);
    }

    #[test]
    fn short_input_is_all_nan() {
        let close = vec![1.0, 2.0];
        assert!(alma(&close, 5, 0.85, 6.0).iter().all(|v| v.is_nan()));
        assert!(zlema(&close, 5).iter().all(|v| v.is_nan()));
        assert!(frama(&close, 8).iter().all(|v| v.is_nan()));
        assert!(mcginley(&close, 5).iter().all(|v| v.is_nan()));
    }
}
