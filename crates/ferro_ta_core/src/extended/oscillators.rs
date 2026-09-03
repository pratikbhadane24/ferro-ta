//! Oscillator extended indicators (AO, AC, KST, TSI, STC, and related).

use crate::math;
use crate::math_ops;
use crate::momentum;
use crate::overlap;
use crate::price_transform;
use crate::volatility as vol;
use crate::volume;

fn sma_nan_aware(src: &[f64], timeperiod: usize) -> Vec<f64> {
    if timeperiod < 1 {
        return vec![f64::NAN; src.len()];
    }
    let sums = math_ops::rolling_sum(src, timeperiod);
    let p = timeperiod as f64;
    sums.into_iter()
        .map(|s| if s.is_finite() { s / p } else { f64::NAN })
        .collect()
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
    let mut result = vec![f64::NAN; n];
    if fastperiod < 1 || slowperiod < 1 || n == 0 {
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
    let smooth = sma_nan_aware(&awesome, timeperiod);
    let mut result = vec![f64::NAN; n];
    for i in 0..n {
        if awesome[i].is_finite() && smooth[i].is_finite() {
            result[i] = awesome[i] - smooth[i];
        }
    }
    result
}

/// Price Oscillator (SMA): `SMA(close, fast) − SMA(close, slow)`.
pub fn po(close: &[f64], fastperiod: usize, slowperiod: usize) -> Vec<f64> {
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

/// Detrended Price Oscillator: `close[i − shift] − SMA(close, timeperiod)`,
/// where `shift = timeperiod / 2 + 1`.
pub fn dpo(close: &[f64], timeperiod: usize) -> Vec<f64> {
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

/// Relative Vigor Index and its 4-bar weighted signal.
///
/// `num = (C−O) + 2(C1−O1) + 2(C2−O2) + (C3−O3)`,
/// `den = (H−L) + 2(H1−L1) + 2(H2−L2) + (H3−L3)`,
/// `RVI = SMA(num) / SMA(den)`,
/// `signal = (RVI + 2 RVI1 + 2 RVI2 + RVI3) / 6`.
pub fn rvi(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut rvi_out = vec![f64::NAN; n];
    let mut signal = vec![f64::NAN; n];
    if timeperiod < 1 || n < 4 {
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
    let num_s = sma_nan_aware(&num, timeperiod);
    let den_s = sma_nan_aware(&den, timeperiod);
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

/// Know Sure Thing: weighted sum of four ROC SMAs, plus a signal SMA.
///
/// Defaults match the classic 10/15/20/30 ROC windows with 10/10/10/15
/// smoothers and a 9-bar signal.
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
    let r1 = sma_nan_aware(&momentum::roc(close, roc1), sma1);
    let r2 = sma_nan_aware(&momentum::roc(close, roc2), sma2);
    let r3 = sma_nan_aware(&momentum::roc(close, roc3), sma3);
    let r4 = sma_nan_aware(&momentum::roc(close, roc4), sma4);
    let mut kst_out = vec![f64::NAN; n];
    for i in 0..n {
        if r1[i].is_finite() && r2[i].is_finite() && r3[i].is_finite() && r4[i].is_finite() {
            kst_out[i] = r1[i] + 2.0 * r2[i] + 3.0 * r3[i] + 4.0 * r4[i];
        }
    }
    let signal = sma_nan_aware(&kst_out, signalperiod);
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

/// Vortex Indicator: `+VI` and `−VI` over `timeperiod`.
pub fn vortex(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut plus = vec![f64::NAN; n];
    let mut minus = vec![f64::NAN; n];
    if timeperiod < 1 || n < 2 {
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
    let stoch1 = stochastic_of(&macd, cycleperiod);
    let pf = overlap::ema(&stoch1, d1);
    let stoch2 = stochastic_of(&pf, cycleperiod);
    let stc_out = overlap::ema(&stoch2, d2);
    for i in 0..n {
        if stc_out[i].is_finite() {
            result[i] = stc_out[i].clamp(0.0, 100.0);
        }
    }
    result
}

fn stochastic_of(src: &[f64], timeperiod: usize) -> Vec<f64> {
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
    let (jaw, teeth, lips) = super::alligator(
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

/// Coppock Curve: `WMA(ROC(roc1) + ROC(roc2), wma_period)`.
pub fn coppock(close: &[f64], wma_period: usize, roc1: usize, roc2: usize) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_hl(n: usize) -> (Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (1..=n).map(|i| i as f64 + 1.0).collect();
        let low: Vec<f64> = (1..=n).map(|i| i as f64 - 1.0).collect();
        (high, low)
    }

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
}
