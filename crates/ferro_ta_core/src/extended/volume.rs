//! Volume extended indicators (CMF, EMV, FORCE_INDEX, NVI/PVI, KVO, and related).

use crate::math_ops;
use crate::overlap;
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

/// On-Balance Volume smoothed with [`overlap::ma`].
///
/// `matype` matches `MA` (`0`=SMA … `7`|`8`=T3). Default wrappers use EMA (`1`).
///
/// `0`-`6` and `8` match TA-Lib's `TA_MAType`; `7` is T3 here where
/// TA-Lib's `7` is MAMA, and MAMA is not reachable through any `matype`
/// (call [`overlap::mama`] directly). Values above `8` yield all-`NaN`.
pub fn obv_smoothed(close: &[f64], volume: &[f64], timeperiod: usize, matype: u8) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 || n == 0 || volume.len() != n {
        return vec![f64::NAN; n];
    }
    let raw = volume::obv(close, volume);
    overlap::ma(&raw, timeperiod, matype)
}

/// Chaikin Money Flow: rolling sum of CLV × volume over rolling volume.
///
/// `CLV = ((close - low) - (high - close)) / (high - low)`.
/// First valid value is at `timeperiod - 1`.
pub fn cmf(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timeperiod: usize,
) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod || high.len() != n || low.len() != n || volume.len() != n {
        return result;
    }
    let mut mfv = vec![0.0; n];
    for i in 0..n {
        let hl = high[i] - low[i];
        let clv = if hl != 0.0 {
            ((close[i] - low[i]) - (high[i] - close[i])) / hl
        } else {
            0.0
        };
        mfv[i] = clv * volume[i];
    }
    let mut mfv_sum: f64 = mfv[..timeperiod].iter().sum();
    let mut vol_sum: f64 = volume[..timeperiod].iter().sum();
    if vol_sum != 0.0 {
        result[timeperiod - 1] = mfv_sum / vol_sum;
    }
    for i in timeperiod..n {
        mfv_sum += mfv[i] - mfv[i - timeperiod];
        vol_sum += volume[i] - volume[i - timeperiod];
        if vol_sum != 0.0 {
            result[i] = mfv_sum / vol_sum;
        }
    }
    result
}

/// Ease of Movement, then SMA.
///
/// `distance = mid - mid[1]`, `emv = distance * (high - low) * scale / volume`,
/// then `SMA(emv, timeperiod)`. `scale` is typically 10_000.
pub fn emv(high: &[f64], low: &[f64], volume: &[f64], timeperiod: usize, scale: f64) -> Vec<f64> {
    let n = high.len();
    let mut raw = vec![f64::NAN; n];
    if n == 0 || low.len() != n || volume.len() != n {
        return raw;
    }
    for i in 1..n {
        let vol = volume[i];
        if vol != 0.0 {
            let distance = (high[i] + low[i] - high[i - 1] - low[i - 1]) * 0.5;
            raw[i] = distance * (high[i] - low[i]) * scale / vol;
        }
    }
    if timeperiod <= 1 {
        return raw;
    }
    sma_nan_aware(&raw, timeperiod)
}

/// Force Index: `(close - close[1]) * volume`, optionally EMA-smoothed.
///
/// `timeperiod <= 1` returns the raw 1-bar force. Otherwise `EMA(raw, timeperiod)`.
pub fn force_index(close: &[f64], volume: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut raw = vec![f64::NAN; n];
    if n == 0 || volume.len() != n {
        return raw;
    }
    for i in 1..n {
        raw[i] = (close[i] - close[i - 1]) * volume[i];
    }
    if timeperiod <= 1 {
        return raw;
    }
    overlap::ema(&raw, timeperiod)
}

/// Negative Volume Index, seeded at 1000.
///
/// Updates only when `volume[i] < volume[i-1]`:
/// `NVI *= close[i] / close[i-1]`.
pub fn nvi(close: &[f64], volume: &[f64]) -> Vec<f64> {
    volume_index(close, volume, false)
}

/// NVI plus an EMA signal of that series.
pub fn nvi_with_ema(close: &[f64], volume: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
    let idx = nvi(close, volume);
    let signal = if timeperiod < 1 {
        vec![f64::NAN; idx.len()]
    } else {
        overlap::ema(&idx, timeperiod)
    };
    (idx, signal)
}

/// Positive Volume Index, seeded at 1000.
///
/// Updates only when `volume[i] > volume[i-1]`.
pub fn pvi(close: &[f64], volume: &[f64]) -> Vec<f64> {
    volume_index(close, volume, true)
}

/// PVI plus a moving-average signal (`matype` matches `MA`, `0`-`8`).
///
/// `0`-`6` and `8` match TA-Lib's `TA_MAType`; `7` is T3 here where
/// TA-Lib's `7` is MAMA, and MAMA is not reachable through any `matype`
/// (call [`overlap::mama`] directly). Values above `8` yield all-`NaN`.
pub fn pvi_with_signal(
    close: &[f64],
    volume: &[f64],
    timeperiod: usize,
    matype: u8,
) -> (Vec<f64>, Vec<f64>) {
    let idx = pvi(close, volume);
    let signal = if timeperiod < 1 {
        vec![f64::NAN; idx.len()]
    } else {
        overlap::ma(&idx, timeperiod, matype)
    };
    (idx, signal)
}

fn volume_index(close: &[f64], volume: &[f64], positive: bool) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 || volume.len() != n {
        return result;
    }
    result[0] = 1000.0;
    let mut value = 1000.0;
    for i in 1..n {
        let take = if positive {
            volume[i] > volume[i - 1]
        } else {
            volume[i] < volume[i - 1]
        };
        if take && close[i - 1] != 0.0 {
            value *= close[i] / close[i - 1];
        }
        result[i] = value;
    }
    result
}

/// Volume oscillator: `100 * (SMA(vol, fast) - SMA(vol, slow)) / SMA(vol, slow)`.
pub fn volosc(volume: &[f64], fastperiod: usize, slowperiod: usize) -> Vec<f64> {
    let n = volume.len();
    let mut result = vec![f64::NAN; n];
    if fastperiod < 1 || slowperiod < 1 || n == 0 {
        return result;
    }
    let fast = overlap::sma(volume, fastperiod);
    let slow = overlap::sma(volume, slowperiod);
    for i in 0..n {
        if fast[i].is_finite() && slow[i].is_finite() && slow[i] != 0.0 {
            result[i] = 100.0 * (fast[i] - slow[i]) / slow[i];
        }
    }
    result
}

/// Volume rate of change: `100 * (volume - volume[timeperiod]) / volume[timeperiod]`.
pub fn vroc(volume: &[f64], timeperiod: usize) -> Vec<f64> {
    crate::momentum::roc(volume, timeperiod)
}

/// Klinger Volume Oscillator and its EMA signal.
///
/// Volume force uses the classic trend / cumulative-range construction, then
/// `KVO = EMA(VF, fast) - EMA(VF, slow)` and `signal = EMA(KVO, signalperiod)`.
pub fn kvo(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan = vec![f64::NAN; n];
    if n == 0
        || high.len() != n
        || low.len() != n
        || volume.len() != n
        || fastperiod < 1
        || slowperiod < 1
        || signalperiod < 1
    {
        return (nan.clone(), nan);
    }
    let mut vf = vec![f64::NAN; n];
    let mut trend = 0.0_f64;
    let mut cm = 0.0_f64;
    let mut prev_dm = 0.0_f64;
    for i in 0..n {
        let dm = high[i] - low[i];
        if i == 0 {
            prev_dm = dm;
            continue;
        }
        let sv = high[i] + low[i] + close[i];
        let sv_prev = high[i - 1] + low[i - 1] + close[i - 1];
        let next_trend = if sv > sv_prev {
            1.0
        } else if sv < sv_prev {
            -1.0
        } else {
            trend
        };
        if next_trend == trend && i > 1 {
            cm += dm;
        } else {
            cm = dm + prev_dm;
        }
        vf[i] = if cm != 0.0 {
            volume[i] * (2.0 * dm / cm - 1.0).abs() * next_trend * 100.0
        } else {
            0.0
        };
        trend = next_trend;
        prev_dm = dm;
    }
    let fast = overlap::ema(&vf, fastperiod);
    let slow = overlap::ema(&vf, slowperiod);
    let mut kvo = vec![f64::NAN; n];
    for i in 0..n {
        if fast[i].is_finite() && slow[i].is_finite() {
            kvo[i] = fast[i] - slow[i];
        }
    }
    let signal = overlap::ema(&kvo, signalperiod);
    (kvo, signal)
}

/// Price Volume Trend: cumulative `volume * (close - close[1]) / close[1]`.
///
/// Bar 0 is seeded at 0.
pub fn pvt(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.0; n];
    if n == 0 || volume.len() != n {
        return result;
    }
    result[0] = 0.0;
    for i in 1..n {
        let prev = close[i - 1];
        let delta = if prev != 0.0 {
            volume[i] * (close[i] - prev) / prev
        } else {
            0.0
        };
        result[i] = result[i - 1] + delta;
    }
    result
}

/// Relative volume: `volume / SMA(volume, timeperiod)`.
pub fn rvol(volume: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = volume.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n == 0 {
        return result;
    }
    let avg = overlap::sma(volume, timeperiod);
    for i in 0..n {
        if avg[i].is_finite() && avg[i] != 0.0 {
            result[i] = volume[i] / avg[i];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obv_smoothed_matches_ema_of_obv() {
        let c = [1.0, 2.0, 3.0, 2.0, 4.0];
        let v = [10.0, 20.0, 30.0, 40.0, 50.0];
        let got = obv_smoothed(&c, &v, 3, 1);
        let exp = overlap::ema(&volume::obv(&c, &v), 3);
        for i in 0..c.len() {
            assert!((got[i].is_nan() && exp[i].is_nan()) || (got[i] - exp[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn cmf_mid_close_is_zero() {
        let h = [10.0, 12.0, 11.0, 13.0];
        let l = [8.0, 9.0, 9.0, 10.0];
        let c = [9.0, 10.5, 10.0, 11.5];
        let v = [100.0, 100.0, 100.0, 100.0];
        let result = cmf(&h, &l, &c, &v, 3);
        assert!(result[0].is_nan() && result[1].is_nan());
        assert!(result[2].abs() < 1e-12);
        assert!(result[3].abs() < 1e-12);
    }

    #[test]
    fn nvi_updates_only_on_lower_volume() {
        let c = [10.0, 11.0, 12.0];
        let v = [100.0, 80.0, 90.0];
        let result = nvi(&c, &v);
        assert!((result[0] - 1000.0).abs() < 1e-12);
        assert!((result[1] - 1100.0).abs() < 1e-12);
        assert!((result[2] - 1100.0).abs() < 1e-12);
    }

    #[test]
    fn pvi_updates_only_on_higher_volume() {
        let c = [10.0, 11.0, 12.0];
        let v = [100.0, 120.0, 90.0];
        let result = pvi(&c, &v);
        assert!((result[0] - 1000.0).abs() < 1e-12);
        assert!((result[1] - 1100.0).abs() < 1e-12);
        assert!((result[2] - 1100.0).abs() < 1e-12);
    }

    #[test]
    fn force_index_raw() {
        let c = [10.0, 12.0, 11.0];
        let v = [100.0, 200.0, 150.0];
        let result = force_index(&c, &v, 1);
        assert!(result[0].is_nan());
        assert!((result[1] - 400.0).abs() < 1e-12);
        assert!((result[2] + 150.0).abs() < 1e-12);
    }

    #[test]
    fn pvt_accumulates_pct_volume() {
        let c = [10.0, 11.0, 10.0];
        let v = [100.0, 200.0, 50.0];
        let result = pvt(&c, &v);
        assert!((result[0] - 0.0).abs() < 1e-12);
        assert!((result[1] - 20.0).abs() < 1e-12);
        assert!((result[2] - (20.0 - 50.0 / 11.0)).abs() < 1e-12);
    }

    #[test]
    fn rvol_constant_volume_is_one() {
        let v = [10.0, 10.0, 10.0, 10.0];
        let result = rvol(&v, 2);
        assert!(result[0].is_nan());
        for &x in &result[1..] {
            assert!((x - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn volosc_equal_periods_is_zero() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = volosc(&v, 3, 3);
        for &x in result.iter().filter(|x| x.is_finite()) {
            assert!(x.abs() < 1e-12);
        }
    }

    #[test]
    fn emv_empty() {
        assert!(emv(&[], &[], &[], 14, 10_000.0).is_empty());
    }

    #[test]
    fn emv_period1_known_and_smoothed_finite() {
        let h = [12.0, 14.0, 15.0, 16.0, 17.0];
        let l = [10.0, 11.0, 12.0, 13.0, 14.0];
        let v = [1000.0, 2000.0, 1500.0, 1800.0, 1600.0];
        let raw = emv(&h, &l, &v, 1, 10_000.0);
        assert!(raw[0].is_nan());
        assert!((raw[1] - 22.5).abs() < 1e-12);
        let smoothed = emv(&h, &l, &v, 3, 10_000.0);
        assert!(smoothed[3].is_finite(), "EMV SMA poisoned by leading NaN");
    }

    #[test]
    fn cmf_length_mismatch_returns_nan() {
        let h = [10.0, 12.0, 11.0, 13.0];
        let l = [8.0, 9.0, 9.0, 10.0];
        let c = [9.0, 10.5, 10.0, 11.5];
        let v = [100.0, 100.0, 100.0, 100.0];
        let short = [1.0, 2.0];
        let long = [1.0, 2.0, 3.0, 4.0, 5.0];
        for other in [short.as_slice(), long.as_slice()] {
            for result in [
                cmf(other, &l, &c, &v, 3),
                cmf(&h, other, &c, &v, 3),
                cmf(&h, &l, &c, other, 3),
            ] {
                assert_eq!(result.len(), c.len());
                assert!(result.iter().all(|x| x.is_nan()));
            }
        }
    }

    const SHORT: [f64; 2] = [1.0, 2.0];
    const LONG: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

    fn all_nan(xs: &[f64]) -> bool {
        xs.iter().all(|x| x.is_nan())
    }

    fn all_zero(xs: &[f64]) -> bool {
        xs.iter().all(|x| *x == 0.0)
    }

    #[test]
    fn obv_smoothed_length_mismatch_returns_nan() {
        let c = [1.0, 2.0, 3.0, 2.0];
        for other in [SHORT.as_slice(), LONG.as_slice()] {
            let result = obv_smoothed(&c, other, 3, 1);
            assert_eq!(result.len(), c.len());
            assert!(all_nan(&result));
        }
    }

    #[test]
    fn emv_length_mismatch_returns_nan() {
        let h = [12.0, 14.0, 15.0, 16.0];
        let l = [10.0, 11.0, 12.0, 13.0];
        let v = [1000.0, 2000.0, 1500.0, 1800.0];
        for other in [SHORT.as_slice(), LONG.as_slice()] {
            for result in [
                emv(&h, other, &v, 3, 10_000.0),
                emv(&h, &l, other, 3, 10_000.0),
            ] {
                assert_eq!(result.len(), h.len());
                assert!(all_nan(&result));
            }
        }
    }

    #[test]
    fn force_index_length_mismatch_returns_nan() {
        let c = [10.0, 12.0, 11.0, 13.0];
        for other in [SHORT.as_slice(), LONG.as_slice()] {
            for result in [force_index(&c, other, 1), force_index(&c, other, 3)] {
                assert_eq!(result.len(), c.len());
                assert!(all_nan(&result));
            }
        }
    }

    #[test]
    fn nvi_length_mismatch_returns_nan() {
        let c = [10.0, 11.0, 12.0, 11.5];
        for other in [SHORT.as_slice(), LONG.as_slice()] {
            let idx = nvi(&c, other);
            assert_eq!(idx.len(), c.len());
            assert!(all_nan(&idx));
            let (idx, signal) = nvi_with_ema(&c, other, 3);
            assert_eq!(idx.len(), c.len());
            assert_eq!(signal.len(), c.len());
            assert!(all_nan(&idx) && all_nan(&signal));
        }
    }

    #[test]
    fn pvi_length_mismatch_returns_nan() {
        let c = [10.0, 11.0, 12.0, 11.5];
        for other in [SHORT.as_slice(), LONG.as_slice()] {
            let idx = pvi(&c, other);
            assert_eq!(idx.len(), c.len());
            assert!(all_nan(&idx));
            let (idx, signal) = pvi_with_signal(&c, other, 3, 1);
            assert_eq!(idx.len(), c.len());
            assert_eq!(signal.len(), c.len());
            assert!(all_nan(&idx) && all_nan(&signal));
        }
    }

    #[test]
    fn kvo_length_mismatch_returns_nan() {
        let h = [12.0, 14.0, 15.0, 16.0];
        let l = [10.0, 11.0, 12.0, 13.0];
        let c = [11.0, 13.0, 14.0, 15.0];
        let v = [1000.0, 2000.0, 1500.0, 1800.0];
        for other in [SHORT.as_slice(), LONG.as_slice()] {
            for (kvo_out, signal) in [
                kvo(other, &l, &c, &v, 2, 3, 2),
                kvo(&h, other, &c, &v, 2, 3, 2),
                kvo(&h, &l, &c, other, 2, 3, 2),
            ] {
                assert_eq!(kvo_out.len(), c.len());
                assert_eq!(signal.len(), c.len());
                assert!(all_nan(&kvo_out) && all_nan(&signal));
            }
        }
    }

    #[test]
    fn pvt_length_mismatch_returns_zeros() {
        let c = [10.0, 11.0, 10.0, 12.0];
        for other in [SHORT.as_slice(), LONG.as_slice()] {
            let result = pvt(&c, other);
            assert_eq!(result.len(), c.len());
            assert!(all_zero(&result));
        }
    }
}
