//! Momentum extended indicators (ELDER_RAY, FISHER, CRSI).

use crate::math;
use crate::momentum;
use crate::overlap;
use crate::price_transform;

/// Elder Ray Index: bull power and bear power versus an EMA of close.
///
/// `bull = high - EMA(close)`, `bear = low - EMA(close)`.
/// Leading `timeperiod - 1` values are `NaN` (EMA warmup).
///
/// # Arguments
/// * `high` / `low` / `close` — equal-length OHLC slices.
/// * `timeperiod` — EMA length (classic default 13).
pub fn elder_ray(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut bull = vec![f64::NAN; n];
    let mut bear = vec![f64::NAN; n];
    if timeperiod < 1 || n == 0 {
        return (bull, bear);
    }

    let ema = overlap::ema(close, timeperiod);
    for i in 0..n {
        if ema[i].is_finite() {
            bull[i] = high[i] - ema[i];
            bear[i] = low[i] - ema[i];
        }
    }
    (bull, bear)
}

/// Ehlers Fisher Transform of median price, plus a 1-bar trigger.
///
/// Median price is normalized to `[-1, 1]` over `timeperiod`, smoothed with
/// Ehlers' 0.33 / 0.67 recurrence, then passed through `0.5 * ln((1+x)/(1-x))`
/// with a 0.5 feedback term. `signal` is the previous Fisher value.
///
/// Leading `timeperiod - 1` Fisher values are `NaN`. The first valid signal
/// bar is one bar later.
///
/// # Arguments
/// * `high` / `low` — equal-length price slices.
/// * `timeperiod` — highest/lowest lookback (classic default 9).
pub fn fisher(high: &[f64], low: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut fish = vec![f64::NAN; n];
    let mut signal = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return (fish, signal);
    }

    let hl2 = price_transform::medprice(high, low);
    let max_h = math::max(&hl2, timeperiod);
    let min_l = math::min(&hl2, timeperiod);

    let mut value = 0.0_f64;
    let mut prev_fish = 0.0_f64;
    let mut have_fish = false;
    for i in 0..n {
        if !max_h[i].is_finite() || !min_l[i].is_finite() {
            continue;
        }
        let range = max_h[i] - min_l[i];
        let norm = if range > 0.0 {
            2.0 * ((hl2[i] - min_l[i]) / range - 0.5)
        } else {
            0.0
        };
        value = (0.33 * norm + 0.67 * value).clamp(-0.999, 0.999);
        let next = 0.5 * ((1.0 + value) / (1.0 - value)).ln() + 0.5 * prev_fish;
        fish[i] = next;
        if have_fish {
            signal[i] = prev_fish;
        }
        prev_fish = next;
        have_fish = true;
    }
    (fish, signal)
}

/// Connors RSI: average of price RSI, streak RSI, and percent rank of ROC.
///
/// `CRSI = (RSI(close, timeperiod) + RSI(streak, streakperiod)
///          + PercentRank(ROC(close, 1), rankperiod)) / 3`
///
/// Streak is the run length of consecutive up (+) or down (−) closes
/// (0 on an unchanged bar). Percent rank is
/// `100 * count(window < current) / rankperiod` over the last `rankperiod`
/// one-bar ROC values.
///
/// # Arguments
/// * `close` — price series.
/// * `timeperiod` — RSI length of close (classic default 3).
/// * `streakperiod` — RSI length of the streak series (classic default 2).
/// * `rankperiod` — percent-rank lookback (classic default 100).
pub fn crsi(close: &[f64], timeperiod: usize, streakperiod: usize, rankperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || streakperiod < 1 || rankperiod < 1 || n == 0 {
        return result;
    }

    let rsi_price = momentum::rsi(close, timeperiod);
    let streak = up_down_streak(close);
    let rsi_streak = momentum::rsi(&streak, streakperiod);
    let roc1 = momentum::roc(close, 1);
    let pct_rank = percent_rank(&roc1, rankperiod);

    for i in 0..n {
        let a = rsi_price[i];
        let b = rsi_streak[i];
        let c = pct_rank[i];
        if a.is_finite() && b.is_finite() && c.is_finite() {
            result[i] = (a + b + c) / 3.0;
        }
    }
    result
}

fn up_down_streak(close: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut streak = vec![0.0; n];
    if n == 0 {
        return streak;
    }
    for i in 1..n {
        let diff = close[i] - close[i - 1];
        streak[i] = if diff > 0.0 {
            if streak[i - 1] > 0.0 {
                streak[i - 1] + 1.0
            } else {
                1.0
            }
        } else if diff < 0.0 {
            if streak[i - 1] < 0.0 {
                streak[i - 1] - 1.0
            } else {
                -1.0
            }
        } else {
            0.0
        };
    }
    streak
}

/// Rolling percent rank: `100 * count(window < current) / timeperiod`.
///
/// Windows that contain a non-finite value stay `NaN`.
fn percent_rank(src: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = src.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    let scale = 100.0 / timeperiod as f64;
    for i in (timeperiod - 1)..n {
        let start = i + 1 - timeperiod;
        let current = src[i];
        if !current.is_finite() {
            continue;
        }
        let mut count = 0.0;
        let mut ok = true;
        for &v in &src[start..=i] {
            if !v.is_finite() {
                ok = false;
                break;
            }
            if v < current {
                count += 1.0;
            }
        }
        if ok {
            result[i] = count * scale;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        (high, low, close)
    }

    #[test]
    fn elder_ray_empty() {
        let (bull, bear) = elder_ray(&[], &[], &[], 13);
        assert!(bull.is_empty() && bear.is_empty());
    }

    #[test]
    fn elder_ray_golden_linear() {
        // cargo: elder_ray_golden_linear — EMA(3) on 1..=10 is i at index i;
        // high = close+1, low = close-1 → bull=2, bear=0 after warmup.
        let (h, l, c) = linear_ohlc(10);
        let (bull, bear) = elder_ray(&h, &l, &c, 3);
        assert_eq!(bull.len(), 10);
        assert!(bull[0].is_nan() && bull[1].is_nan());
        for i in 2..10 {
            assert!((bull[i] - 2.0).abs() < 1e-10, "bull[{i}]={}", bull[i]);
            assert!((bear[i] - 0.0).abs() < 1e-10, "bear[{i}]={}", bear[i]);
        }
    }

    #[test]
    fn fisher_empty_and_short() {
        let (f, s) = fisher(&[], &[], 9);
        assert!(f.is_empty() && s.is_empty());
        let (f, s) = fisher(&[1.0, 2.0], &[0.5, 1.5], 9);
        assert!(f.iter().all(|v| v.is_nan()));
        assert!(s.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn fisher_golden_period3() {
        // cargo: fisher_golden_period3 — Ehlers 0.33/0.67 on hl2=1..=5
        let close = [1.0, 2.0, 3.0, 4.0, 5.0];
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let (fish, signal) = fisher(&high, &low, 3);
        assert!(fish[0].is_nan() && fish[1].is_nan());
        assert!(signal[2].is_nan());
        // value0 = 0.33, fish0 = 0.5 * ln(1.33/0.67)
        let fish0 = 0.5 * (1.33_f64 / 0.67).ln();
        assert!((fish[2] - fish0).abs() < 1e-12);
        let value1 = 0.33 + 0.67 * 0.33;
        let fish1 = 0.5 * ((1.0_f64 + value1) / (1.0 - value1)).ln() + 0.5 * fish0;
        assert!((fish[3] - fish1).abs() < 1e-12);
        assert!((signal[3] - fish[2]).abs() < 1e-12);
    }

    #[test]
    fn crsi_empty() {
        assert!(crsi(&[], 3, 2, 100).is_empty());
    }

    #[test]
    fn crsi_golden_small_periods() {
        // cargo: crsi_golden_small_periods
        let close = [10.0, 11.0, 12.0, 11.0, 13.0];
        let result = crsi(&close, 2, 2, 2);
        assert!(result[0].is_nan() && result[1].is_nan());
        // RSI(close,2)[2]=100, RSI(streak,2)[2]=100, pctrank=0 → 200/3
        assert!((result[2] - 200.0 / 3.0).abs() < 1e-10);
        // RSI(close,2)[3]=50, RSI(streak,2)[3]=25, pctrank=0 → 25
        assert!((result[3] - 25.0).abs() < 1e-10);
        // RSI(close,2)[4]=250/3, RSI(streak,2)[4]=62.5, pctrank=50 → 587.5/9
        assert!((result[4] - 587.5 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn up_down_streak_runs() {
        let close = [1.0, 2.0, 3.0, 2.0, 2.0, 1.0];
        let s = up_down_streak(&close);
        assert_eq!(s, vec![0.0, 1.0, 2.0, -1.0, 0.0, -1.0]);
    }
}
