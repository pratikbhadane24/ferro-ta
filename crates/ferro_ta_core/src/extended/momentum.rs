//! Momentum extended indicators (ELDER_RAY, FISHER, CRSI).

use crate::math;
use crate::momentum;
use crate::overlap;
use crate::price_transform;
use crate::simd;

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
    if timeperiod < 1 || n == 0 {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }

    let ema = overlap::ema(close, timeperiod);
    // Warmup bars where the EMA is not finite keep the initialized `NaN` in
    // *both* outputs, exactly as the two `push(f64::NAN)` calls did.
    let mut bull = vec![f64::NAN; n];
    let mut bear = vec![f64::NAN; n];
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
    if timeperiod < 1 || n < timeperiod {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }

    let hl2 = price_transform::medprice(high, low);
    let max_h = math::max(&hl2, timeperiod);
    let min_l = math::min(&hl2, timeperiod);

    let mut fish = vec![f64::NAN; n];
    let mut signal = vec![f64::NAN; n];
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
/// `100 * count(previous <= current) / rankperiod` over the `rankperiod`
/// one-bar ROC values *preceding* the current bar, matching Connors'
/// definition and TradingView's `ta.percentrank`.
///
/// A bar is `NaN` unless all three components are finite, so the output
/// warms up with the slowest of the three.
///
/// # Arguments
/// * `close` — price series.
/// * `timeperiod` — RSI length of close (classic default 3).
/// * `streakperiod` — RSI length of the streak series (classic default 2).
/// * `rankperiod` — percent-rank lookback (classic default 100).
pub fn crsi(close: &[f64], timeperiod: usize, streakperiod: usize, rankperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 || streakperiod < 1 || rankperiod < 1 || n == 0 {
        return vec![f64::NAN; n];
    }

    let rsi_price = momentum::rsi(close, timeperiod);
    let streak = up_down_streak(close);
    let rsi_streak = momentum::rsi(&streak, streakperiod);
    let roc1 = momentum::roc(close, 1);
    let pct_rank = percent_rank(&roc1, rankperiod);

    let mut result = vec![f64::NAN; n];
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

/// Rolling percent rank: `100 * count(previous <= current) / timeperiod`.
///
/// The window is the `timeperiod` values *preceding* index `i`
/// (`src[i - timeperiod .. i]`), exclusive of the current bar, so the first
/// output lands at index `timeperiod` and the result can reach a full 100.
/// This matches TradingView's `ta.percentrank(source, length)`.
///
/// Windows that contain a non-finite value stay `NaN`.
///
/// The `<=` tally is the hot loop (CRSI's default `rankperiod` is 100, so it
/// dominates the whole kernel), so it is delegated to the vectorized
/// [`simd::count_le`]. The "does this window hold a non-finite value?" guard
/// that used to short-circuit that same scan is instead maintained
/// incrementally as a sliding count, keeping it O(1) per bar and leaving the
/// inner loop a pure branchless compare-and-accumulate.
fn percent_rank(src: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = src.len();
    if timeperiod < 1 || n <= timeperiod {
        return vec![f64::NAN; n];
    }
    let scale = 100.0 / timeperiod as f64;
    let mut result = vec![f64::NAN; n];

    // Number of non-finite values in the current window `src[i - P .. i]`.
    let mut holes = src[..timeperiod].iter().filter(|v| !v.is_finite()).count();
    for i in timeperiod..n {
        let current = src[i];
        if holes == 0 && current.is_finite() {
            let count = simd::count_le(&src[(i - timeperiod)..i], current);
            result[i] = count as f64 * scale;
        }
        // Slide the window forward one bar: drop `src[i - P]`, admit `src[i]`.
        holes -= usize::from(!src[i - timeperiod].is_finite());
        holes += usize::from(!current.is_finite());
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
        // ROC(close,1) = [NaN, 10, 100/11, -25/3, 200/11].
        // PercentRank(ROC,2) is NaN at i=2 (its window {ROC[0], ROC[1]}
        // holds a NaN), 0 at i=3 ({10, 9.09} vs -8.33) and 100 at i=4
        // ({9.09, -8.33} vs 18.18). CRSI propagates NaN, so index 2 is NaN.
        assert!(result[0].is_nan() && result[1].is_nan() && result[2].is_nan());
        // RSI(close,2)[3]=50, RSI(streak,2)[3]=25, pctrank=0 → 25
        assert!((result[3] - 25.0).abs() < 1e-10);
        // RSI(close,2)[4]=250/3, RSI(streak,2)[4]=62.5, pctrank=100 → 1475/18
        assert!((result[4] - 1475.0 / 18.0).abs() < 1e-10);
    }

    #[test]
    fn percent_rank_is_exclusive_and_reaches_100() {
        // Previous `timeperiod` values only, `<=` comparison, denominator 2.
        let src = [1.0, 2.0, 3.0, 0.0, 5.0];
        let pr = percent_rank(&src, 2);
        assert!(pr[0].is_nan() && pr[1].is_nan());
        // i=2: {1,2} vs 3 → 2/2
        assert!((pr[2] - 100.0).abs() < 1e-12);
        // i=3: {2,3} vs 0 → 0/2
        assert!((pr[3] - 0.0).abs() < 1e-12);
        // i=4: {3,0} vs 5 → 2/2
        assert!((pr[4] - 100.0).abs() < 1e-12);
        // A tie counts (`<=`).
        let flat = [4.0, 4.0, 4.0];
        assert!((percent_rank(&flat, 2)[2] - 100.0).abs() < 1e-12);
        // Non-finite in the window keeps the bar NaN.
        let holed = [f64::NAN, 1.0, 2.0, 3.0];
        let pr = percent_rank(&holed, 2);
        assert!(pr[2].is_nan());
        assert!((pr[3] - 100.0).abs() < 1e-12);
        // n == timeperiod leaves no room for an exclusive window.
        assert!(percent_rank(&[1.0, 2.0], 2).iter().all(|v| v.is_nan()));
    }

    /// Verbatim copy of the pre-vectorization `percent_rank`: a full scalar
    /// scan of every window with an early break on the first non-finite
    /// value. Kept as the ground truth for the bit-identity test below.
    fn reference_percent_rank(src: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = src.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n <= timeperiod {
            return result;
        }
        let scale = 100.0 / timeperiod as f64;
        for i in timeperiod..n {
            let current = src[i];
            if !current.is_finite() {
                continue;
            }
            let mut count = 0.0;
            let mut ok = true;
            for &v in &src[(i - timeperiod)..i] {
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                if v <= current {
                    count += 1.0;
                }
            }
            if ok {
                result[i] = count * scale;
            }
        }
        result
    }

    /// Outputs are `100 * k / P` for integer `k`, so exact bit equality is
    /// the right bar — an epsilon would hide a semantic change.
    #[test]
    fn percent_rank_is_bit_identical_to_reference() {
        let mut series: Vec<Vec<f64>> = vec![
            vec![],
            vec![1.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0, 3.0, 0.0, 5.0],
            // Plateau: every element ties, so `<=` must count the whole window.
            vec![4.0; 40],
            // Plateau followed by a step down, then back up.
            [vec![7.0; 15], vec![1.0; 15], vec![7.0; 15]].concat(),
            // Monotone up and monotone down (0% and 100% extremes).
            (0..60).map(|i| i as f64).collect(),
            (0..60).map(|i| -(i as f64)).collect(),
            // Non-finite values in assorted positions, including runs.
            vec![
                f64::NAN,
                1.0,
                2.0,
                3.0,
                f64::INFINITY,
                4.0,
                5.0,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::NAN,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
            ],
            // Signed zeros, which compare equal.
            vec![-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
        ];
        // A longer pseudo-random walk with sprinkled holes, to hit the
        // lane-remainder paths at many window offsets.
        let mut x = 0.0f64;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut walk = Vec::with_capacity(500);
        for i in 0..500 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            x += ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
            walk.push(if i % 97 == 0 { f64::NAN } else { x });
        }
        series.push(walk);

        for src in &series {
            for period in [1usize, 2, 3, 7, 8, 9, 16, 17, 100] {
                let got = percent_rank(src, period);
                let want = reference_percent_rank(src, period);
                assert_eq!(got.len(), want.len());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "percent_rank bit mismatch at i={i} (len {}, period {period}): \
                         got {g}, want {w}",
                        src.len()
                    );
                }
            }
        }
    }

    #[test]
    fn up_down_streak_runs() {
        let close = [1.0, 2.0, 3.0, 2.0, 2.0, 1.0];
        let s = up_down_streak(&close);
        assert_eq!(s, vec![0.0, 1.0, 2.0, -1.0, 0.0, -1.0]);
    }
}
