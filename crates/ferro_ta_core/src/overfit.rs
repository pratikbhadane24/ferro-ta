//! Backtest-overfitting statistics.
//!
//! Tools for answering "is this backtest lying?" after a parameter search:
//!
//! - [`probabilistic_sharpe_ratio`] — probability the true Sharpe exceeds a
//!   benchmark, correcting for sample length, skewness and fat tails.
//! - [`deflated_sharpe_ratio`] — PSR against the Sharpe you would expect from
//!   the *best of N* pure-noise trials (selection-bias correction).
//! - [`pbo_cscv`] — Probability of Backtest Overfitting via combinatorially
//!   symmetric cross-validation.
//! - [`purged_kfold_indices`] — k-fold splits with purge/embargo gaps so
//!   overlapping labels cannot leak across the train/test boundary.
//!
//! All Sharpe ratios here are **per-period** (not annualised), matching the
//! formulas in Bailey & López de Prado (2012, 2014) and Bailey et al. (2017).

use crate::options::normal;

const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
/// C(20, 10) = 184 756 combinations — the largest grid we enumerate.
const MAX_CSCV_SPLITS: usize = 20;
const MIN_ROWS_PER_SPLIT: usize = 2;
const MIN_SHARPE_OBS: usize = 3;
const ZERO_VARIANCE_EPS: f64 = 1e-300;
/// Variance below this fraction of E[x²] is cancellation noise from the
/// running-sums formula, i.e. a (near-)constant series.
const RELATIVE_VARIANCE_EPS: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Sharpe moments + PSR / DSR
// ---------------------------------------------------------------------------

/// Per-period Sharpe ratio and the higher moments PSR needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpeMoments {
    /// Mean / sample std (ddof = 1), not annualised.
    pub sharpe: f64,
    /// Population skewness.
    pub skewness: f64,
    /// Population kurtosis (non-excess; 3.0 for a normal distribution).
    pub kurtosis: f64,
    pub n_obs: usize,
}

/// Compute per-period Sharpe, skewness and (non-excess) kurtosis.
pub fn sharpe_moments(returns: &[f64]) -> Result<SharpeMoments, String> {
    let n = returns.len();
    if n < MIN_SHARPE_OBS {
        return Err(format!(
            "returns must have at least {MIN_SHARPE_OBS} elements"
        ));
    }
    if returns.iter().any(|r| !r.is_finite()) {
        return Err("returns must be finite".to_string());
    }
    let nf = n as f64;
    let mean = returns.iter().sum::<f64>() / nf;
    let (m2, m3, m4) = returns.iter().fold((0.0, 0.0, 0.0), |(a, b, c), &r| {
        let d = r - mean;
        let d2 = d * d;
        (a + d2, b + d2 * d, c + d2 * d2)
    });
    let pop_var = m2 / nf;
    if pop_var <= (RELATIVE_VARIANCE_EPS * mean * mean).max(ZERO_VARIANCE_EPS) {
        return Err("returns have zero variance".to_string());
    }
    let sample_std = (m2 / (nf - 1.0)).sqrt();
    Ok(SharpeMoments {
        sharpe: mean / sample_std,
        skewness: (m3 / nf) / pop_var.powf(1.5),
        kurtosis: (m4 / nf) / (pop_var * pop_var),
        n_obs: n,
    })
}

fn psr_from_moments(m: &SharpeMoments, benchmark_sharpe: f64) -> f64 {
    let sr = m.sharpe;
    let denom_sq = 1.0 - m.skewness * sr + (m.kurtosis - 1.0) / 4.0 * sr * sr;
    if denom_sq <= 0.0 {
        return f64::NAN;
    }
    let z = (sr - benchmark_sharpe) * ((m.n_obs - 1) as f64).sqrt() / denom_sq.sqrt();
    normal::cdf(z)
}

/// Probabilistic Sharpe Ratio: P(true Sharpe > `benchmark_sharpe`).
///
/// `benchmark_sharpe` is per-period. Returns NaN when the moment correction
/// term is non-positive (extreme skew/Sharpe combinations).
pub fn probabilistic_sharpe_ratio(returns: &[f64], benchmark_sharpe: f64) -> Result<f64, String> {
    if !benchmark_sharpe.is_finite() {
        return Err("benchmark_sharpe must be finite".to_string());
    }
    Ok(psr_from_moments(
        &sharpe_moments(returns)?,
        benchmark_sharpe,
    ))
}

/// Expected maximum per-period Sharpe among `n_trials` independent zero-skill
/// strategies whose Sharpe estimates have variance `trial_sharpe_variance`.
pub fn expected_max_sharpe(n_trials: usize, trial_sharpe_variance: f64) -> Result<f64, String> {
    if n_trials == 0 {
        return Err("n_trials must be >= 1".to_string());
    }
    if !trial_sharpe_variance.is_finite() || trial_sharpe_variance < 0.0 {
        return Err("trial_sharpe_variance must be finite and >= 0".to_string());
    }
    if n_trials == 1 {
        return Ok(0.0);
    }
    let n = n_trials as f64;
    let z1 = normal::inv_cdf(1.0 - 1.0 / n);
    let z2 = normal::inv_cdf(1.0 - 1.0 / (n * std::f64::consts::E));
    Ok(trial_sharpe_variance.sqrt() * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))
}

/// Deflated Sharpe Ratio: PSR against [`expected_max_sharpe`] of the trials.
///
/// `returns` are the selected (best) strategy's per-period returns,
/// `n_trials` is how many configurations were tried and
/// `trial_sharpe_variance` the variance of their per-period Sharpe ratios.
pub fn deflated_sharpe_ratio(
    returns: &[f64],
    n_trials: usize,
    trial_sharpe_variance: f64,
) -> Result<f64, String> {
    let sr0 = expected_max_sharpe(n_trials, trial_sharpe_variance)?;
    Ok(psr_from_moments(&sharpe_moments(returns)?, sr0))
}

// ---------------------------------------------------------------------------
// PBO via CSCV
// ---------------------------------------------------------------------------

/// Output of [`pbo_cscv`]; vectors have one entry per IS/OOS combination.
#[derive(Clone, Debug)]
pub struct PboResult {
    /// Fraction of combinations whose IS-best strategy ranked at or below
    /// the OOS median (logit <= 0).
    pub pbo: f64,
    /// ln(ω / (1 − ω)) of the IS-best strategy's OOS relative rank ω.
    pub logits: Vec<f64>,
    /// IS Sharpe of the IS-best strategy.
    pub is_sharpe: Vec<f64>,
    /// OOS Sharpe of that same strategy.
    pub oos_sharpe: Vec<f64>,
    /// Column index of the IS-best strategy.
    pub best_index: Vec<usize>,
}

/// Per-block, per-strategy running sums so each combination costs
/// O(n_splits × n_strategies) instead of O(n_bars × n_strategies).
struct BlockSums {
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    count: Vec<usize>,
}

fn block_sums(returns: &[f64], n_bars: usize, n_strategies: usize, n_splits: usize) -> BlockSums {
    let mut sum = vec![0.0; n_splits * n_strategies];
    let mut sum_sq = vec![0.0; n_splits * n_strategies];
    let mut count = vec![0usize; n_splits];
    for b in 0..n_splits {
        let (lo, hi) = (b * n_bars / n_splits, (b + 1) * n_bars / n_splits);
        count[b] = hi - lo;
        let acc = b * n_strategies;
        for row in returns[lo * n_strategies..hi * n_strategies].chunks_exact(n_strategies) {
            for (j, &r) in row.iter().enumerate() {
                sum[acc + j] += r;
                sum_sq[acc + j] += r * r;
            }
        }
    }
    BlockSums { sum, sum_sq, count }
}

fn sharpe_from_sums(sum: f64, sum_sq: f64, n: usize) -> f64 {
    let nf = n as f64;
    let mean = sum / nf;
    let var = (sum_sq - sum * mean) / (nf - 1.0);
    let noise_floor = (RELATIVE_VARIANCE_EPS * sum_sq / nf).max(ZERO_VARIANCE_EPS);
    if var <= noise_floor {
        0.0
    } else {
        mean / var.sqrt()
    }
}

/// Sharpe of every strategy over the blocks selected (or not) by `mask`.
fn masked_sharpes(bs: &BlockSums, mask: u32, want_in_mask: bool, n_strategies: usize) -> Vec<f64> {
    let mut sum = vec![0.0; n_strategies];
    let mut sum_sq = vec![0.0; n_strategies];
    let mut n = 0usize;
    for (b, &cnt) in bs.count.iter().enumerate() {
        if ((mask >> b) & 1 == 1) != want_in_mask {
            continue;
        }
        n += cnt;
        let acc = b * n_strategies;
        for j in 0..n_strategies {
            sum[j] += bs.sum[acc + j];
            sum_sq[j] += bs.sum_sq[acc + j];
        }
    }
    (0..n_strategies)
        .map(|j| sharpe_from_sums(sum[j], sum_sq[j], n))
        .collect()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// Relative rank ω ∈ (0, 1) of `values[idx]`; ties count half.
fn relative_rank(values: &[f64], idx: usize) -> f64 {
    let target = values[idx];
    let (less, equal) = values.iter().enumerate().filter(|&(i, _)| i != idx).fold(
        (0usize, 0usize),
        |(l, e), (_, &v)| {
            if v < target {
                (l + 1, e)
            } else if v == target {
                (l, e + 1)
            } else {
                (l, e)
            }
        },
    );
    (less as f64 + 0.5 * equal as f64 + 1.0) / (values.len() as f64 + 1.0)
}

fn validate_pbo_inputs(len: usize, n_strategies: usize, n_splits: usize) -> Result<usize, String> {
    if n_strategies < 2 {
        return Err("need at least 2 strategies (columns)".to_string());
    }
    if !len.is_multiple_of(n_strategies) {
        return Err(format!(
            "returns length ({len}) is not a multiple of n_strategies ({n_strategies})"
        ));
    }
    cscv_combinations(n_splits)?;
    let n_bars = len / n_strategies;
    if n_bars < n_splits * MIN_ROWS_PER_SPLIT {
        return Err(format!(
            "need at least {} bars for {n_splits} splits, got {n_bars}",
            n_splits * MIN_ROWS_PER_SPLIT
        ));
    }
    Ok(n_bars)
}

/// Number of IS/OOS combinations [`pbo_cscv`] evaluates for `n_splits`:
/// C(n_splits, n_splits / 2). Errors unless `n_splits` is even and in
/// `[2, 20]`.
pub fn cscv_combinations(n_splits: usize) -> Result<usize, String> {
    if n_splits < 2 || !n_splits.is_multiple_of(2) || n_splits > MAX_CSCV_SPLITS {
        return Err(format!(
            "n_splits must be even and in [2, {MAX_CSCV_SPLITS}]"
        ));
    }
    let half = n_splits / 2;
    Ok((0..half).fold(1usize, |acc, k| acc * (n_splits - k) / (k + 1)))
}

/// Probability of Backtest Overfitting via combinatorially symmetric CV.
///
/// `returns` is a row-major `(n_bars, n_strategies)` matrix of per-period
/// strategy returns (one column per parameter configuration). Rows are split
/// into `n_splits` contiguous blocks; every way of choosing half the blocks as
/// in-sample is evaluated (C(n_splits, n_splits/2) combinations).
pub fn pbo_cscv(
    returns: &[f64],
    n_strategies: usize,
    n_splits: usize,
) -> Result<PboResult, String> {
    let n_bars = validate_pbo_inputs(returns.len(), n_strategies, n_splits)?;
    if returns.iter().any(|r| !r.is_finite()) {
        return Err("returns must be finite".to_string());
    }
    let bs = block_sums(returns, n_bars, n_strategies, n_splits);
    let half = (n_splits / 2) as u32;

    let mut result = PboResult {
        pbo: 0.0,
        logits: Vec::new(),
        is_sharpe: Vec::new(),
        oos_sharpe: Vec::new(),
        best_index: Vec::new(),
    };
    for mask in (0u32..(1u32 << n_splits)).filter(|m| m.count_ones() == half) {
        let is = masked_sharpes(&bs, mask, true, n_strategies);
        let oos = masked_sharpes(&bs, mask, false, n_strategies);
        let best = argmax(&is);
        let omega = relative_rank(&oos, best);
        result.logits.push((omega / (1.0 - omega)).ln());
        result.is_sharpe.push(is[best]);
        result.oos_sharpe.push(oos[best]);
        result.best_index.push(best);
    }
    let overfit = result.logits.iter().filter(|&&l| l <= 0.0).count();
    result.pbo = overfit as f64 / result.logits.len() as f64;
    Ok(result)
}

// ---------------------------------------------------------------------------
// Purged k-fold
// ---------------------------------------------------------------------------

/// Purged + embargoed k-fold split boundaries.
///
/// Returns one `[train_left_end, test_start, test_end, train_right_start]`
/// row per fold: train = `[0, train_left_end) ∪ [train_right_start, n_bars)`,
/// test = `[test_start, test_end)`. `purge_bars` are dropped before each test
/// fold (labels that look ahead into it); `embargo_bars` after it (serial
/// correlation leaking back into training).
pub fn purged_kfold_indices(
    n_bars: usize,
    n_folds: usize,
    purge_bars: usize,
    embargo_bars: usize,
) -> Result<Vec<[i64; 4]>, String> {
    if n_folds < 2 {
        return Err("n_folds must be >= 2".to_string());
    }
    if n_folds > n_bars {
        return Err("n_folds must be <= n_bars".to_string());
    }
    Ok((0..n_folds)
        .map(|k| {
            let test_start = k * n_bars / n_folds;
            let test_end = (k + 1) * n_bars / n_folds;
            [
                test_start.saturating_sub(purge_bars) as i64,
                test_start as i64,
                test_end as i64,
                test_end.saturating_add(embargo_bars).min(n_bars) as i64,
            ]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn sharpe_moments_symmetric_series() {
        let m = sharpe_moments(&[-1.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
        // mean 1, sample std sqrt(2.5)
        assert!((m.sharpe - 1.0 / 2.5_f64.sqrt()).abs() < 1e-12);
        assert!(m.skewness.abs() < 1e-12);
        assert!((m.kurtosis - 1.7).abs() < 1e-12);
        assert_eq!(m.n_obs, 5);
    }

    #[test]
    fn sharpe_moments_rejects_bad_input() {
        assert!(sharpe_moments(&[1.0, 2.0]).is_err());
        assert!(sharpe_moments(&[1.0, f64::NAN, 2.0]).is_err());
        assert!(sharpe_moments(&[0.5, 0.5, 0.5]).is_err());
        // 0.1 * 3 is inexact: rounding leaves ~1e-34 variance, still constant.
        assert!(sharpe_moments(&[0.1, 0.1, 0.1]).is_err());
    }

    #[test]
    fn psr_is_half_when_benchmark_equals_sharpe() {
        let r = lcg_noise(500, 7);
        let sr = sharpe_moments(&r).unwrap().sharpe;
        let psr = probabilistic_sharpe_ratio(&r, sr).unwrap();
        assert!((psr - 0.5).abs() < 1e-7);
    }

    #[test]
    fn psr_grows_with_drift() {
        let noise = lcg_noise(1000, 3);
        let drifted: Vec<f64> = noise.iter().map(|x| x + 0.05).collect();
        let p0 = probabilistic_sharpe_ratio(&noise, 0.0).unwrap();
        let p1 = probabilistic_sharpe_ratio(&drifted, 0.0).unwrap();
        assert!(p1 > p0);
        assert!(p1 > 0.99);
    }

    #[test]
    fn expected_max_sharpe_edges_and_monotonic() {
        assert_eq!(expected_max_sharpe(1, 0.5).unwrap(), 0.0);
        assert_eq!(expected_max_sharpe(100, 0.0).unwrap(), 0.0);
        assert!(expected_max_sharpe(0, 1.0).is_err());
        assert!(expected_max_sharpe(10, -1.0).is_err());
        let a = expected_max_sharpe(10, 1.0).unwrap();
        let b = expected_max_sharpe(1000, 1.0).unwrap();
        assert!(b > a && a > 0.0);
        // Reference value: Bailey & López de Prado closed form, N=1000, V=1.
        assert!((b - 3.255_121_5).abs() < 1e-6);
    }

    #[test]
    fn dsr_is_below_psr_for_many_trials() {
        let r: Vec<f64> = lcg_noise(750, 11).iter().map(|x| x + 0.02).collect();
        let psr = probabilistic_sharpe_ratio(&r, 0.0).unwrap();
        let dsr = deflated_sharpe_ratio(&r, 200, 0.004).unwrap();
        assert!(dsr < psr);
        assert_eq!(
            deflated_sharpe_ratio(&r, 1, 0.004).unwrap(),
            psr,
            "one trial means no deflation"
        );
    }

    fn matrix_from_columns(cols: &[Vec<f64>]) -> Vec<f64> {
        let n_bars = cols[0].len();
        (0..n_bars)
            .flat_map(|i| cols.iter().map(move |c| c[i]))
            .collect()
    }

    #[test]
    fn pbo_combination_count_and_bounds() {
        let cols: Vec<Vec<f64>> = (0..8).map(|j| lcg_noise(320, 50 + j)).collect();
        let res = pbo_cscv(&matrix_from_columns(&cols), 8, 8).unwrap();
        assert_eq!(res.logits.len(), 70); // C(8, 4)
        assert!((0.0..=1.0).contains(&res.pbo));
        assert!(res.best_index.iter().all(|&i| i < 8));
    }

    #[test]
    fn pbo_low_when_one_strategy_dominates() {
        let mut cols: Vec<Vec<f64>> = (0..10).map(|j| lcg_noise(400, 900 + j)).collect();
        cols[3] = cols[3].iter().map(|x| x + 0.3).collect();
        let res = pbo_cscv(&matrix_from_columns(&cols), 10, 10).unwrap();
        assert_eq!(res.pbo, 0.0);
        assert!(res.best_index.iter().all(|&i| i == 3));
    }

    #[test]
    fn pbo_near_half_for_pure_noise() {
        let cols: Vec<Vec<f64>> = (0..50).map(|j| lcg_noise(1000, 7_000 + j)).collect();
        let res = pbo_cscv(&matrix_from_columns(&cols), 50, 10).unwrap();
        assert!(res.pbo > 0.2 && res.pbo < 0.8, "pbo = {}", res.pbo);
    }

    #[test]
    fn constant_column_is_not_an_infinite_sharpe_winner() {
        let mut cols: Vec<Vec<f64>> = (0..6).map(|j| lcg_noise(240, 300 + j)).collect();
        cols[2] = cols[2].iter().map(|x| x + 0.2).collect();
        cols[5] = vec![0.003; 240]; // cash-like leg: rounding noise, not edge
        let res = pbo_cscv(&matrix_from_columns(&cols), 6, 6).unwrap();
        assert!(res.best_index.iter().all(|&i| i == 2));
        assert!(res.is_sharpe.iter().all(|s| s.abs() < 10.0));
    }

    #[test]
    fn sharpe_from_sums_zero_for_constant_series() {
        for &(c, n) in &[(1e-4, 20usize), (3e-3, 137), (1e-2, 500)] {
            let (sum, sum_sq) = (0..n).fold((0.0, 0.0), |(a, b), _| (a + c, b + c * c));
            assert_eq!(sharpe_from_sums(sum, sum_sq, n), 0.0, "c={c} n={n}");
        }
    }

    #[test]
    fn pbo_rejects_bad_shapes() {
        let m = vec![0.1; 40];
        assert!(pbo_cscv(&m, 1, 4).is_err());
        assert!(pbo_cscv(&m, 3, 4).is_err());
        assert!(pbo_cscv(&m, 2, 3).is_err());
        assert!(pbo_cscv(&m, 2, 22).is_err());
        assert!(pbo_cscv(&m, 2, 12).is_err());
        let mut nan = m.clone();
        nan[5] = f64::NAN;
        assert!(pbo_cscv(&nan, 2, 4).is_err());
    }

    #[test]
    fn cscv_combination_counts() {
        assert_eq!(cscv_combinations(2), Ok(2));
        assert_eq!(cscv_combinations(8), Ok(70));
        assert_eq!(cscv_combinations(16), Ok(12_870));
        assert_eq!(cscv_combinations(20), Ok(184_756));
        assert!(cscv_combinations(3).is_err() && cscv_combinations(22).is_err());
    }

    #[test]
    fn relative_rank_handles_ties() {
        assert_eq!(relative_rank(&[1.0, 1.0, 1.0], 0), 0.5);
        assert_eq!(relative_rank(&[1.0, 2.0, 3.0], 2), 0.75);
        assert_eq!(relative_rank(&[1.0, 2.0, 3.0], 0), 0.25);
    }

    #[test]
    fn purged_kfold_boundaries() {
        let folds = purged_kfold_indices(100, 4, 3, 2).unwrap();
        assert_eq!(folds.len(), 4);
        assert_eq!(folds[0], [0, 0, 25, 27]);
        assert_eq!(folds[1], [22, 25, 50, 52]);
        assert_eq!(folds[3], [72, 75, 100, 100]);
        assert!(purged_kfold_indices(10, 1, 0, 0).is_err());
        assert!(purged_kfold_indices(3, 4, 0, 0).is_err());
    }
}
