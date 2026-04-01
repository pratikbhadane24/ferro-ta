//! Pure Rust portfolio analytics — no PyO3, no numpy, no ndarray.
//!
//! Functions:
//! - `portfolio_volatility`  — sqrt(w' Σ w)
//! - `beta_full`             — Cov/Var OLS beta
//! - `rolling_beta`          — rolling beta with NaN warmup
//! - `drawdown_series`       — per-bar drawdown + max drawdown
//! - `correlation_matrix`    — pairwise Pearson correlation
//! - `relative_strength`     — cumulative return ratio
//! - `spread`                — A - hedge * B
//! - `ratio`                 — A / B (NaN for zero)
//! - `zscore_series`         — rolling z-score, NaN warmup
//! - `compose_weighted`      — weighted sum per row

// ---------------------------------------------------------------------------
// portfolio_volatility
// ---------------------------------------------------------------------------

/// Compute portfolio volatility: sqrt(w' Σ w).
///
/// `cov_matrix` is an n×n covariance matrix stored as a slice of row-Vecs.
/// `weights` has length n.
///
/// Panics if dimensions are inconsistent.
pub fn portfolio_volatility(cov_matrix: &[Vec<f64>], weights: &[f64]) -> f64 {
    let n = weights.len();
    assert!(
        cov_matrix.len() == n,
        "cov_matrix must have {} rows, got {}",
        n,
        cov_matrix.len()
    );
    let mut variance = 0.0_f64;
    for i in 0..n {
        assert!(
            cov_matrix[i].len() == n,
            "cov_matrix row {} must have length {}, got {}",
            i,
            n,
            cov_matrix[i].len()
        );
        let mut row_sum = 0.0_f64;
        for j in 0..n {
            row_sum += weights[j] * cov_matrix[i][j];
        }
        variance += weights[i] * row_sum;
    }
    variance.max(0.0).sqrt()
}

// ---------------------------------------------------------------------------
// beta_full
// ---------------------------------------------------------------------------

/// Compute the full-sample OLS beta of `asset_returns` vs `benchmark_returns`.
///
/// Beta = Cov(asset, bench) / Var(bench).
///
/// Panics if lengths differ or are < 2, or if benchmark has zero variance.
pub fn beta_full(asset_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let n = asset_returns.len();
    assert!(
        n >= 2 && benchmark_returns.len() == n,
        "asset_returns and benchmark_returns must have equal length >= 2"
    );
    let mean_a: f64 = asset_returns.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = benchmark_returns.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0_f64;
    let mut var_b = 0.0_f64;
    for i in 0..n {
        let da = asset_returns[i] - mean_a;
        let db = benchmark_returns[i] - mean_b;
        cov += da * db;
        var_b += db * db;
    }
    assert!(
        var_b != 0.0,
        "benchmark_returns has zero variance; cannot compute beta"
    );
    cov / var_b
}

// ---------------------------------------------------------------------------
// rolling_beta
// ---------------------------------------------------------------------------

/// Compute rolling beta of `asset` vs `benchmark` over a sliding `window`.
///
/// Returns a Vec of the same length as the inputs. The first `window - 1`
/// entries are NaN (warmup period). `window` must be >= 2.
pub fn rolling_beta(asset: &[f64], benchmark: &[f64], window: usize) -> Vec<f64> {
    assert!(window >= 2, "window must be >= 2");
    let n = asset.len();
    assert!(
        n > 0 && benchmark.len() == n,
        "asset and benchmark must be non-empty and equal length"
    );
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        let start = i + 1 - window;
        let a_win = &asset[start..=i];
        let b_win = &benchmark[start..=i];
        let mean_a: f64 = a_win.iter().sum::<f64>() / window as f64;
        let mean_b: f64 = b_win.iter().sum::<f64>() / window as f64;
        let mut cov = 0.0_f64;
        let mut var_b = 0.0_f64;
        for k in 0..window {
            let da = a_win[k] - mean_a;
            let db = b_win[k] - mean_b;
            cov += da * db;
            var_b += db * db;
        }
        result[i] = if var_b == 0.0 { f64::NAN } else { cov / var_b };
    }
    result
}

// ---------------------------------------------------------------------------
// drawdown_series
// ---------------------------------------------------------------------------

/// Compute the drawdown series and maximum drawdown for an equity/price series.
///
/// Drawdown at bar i = (equity[i] - running_max) / running_max  (always <= 0).
///
/// Returns `(dd_array, max_dd)` where `max_dd` is the most negative drawdown.
///
/// Panics if `equity` is empty.
pub fn drawdown_series(equity: &[f64]) -> (Vec<f64>, f64) {
    let n = equity.len();
    assert!(n > 0, "equity must be non-empty");
    let mut dd = vec![0.0_f64; n];
    let mut peak = equity[0];
    let mut max_dd = 0.0_f64;
    for i in 0..n {
        if equity[i] > peak {
            peak = equity[i];
        }
        let d = if peak == 0.0 {
            0.0
        } else {
            (equity[i] - peak) / peak
        };
        dd[i] = d;
        if d < max_dd {
            max_dd = d;
        }
    }
    (dd, max_dd)
}

// ---------------------------------------------------------------------------
// correlation_matrix
// ---------------------------------------------------------------------------

/// Compute the pairwise Pearson correlation matrix.
///
/// `data` is a slice of column vectors — `data[j]` is the return series for
/// asset j, so `data[j][i]` is the return of asset j at bar i. All columns
/// must have the same length (>= 2).
///
/// Returns an n_assets × n_assets matrix stored as `Vec<Vec<f64>>`.
pub fn correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_assets = data.len();
    assert!(n_assets > 0, "data must contain at least one asset column");
    let n_bars = data[0].len();
    assert!(n_bars >= 2, "data must have at least 2 rows (bars)");
    #[allow(clippy::needless_range_loop)]
    for j in 1..n_assets {
        assert!(
            data[j].len() == n_bars,
            "all columns must have equal length; column 0 has {} but column {} has {}",
            n_bars,
            j,
            data[j].len()
        );
    }

    // Means
    let mut means = vec![0.0_f64; n_assets];
    for j in 0..n_assets {
        means[j] = data[j].iter().sum::<f64>() / n_bars as f64;
    }

    // Standard deviations (population)
    let mut stds = vec![0.0_f64; n_assets];
    for j in 0..n_assets {
        let var: f64 = data[j].iter().map(|&v| (v - means[j]).powi(2)).sum::<f64>() / n_bars as f64;
        stds[j] = var.sqrt();
    }

    // Build correlation matrix (exploit symmetry: compute each pair once)
    let mut result = vec![vec![0.0_f64; n_assets]; n_assets];
    #[allow(clippy::needless_range_loop)]
    for j1 in 0..n_assets {
        result[j1][j1] = 1.0;
        for j2 in (j1 + 1)..n_assets {
            let mut cov = 0.0_f64;
            for i in 0..n_bars {
                cov += (data[j1][i] - means[j1]) * (data[j2][i] - means[j2]);
            }
            cov /= n_bars as f64;
            let denom = stds[j1] * stds[j2];
            let corr = if denom == 0.0 { f64::NAN } else { cov / denom };
            result[j1][j2] = corr;
            result[j2][j1] = corr;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// relative_strength
// ---------------------------------------------------------------------------

/// Compute relative strength of an asset vs a benchmark.
///
/// result[i] = cumprod(1 + asset_returns[0..=i]) / cumprod(1 + benchmark_returns[0..=i])
///
/// Panics if lengths differ or are zero.
pub fn relative_strength(asset_returns: &[f64], benchmark_returns: &[f64]) -> Vec<f64> {
    let n = asset_returns.len();
    assert!(
        n > 0 && benchmark_returns.len() == n,
        "asset_returns and benchmark_returns must be non-empty and equal length"
    );
    let mut result = vec![0.0_f64; n];
    let mut cum_a = 1.0_f64;
    let mut cum_b = 1.0_f64;
    for i in 0..n {
        cum_a *= 1.0 + asset_returns[i];
        cum_b *= 1.0 + benchmark_returns[i];
        result[i] = if cum_b == 0.0 {
            f64::NAN
        } else {
            cum_a / cum_b
        };
    }
    result
}

// ---------------------------------------------------------------------------
// spread
// ---------------------------------------------------------------------------

/// Compute the spread between two series: a - hedge * b.
///
/// Panics if lengths differ or are zero.
pub fn spread(a: &[f64], b: &[f64], hedge: f64) -> Vec<f64> {
    let n = a.len();
    assert!(
        n > 0 && b.len() == n,
        "a and b must be non-empty and equal length"
    );
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x - hedge * y)
        .collect()
}

// ---------------------------------------------------------------------------
// ratio
// ---------------------------------------------------------------------------

/// Compute the ratio between two series: a / b.
///
/// Where b is 0, returns NaN.
///
/// Panics if lengths differ or are zero.
pub fn ratio(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    assert!(
        n > 0 && b.len() == n,
        "a and b must be non-empty and equal length"
    );
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| if y == 0.0 { f64::NAN } else { x / y })
        .collect()
}

// ---------------------------------------------------------------------------
// zscore_series
// ---------------------------------------------------------------------------

/// Compute the rolling Z-score of a 1-D series.
///
/// Z[i] = (x[i] - mean(window)) / std(window)
///
/// The first `window - 1` entries are NaN. `window` must be >= 2.
///
/// Panics if `x` is empty or `window < 2`.
pub fn zscore_series(x: &[f64], window: usize) -> Vec<f64> {
    assert!(window >= 2, "window must be >= 2");
    let n = x.len();
    assert!(n > 0, "x must be non-empty");
    let mut result = vec![f64::NAN; n];
    for i in (window - 1)..n {
        let win = &x[i + 1 - window..=i];
        let mean: f64 = win.iter().sum::<f64>() / window as f64;
        let var: f64 = win.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window as f64;
        let std = var.sqrt();
        result[i] = if std == 0.0 {
            f64::NAN
        } else {
            (x[i] - mean) / std
        };
    }
    result
}

// ---------------------------------------------------------------------------
// compose_weighted
// ---------------------------------------------------------------------------

/// Weighted combination of multiple signal columns.
///
/// `data` is a slice of column vectors — `data[j]` is one signal column.
/// `weights` has one entry per column.
///
/// Returns a Vec of length n_bars where each entry is the weighted sum across
/// columns for that bar.
///
/// Panics if weights length != number of columns, or columns have unequal lengths.
pub fn compose_weighted(data: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let n_sigs = data.len();
    assert!(
        weights.len() == n_sigs,
        "weights length ({}) must equal number of signal columns ({})",
        weights.len(),
        n_sigs
    );
    if n_sigs == 0 {
        return vec![];
    }
    let n_bars = data[0].len();
    #[allow(clippy::needless_range_loop)]
    for j in 1..n_sigs {
        assert!(
            data[j].len() == n_bars,
            "all columns must have equal length"
        );
    }
    let mut result = vec![0.0_f64; n_bars];
    for i in 0..n_bars {
        let mut s = 0.0_f64;
        for j in 0..n_sigs {
            s += data[j][i] * weights[j];
        }
        result[i] = s;
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // -- portfolio_volatility -------------------------------------------------

    #[test]
    fn test_portfolio_volatility_identity_cov() {
        // Identity covariance, equal weights => sqrt(sum(w_i^2))
        let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let w = vec![0.5, 0.5];
        let vol = portfolio_volatility(&cov, &w);
        // w' I w = 0.25 + 0.25 = 0.5, sqrt = 0.7071...
        assert!(approx_eq(vol, (0.5_f64).sqrt()));
    }

    #[test]
    fn test_portfolio_volatility_single_asset() {
        let cov = vec![vec![0.04]];
        let w = vec![1.0];
        assert!(approx_eq(portfolio_volatility(&cov, &w), 0.2));
    }

    #[test]
    fn test_portfolio_volatility_correlated() {
        // Fully correlated: cov = [[0.04, 0.04], [0.04, 0.04]]
        let cov = vec![vec![0.04, 0.04], vec![0.04, 0.04]];
        let w = vec![0.5, 0.5];
        // w' Σ w = 0.04, sqrt = 0.2
        let vol = portfolio_volatility(&cov, &w);
        assert!(approx_eq(vol, 0.2));
    }

    // -- beta_full ------------------------------------------------------------

    #[test]
    fn test_beta_full_same_series() {
        let r = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        assert!(approx_eq(beta_full(&r, &r), 1.0));
    }

    #[test]
    fn test_beta_full_double() {
        let bench = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let asset: Vec<f64> = bench.iter().map(|x| x * 2.0).collect();
        assert!(approx_eq(beta_full(&asset, &bench), 2.0));
    }

    #[test]
    #[should_panic]
    fn test_beta_full_zero_variance() {
        let a = vec![0.01, 0.02];
        let b = vec![0.05, 0.05]; // zero variance
        beta_full(&a, &b);
    }

    // -- rolling_beta ---------------------------------------------------------

    #[test]
    fn test_rolling_beta_warmup_nan() {
        let a = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let b = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let rb = rolling_beta(&a, &b, 3);
        assert_eq!(rb.len(), 5);
        assert!(rb[0].is_nan());
        assert!(rb[1].is_nan());
        // From index 2 onward, beta of identical series = 1.0
        assert!(approx_eq(rb[2], 1.0));
        assert!(approx_eq(rb[3], 1.0));
        assert!(approx_eq(rb[4], 1.0));
    }

    #[test]
    fn test_rolling_beta_double() {
        let bench = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let asset: Vec<f64> = bench.iter().map(|x| x * 3.0).collect();
        let rb = rolling_beta(&asset, &bench, 3);
        for i in 2..5 {
            assert!(approx_eq(rb[i], 3.0));
        }
    }

    // -- drawdown_series ------------------------------------------------------

    #[test]
    fn test_drawdown_series_monotonic_up() {
        let eq = vec![100.0, 110.0, 120.0, 130.0];
        let (dd, max_dd) = drawdown_series(&eq);
        for &d in &dd {
            assert!(approx_eq(d, 0.0));
        }
        assert!(approx_eq(max_dd, 0.0));
    }

    #[test]
    fn test_drawdown_series_with_dip() {
        let eq = vec![100.0, 120.0, 90.0, 110.0];
        let (dd, max_dd) = drawdown_series(&eq);
        assert!(approx_eq(dd[0], 0.0));
        assert!(approx_eq(dd[1], 0.0));
        // dd[2] = (90 - 120) / 120 = -0.25
        assert!(approx_eq(dd[2], -0.25));
        // dd[3] = (110 - 120) / 120 = -1/12
        assert!((dd[3] - (-1.0 / 12.0)).abs() < EPS);
        assert!(approx_eq(max_dd, -0.25));
    }

    // -- correlation_matrix ---------------------------------------------------

    #[test]
    fn test_correlation_matrix_identical() {
        let col = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let data = vec![col.clone(), col.clone()];
        let cm = correlation_matrix(&data);
        assert_eq!(cm.len(), 2);
        assert!(approx_eq(cm[0][0], 1.0));
        assert!(approx_eq(cm[1][1], 1.0));
        assert!(approx_eq(cm[0][1], 1.0));
        assert!(approx_eq(cm[1][0], 1.0));
    }

    #[test]
    fn test_correlation_matrix_negatively_correlated() {
        let col_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let col_b: Vec<f64> = col_a.iter().map(|x| -x).collect();
        let data = vec![col_a, col_b];
        let cm = correlation_matrix(&data);
        assert!(approx_eq(cm[0][1], -1.0));
        assert!(approx_eq(cm[1][0], -1.0));
    }

    #[test]
    fn test_correlation_matrix_single_asset() {
        let data = vec![vec![1.0, 2.0, 3.0]];
        let cm = correlation_matrix(&data);
        assert_eq!(cm.len(), 1);
        assert!(approx_eq(cm[0][0], 1.0));
    }

    // -- relative_strength ----------------------------------------------------

    #[test]
    fn test_relative_strength_equal() {
        let r = vec![0.01, -0.02, 0.03];
        let rs = relative_strength(&r, &r);
        for &v in &rs {
            assert!(approx_eq(v, 1.0));
        }
    }

    #[test]
    fn test_relative_strength_outperformance() {
        let a = vec![0.10, 0.10];
        let b = vec![0.05, 0.05];
        let rs = relative_strength(&a, &b);
        // rs[0] = 1.10 / 1.05
        assert!((rs[0] - 1.10 / 1.05).abs() < EPS);
        // rs[1] = 1.21 / 1.1025
        assert!((rs[1] - 1.21 / 1.1025).abs() < EPS);
    }

    // -- spread ---------------------------------------------------------------

    #[test]
    fn test_spread_basic() {
        let a = vec![10.0, 20.0, 30.0];
        let b = vec![5.0, 10.0, 15.0];
        let s = spread(&a, &b, 2.0);
        assert!(approx_eq(s[0], 0.0));
        assert!(approx_eq(s[1], 0.0));
        assert!(approx_eq(s[2], 0.0));
    }

    #[test]
    fn test_spread_hedge_one() {
        let a = vec![10.0, 20.0];
        let b = vec![3.0, 7.0];
        let s = spread(&a, &b, 1.0);
        assert!(approx_eq(s[0], 7.0));
        assert!(approx_eq(s[1], 13.0));
    }

    // -- ratio ----------------------------------------------------------------

    #[test]
    fn test_ratio_basic() {
        let a = vec![10.0, 20.0, 30.0];
        let b = vec![5.0, 10.0, 15.0];
        let r = ratio(&a, &b);
        assert!(approx_eq(r[0], 2.0));
        assert!(approx_eq(r[1], 2.0));
        assert!(approx_eq(r[2], 2.0));
    }

    #[test]
    fn test_ratio_zero_denominator() {
        let a = vec![10.0, 20.0];
        let b = vec![0.0, 5.0];
        let r = ratio(&a, &b);
        assert!(r[0].is_nan());
        assert!(approx_eq(r[1], 4.0));
    }

    // -- zscore_series --------------------------------------------------------

    #[test]
    fn test_zscore_warmup_nan() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = zscore_series(&x, 3);
        assert!(z[0].is_nan());
        assert!(z[1].is_nan());
        assert!(!z[2].is_nan());
        assert!(!z[3].is_nan());
        assert!(!z[4].is_nan());
    }

    #[test]
    fn test_zscore_constant_window() {
        // All same values in window => std = 0 => NaN
        let x = vec![5.0, 5.0, 5.0, 5.0];
        let z = zscore_series(&x, 3);
        assert!(z[2].is_nan());
        assert!(z[3].is_nan());
    }

    #[test]
    fn test_zscore_known_value() {
        // Window [1, 2, 3]: mean=2, pop_std = sqrt(2/3) ~0.8165
        // z = (3 - 2) / sqrt(2/3) = sqrt(3/2) ~ 1.2247
        let x = vec![1.0, 2.0, 3.0];
        let z = zscore_series(&x, 3);
        let expected = (3.0_f64 / 2.0).sqrt();
        assert!((z[2] - expected).abs() < EPS);
    }

    // -- compose_weighted -----------------------------------------------------

    #[test]
    fn test_compose_weighted_basic() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let weights = vec![0.3, 0.7];
        let cw = compose_weighted(&data, &weights);
        // bar 0: 1*0.3 + 4*0.7 = 3.1
        assert!(approx_eq(cw[0], 3.1));
        // bar 1: 2*0.3 + 5*0.7 = 4.1
        assert!(approx_eq(cw[1], 4.1));
        // bar 2: 3*0.3 + 6*0.7 = 5.1
        assert!(approx_eq(cw[2], 5.1));
    }

    #[test]
    fn test_compose_weighted_single_column() {
        let data = vec![vec![10.0, 20.0]];
        let weights = vec![2.0];
        let cw = compose_weighted(&data, &weights);
        assert!(approx_eq(cw[0], 20.0));
        assert!(approx_eq(cw[1], 40.0));
    }

    #[test]
    fn test_compose_weighted_empty() {
        let data: Vec<Vec<f64>> = vec![];
        let weights: Vec<f64> = vec![];
        let cw = compose_weighted(&data, &weights);
        assert!(cw.is_empty());
    }
}
