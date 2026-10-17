//! Kernel C exports for `ferro_ta_core::overfit`.

use super::invalid;
use crate::marshal::{elems, kernel_check_scalar, kernel_in, kernel_out};
use crate::status::Status;
use ferro_ta_core::overfit;

crate::ffi_kernel_exports! {
    group: "overfit",
    /// Probabilistic Sharpe Ratio: probability that the true per-period Sharpe of `returns` exceeds `benchmark_sharpe` (per-period), corrected for sample length, skewness and kurtosis. Needs n_obs >= 3 finite, non-constant returns.
    ft_probabilistic_sharpe_ratio(
        returns: in f64 [n_obs],
        n_obs: size usize,
        benchmark_sharpe: param float,
        out_value: out_scalar f64,
    ) => {
        kernel_check_scalar(out_value)?;
        // SAFETY: caller contract — `returns` valid for `n_obs` reads.
        let r = unsafe { kernel_in(returns, n_obs)? };
        let v = overfit::probabilistic_sharpe_ratio(r, benchmark_sharpe).map_err(invalid)?;
        // SAFETY: checked non-null; valid for one write.
        unsafe { out_value.write(v) };
        Ok(())
    };
    /// Deflated Sharpe Ratio of the selected strategy's `returns`, against the expected best of `n_trials` zero-skill trials whose per-period Sharpe ratios have variance `trial_sharpe_variance`.
    ft_deflated_sharpe_ratio(
        returns: in f64 [n_obs],
        n_obs: size usize,
        n_trials: param period,
        trial_sharpe_variance: param float,
        out_value: out_scalar f64,
    ) => {
        kernel_check_scalar(out_value)?;
        // SAFETY: caller contract — `returns` valid for `n_obs` reads.
        let r = unsafe { kernel_in(returns, n_obs)? };
        let v = overfit::deflated_sharpe_ratio(r, n_trials, trial_sharpe_variance)
            .map_err(invalid)?;
        // SAFETY: checked non-null; valid for one write.
        unsafe { out_value.write(v) };
        Ok(())
    };
    /// Probability of Backtest Overfitting via CSCV over a row-major (n_bars, n_strategies) returns matrix (one column per configuration). Per-combination arrays have ft_pbo_n_combinations(n_splits) entries: IS winner's OOS rank logit, its IS and OOS per-period Sharpe, and its column index.
    ft_pbo_cscv(
        returns: in f64 [n_bars, n_strategies],
        n_bars: size usize,
        n_strategies: size usize,
        n_splits: param period 2,
        n_combinations: size usize <- ft_pbo_n_combinations(n_splits),
        out_pbo: out_scalar f64,
        out_logits: out f64 [n_combinations],
        out_is_sharpe: out f64 [n_combinations],
        out_oos_sharpe: out f64 [n_combinations],
        out_best_index: out i64 [n_combinations],
    ) => {
        if n_combinations != overfit::cscv_combinations(n_splits).map_err(invalid)? {
            return Err(Status::InvalidParam);
        }
        kernel_check_scalar(out_pbo)?;
        crate::marshal::check_output(out_logits, n_combinations)?;
        crate::marshal::check_output(out_is_sharpe, n_combinations)?;
        crate::marshal::check_output(out_oos_sharpe, n_combinations)?;
        crate::marshal::check_output(out_best_index, n_combinations)?;
        // SAFETY: caller contract — `returns` valid for n_bars * n_strategies reads.
        let r = unsafe { kernel_in(returns, elems(&[n_bars, n_strategies])?)? };
        let res = overfit::pbo_cscv(r, n_strategies, n_splits).map_err(invalid)?;
        let best: Vec<i64> = res.best_index.iter().map(|&i| i as i64).collect();
        // SAFETY: outputs checked non-null; valid for n_combinations writes.
        unsafe {
            kernel_out(out_logits, &res.logits, n_combinations)?;
            kernel_out(out_is_sharpe, &res.is_sharpe, n_combinations)?;
            kernel_out(out_oos_sharpe, &res.oos_sharpe, n_combinations)?;
            kernel_out(out_best_index, &best, n_combinations)?;
            out_pbo.write(res.pbo);
        }
        Ok(())
    };
    /// Purged + embargoed k-fold boundaries: row-major (n_folds, 4) rows of [train_left_end, test_start, test_end, train_right_start]; train = [0, train_left_end) U [train_right_start, n_bars).
    ft_purged_kfold_indices(
        n_bars: param count,
        n_folds: param period 2,
        purge_bars: param count,
        embargo_bars: param count,
        out: out i64 [n_folds, {4}],
    ) => {
        let count = elems(&[n_folds, 4])?;
        crate::marshal::check_output(out, count)?;
        let folds = overfit::purged_kfold_indices(n_bars, n_folds, purge_bars, embargo_bars)
            .map_err(invalid)?;
        let flat: Vec<i64> = folds.iter().flatten().copied().collect();
        // SAFETY: checked non-null; valid for n_folds * 4 writes.
        unsafe { kernel_out(out, &flat, count) }
    };
}
