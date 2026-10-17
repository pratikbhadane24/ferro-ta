//! Kernel exports: multi-shape arrays, derived sizes, NaN-filled skips.

use ferro_ta_core::backtest::BacktestMetrics;
use ferro_ta_core::overfit;
use ferro_ta_core::sweep::{self, BuiltinStrategy, SweepConfig};
use ferro_ta_ffi::*;
use std::ptr;

const N_METRICS: usize = BacktestMetrics::N_SCALAR_FIELDS;
const SMA_CROSSOVER: i32 = 1;
const SHARPE: usize = 3;

fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02
        })
        .collect()
}

fn prices(n: usize) -> Vec<f64> {
    noise(n, 9)
        .iter()
        .scan(100.0, |p, r| {
            *p *= 1.0 + r;
            Some(*p)
        })
        .collect()
}

#[test]
fn psr_and_dsr_match_core() {
    let r: Vec<f64> = noise(400, 3).iter().map(|x| x + 0.002).collect();
    let mut psr = 0.0;
    let st = unsafe { ft_probabilistic_sharpe_ratio(r.as_ptr(), r.len(), 0.01, &mut psr) };
    assert_eq!(st, FT_OK);
    assert_eq!(psr, overfit::probabilistic_sharpe_ratio(&r, 0.01).unwrap());

    let mut dsr = 0.0;
    let st = unsafe { ft_deflated_sharpe_ratio(r.as_ptr(), r.len(), 50, 0.004, &mut dsr) };
    assert_eq!(st, FT_OK);
    assert_eq!(dsr, overfit::deflated_sharpe_ratio(&r, 50, 0.004).unwrap());
}

#[test]
fn psr_rejects_bad_input_and_null_output() {
    let r = [0.1, 0.1, 0.1];
    let mut v = 7.0;
    let st = unsafe { ft_probabilistic_sharpe_ratio(r.as_ptr(), 3, 0.0, &mut v) };
    assert_eq!(st, FT_ERR_INVALID_PARAM);
    assert_eq!(v, 7.0, "output untouched on error");
    let st = unsafe { ft_probabilistic_sharpe_ratio(r.as_ptr(), 3, 0.0, ptr::null_mut()) };
    assert_eq!(st, FT_ERR_NULL_PTR);
    let st = unsafe { ft_probabilistic_sharpe_ratio(ptr::null(), 3, 0.0, &mut v) };
    assert_eq!(st, FT_ERR_NULL_PTR);
    let st = unsafe { ft_probabilistic_sharpe_ratio(r.as_ptr(), 3, f64::NAN, &mut v) };
    assert_eq!(st, FT_ERR_INVALID_PARAM);
}

#[test]
fn pbo_matches_core_and_validates_derived_size() {
    let (n_bars, n_strat, n_splits) = (240usize, 6usize, 6i64);
    let m: Vec<f64> = (0..n_bars * n_strat)
        .map(|k| noise(1, k as u64)[0])
        .collect();
    let mut n_comb = 0i64;
    assert_eq!(
        unsafe { ft_pbo_n_combinations(n_splits, &mut n_comb) },
        FT_OK
    );
    assert_eq!(n_comb, 20);
    let n = n_comb as usize;
    let (mut pbo, mut logits, mut is, mut oos, mut best) =
        (0.0, vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0i64; n]);
    let st = unsafe {
        ft_pbo_cscv(
            m.as_ptr(),
            n_bars,
            n_strat,
            n_splits,
            n,
            &mut pbo,
            logits.as_mut_ptr(),
            is.as_mut_ptr(),
            oos.as_mut_ptr(),
            best.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_OK);
    let want = overfit::pbo_cscv(&m, n_strat, 6).unwrap();
    assert_eq!(pbo, want.pbo);
    assert_eq!(logits, want.logits);
    assert_eq!(
        best,
        want.best_index
            .iter()
            .map(|&i| i as i64)
            .collect::<Vec<_>>()
    );

    // A wrong n_combinations must be rejected, never over/under-written.
    let st = unsafe {
        ft_pbo_cscv(
            m.as_ptr(),
            n_bars,
            n_strat,
            n_splits,
            n - 1,
            &mut pbo,
            logits.as_mut_ptr(),
            is.as_mut_ptr(),
            oos.as_mut_ptr(),
            best.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_ERR_INVALID_PARAM);
    let mut bad = 0i64;
    assert_eq!(
        unsafe { ft_pbo_n_combinations(3, &mut bad) },
        FT_ERR_INVALID_PARAM
    );
}

#[test]
fn purged_kfold_writes_row_major_boundaries() {
    let mut out = [0i64; 16];
    assert_eq!(
        unsafe { ft_purged_kfold_indices(100, 4, 3, 2, out.as_mut_ptr()) },
        FT_OK
    );
    assert_eq!(&out[..8], &[0, 0, 25, 27, 22, 25, 50, 52]);
    assert_eq!(
        unsafe { ft_purged_kfold_indices(100, 1, 0, 0, out.as_mut_ptr()) },
        FT_ERR_INVALID_PARAM
    );
}

#[test]
fn sweep_builtin_marks_skips_and_matches_core() {
    let close = prices(300);
    // (fast, slow): (5, 20) valid, (30, 20) invalid, (10, 40) valid.
    let combos = [5.0, 20.0, 30.0, 20.0, 10.0, 40.0];
    let n_combos = 3;
    let mut n_params = 0i64;
    assert_eq!(
        unsafe { ft_sweep_strategy_n_params(SMA_CROSSOVER, &mut n_params) },
        FT_OK
    );
    assert_eq!(n_params, 2);
    let mut valid = [9i8; 3];
    let mut returns = vec![0.0; close.len() * n_combos];
    let mut metrics = vec![0.0; n_combos * N_METRICS];
    let st = unsafe {
        ft_sweep_builtin(
            close.as_ptr(),
            close.len(),
            SMA_CROSSOVER,
            combos.as_ptr(),
            n_combos,
            2,
            0.0,
            0.0,
            100_000.0,
            252.0,
            0.0,
            N_METRICS,
            valid.as_mut_ptr(),
            returns.as_mut_ptr(),
            metrics.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_OK);
    assert_eq!(valid, [1, 0, 1]);
    assert!((0..close.len()).all(|i| returns[i * 3 + 1].is_nan()));
    assert!(metrics[N_METRICS..2 * N_METRICS].iter().all(|v| v.is_nan()));

    let cfg = SweepConfig {
        parallel: false,
        ..SweepConfig::default()
    };
    let core = sweep::sweep_builtin(
        &close,
        BuiltinStrategy::SmaCrossover,
        &["fast", "slow"],
        &[10.0, 40.0],
        &cfg,
    )
    .unwrap();
    for i in 0..close.len() {
        assert_eq!(returns[i * 3 + 2], core.output.returns[i]);
    }
    assert_eq!(
        metrics[2 * N_METRICS + SHARPE],
        core.output.metrics[0].sharpe
    );
}

#[test]
fn sweep_builtin_rejects_wrong_param_count_and_bad_enum() {
    let close = prices(100);
    let combos = [5.0, 20.0, 1.0];
    let (mut valid, mut r, mut m) = ([0i8; 1], vec![0.0; 100], vec![0.0; N_METRICS]);
    let call = |strategy: i32,
                n_params: usize,
                n_metric_cols: usize,
                valid: &mut [i8],
                r: &mut [f64],
                m: &mut [f64]| unsafe {
        ft_sweep_builtin(
            close.as_ptr(),
            close.len(),
            strategy,
            combos.as_ptr(),
            1,
            n_params,
            0.0,
            0.0,
            100_000.0,
            252.0,
            0.0,
            n_metric_cols,
            valid.as_mut_ptr(),
            r.as_mut_ptr(),
            m.as_mut_ptr(),
        )
    };
    assert_eq!(
        call(SMA_CROSSOVER, 3, N_METRICS, &mut valid, &mut r, &mut m),
        FT_ERR_INVALID_PARAM
    );
    assert_eq!(
        call(7, 2, N_METRICS, &mut valid, &mut r, &mut m),
        FT_ERR_INVALID_PARAM
    );
    // A caller built against a header with a different metric count must be
    // rejected before anything is written.
    assert_eq!(
        call(SMA_CROSSOVER, 2, N_METRICS - 1, &mut valid, &mut r, &mut m),
        FT_ERR_INVALID_PARAM
    );
    let mut n_metrics = 0i64;
    assert_eq!(unsafe { ft_sweep_n_metrics(&mut n_metrics) }, FT_OK);
    assert_eq!(n_metrics as usize, N_METRICS);
}

#[test]
fn sweep_signals_layout() {
    let close = prices(50);
    let n_combos = 2;
    let signals: Vec<f64> = (0..50)
        .flat_map(|i| [1.0, if i % 4 < 2 { 1.0 } else { -1.0 }])
        .collect();
    let mut returns = vec![0.0; 50 * n_combos];
    let mut metrics = vec![0.0; n_combos * N_METRICS];
    let st = unsafe {
        ft_sweep_signals(
            close.as_ptr(),
            50,
            signals.as_ptr(),
            n_combos,
            0.0,
            0.0,
            100_000.0,
            252.0,
            0.0,
            N_METRICS,
            returns.as_mut_ptr(),
            metrics.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_OK);
    let cfg = SweepConfig {
        parallel: false,
        ..SweepConfig::default()
    };
    let want = sweep::sweep_signals(&close, &signals, n_combos, &cfg).unwrap();
    assert_eq!(returns, want.returns);
    assert_eq!(metrics, want.metrics_matrix());
}

#[test]
fn plateau_rejects_negative_coords() {
    let values = [1.0, 2.0];
    let mut out = [0.0; 2];
    let ok_coords = [0i64, 1];
    assert_eq!(
        unsafe {
            ft_sweep_plateau_scores(
                values.as_ptr(),
                2,
                ok_coords.as_ptr(),
                1,
                1,
                out.as_mut_ptr(),
            )
        },
        FT_OK
    );
    assert_eq!(out, [1.5, 1.5]);
    let bad = [0i64, -1];
    assert_eq!(
        unsafe {
            ft_sweep_plateau_scores(values.as_ptr(), 2, bad.as_ptr(), 1, 1, out.as_mut_ptr())
        },
        FT_ERR_INVALID_PARAM
    );
}

#[test]
fn every_kernel_is_in_the_spec_with_consistent_dims() {
    use ferro_ta_ffi::spec::{Dim, KernelArg};
    let kernels: Vec<_> = all_kernel_specs().collect();
    assert_eq!(kernels.len(), 17);
    for k in kernels {
        let named: Vec<&str> = k
            .args
            .iter()
            .filter_map(|a| match a {
                KernelArg::Size { name, .. } => Some(*name),
                KernelArg::Param(p) => Some(p.name),
                _ => None,
            })
            .collect();
        for a in k.args {
            if let KernelArg::In { dims, .. } | KernelArg::Out { dims, .. } = a {
                for d in *dims {
                    if let Dim::Named(n) = d {
                        assert!(named.contains(n), "{}: dim {n} is not an argument", k.name);
                    }
                }
            }
        }
    }
}
