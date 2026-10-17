//! OHLCV backtest and performance-metric kernels (settings-vector ABI).

use ferro_ta_core::backtest::{
    backtest_ohlcv_volume_core, compute_performance_metrics, BacktestConfig, BacktestMetrics,
};
use ferro_ta_core::commission::CommissionModel;
use ferro_ta_ffi::*;
use std::ptr;

const N_METRICS: usize = BacktestMetrics::N_SCALAR_FIELDS;

fn index_of(spec: &ferro_ta_ffi::spec::EnumSpec, name: &str) -> usize {
    spec.values.iter().find(|(n, _)| *n == name).expect(name).1 as usize
}

/// `FT_BT_<name>` from the C enum spec.
fn bt(name: &str) -> usize {
    index_of(&ferro_ta_ffi::enums::BACKTEST_SETTING, name)
}

/// `FT_METRIC_<name>` from the C enum spec.
fn metric(name: &str) -> usize {
    index_of(&ferro_ta_ffi::enums::SWEEP_METRIC, name)
}

struct Bars {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    signals: Vec<f64>,
}

fn bars(n: usize) -> Bars {
    let close: Vec<f64> = (0..n)
        .map(|i| 100.0 + ((i as f64) * 0.4).sin() * 6.0 + i as f64 * 0.1)
        .collect();
    let open: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                close[0]
            } else {
                close[i - 1] + ((i as f64) * 1.1).cos() * 0.5
            }
        })
        .collect();
    Bars {
        high: (0..n).map(|i| open[i].max(close[i]) + 1.0).collect(),
        low: (0..n).map(|i| open[i].min(close[i]) - 1.0).collect(),
        volume: (0..n).map(|i| 400.0 + (i % 5) as f64 * 150.0).collect(),
        signals: (0..n)
            .map(|i| if (i / 6) % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
        open,
        close,
    }
}

fn defaults() -> Vec<f64> {
    let mut n = 0i64;
    assert_eq!(unsafe { ft_backtest_n_settings(&mut n) }, FT_OK);
    let mut s = vec![0.0; n as usize];
    assert_eq!(
        unsafe { ft_backtest_default_settings(s.len(), s.as_mut_ptr()) },
        FT_OK
    );
    s
}

type Outs = [Vec<f64>; 5];

unsafe fn run(b: &Bars, volume: *const f64, settings: &[f64]) -> (i32, Outs) {
    let n = b.close.len();
    let mut o: Outs = std::array::from_fn(|_| vec![0.0; n]);
    let [p, f, br, sr, eq] = &mut o;
    let st = unsafe {
        ft_backtest_ohlcv(
            b.open.as_ptr(),
            b.high.as_ptr(),
            b.low.as_ptr(),
            b.close.as_ptr(),
            volume,
            b.signals.as_ptr(),
            ptr::null(),
            n,
            settings.as_ptr(),
            settings.len(),
            p.as_mut_ptr(),
            f.as_mut_ptr(),
            br.as_mut_ptr(),
            sr.as_mut_ptr(),
            eq.as_mut_ptr(),
        )
    };
    (st, o)
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

#[test]
fn full_featured_run_matches_core() {
    let b = bars(80);
    let mut s = defaults();
    s[bt("STOP_LOSS_PCT")] = 0.03;
    s[bt("TRAILING_STOP_PCT")] = 0.05;
    s[bt("SLIPPAGE_ATR_MULT")] = 0.1;
    s[bt("SLIPPAGE_ATR_PERIOD")] = 5.0;
    s[bt("MAX_PARTICIPATION")] = 0.2;
    s[bt("RATE_OF_VALUE")] = 0.0005;
    s[bt("FILL_MODE")] = 0.0;
    let (st, [pos, fill, _, sr, eq]) = unsafe { run(&b, b.volume.as_ptr(), &s) };
    assert_eq!(st, FT_OK);

    let cfg = BacktestConfig {
        stop_loss_pct: 0.03,
        trailing_stop_pct: 0.05,
        slippage_atr_mult: 0.1,
        slippage_atr_period: 5,
        max_participation: 0.2,
        commission: Some(CommissionModel {
            rate_of_value: 0.0005,
            ..Default::default()
        }),
        ..Default::default()
    };
    let want = backtest_ohlcv_volume_core(
        &b.open,
        &b.high,
        &b.low,
        &b.close,
        Some(&b.volume),
        &b.signals,
        &cfg,
        None,
    )
    .unwrap();
    assert_eq!(pos, want.positions);
    assert!(same(&fill, &want.fill_prices));
    assert_eq!(sr, want.strategy_returns);
    assert_eq!(eq, want.equity);
    assert!(
        pos.iter().any(|p| p.abs() > 0.0 && p.abs() < 1.0),
        "partial fills exercised"
    );
}

#[test]
fn empty_settings_mean_defaults() {
    let b = bars(30);
    let (st_empty, out_empty) = unsafe { run(&b, ptr::null(), &[]) };
    let (st_def, out_def) = unsafe { run(&b, ptr::null(), &defaults()) };
    assert_eq!((st_empty, st_def), (FT_OK, FT_OK));
    assert_eq!(out_empty[4], out_def[4]);
}

#[test]
fn invalid_settings_are_rejected_before_writing() {
    let b = bars(20);
    let mut s = defaults();
    s[bt("MAX_PARTICIPATION")] = 0.1; // needs volume, none given
    let (st, out) = unsafe { run(&b, ptr::null(), &s) };
    assert_eq!(st, FT_ERR_INVALID_PARAM);
    assert!(out[0].iter().all(|&v| v == 0.0), "outputs untouched");
    let too_long = vec![0.0; defaults().len() + 1];
    assert_eq!(
        unsafe { run(&b, ptr::null(), &too_long) }.0,
        FT_ERR_INVALID_PARAM
    );
    let mut bad = defaults();
    bad[bt("FILL_MODE")] = 3.0;
    assert_eq!(
        unsafe { run(&b, ptr::null(), &bad) }.0,
        FT_ERR_INVALID_PARAM
    );
}

#[test]
fn default_settings_serve_any_prefix_length() {
    // A caller built against an older (shorter) header asks for its own
    // length and must get exactly that prefix, never more.
    let full = defaults();
    let mut short = vec![-1.0; 4];
    assert_eq!(
        unsafe { ft_backtest_default_settings(4, short.as_mut_ptr()) },
        FT_OK
    );
    assert_eq!(short, full[..4]);
    let mut too_long = vec![0.0; full.len() + 1];
    assert_eq!(
        unsafe { ft_backtest_default_settings(too_long.len(), too_long.as_mut_ptr()) },
        FT_ERR_INVALID_PARAM
    );
    assert_eq!(
        unsafe { ft_backtest_default_settings(0, ptr::null_mut()) },
        FT_OK
    );
}

#[test]
fn atr_period_zero_is_fine_when_atr_slippage_is_off() {
    let b = bars(20);
    let mut s = defaults();
    s[bt("SLIPPAGE_ATR_PERIOD")] = 0.0;
    assert_eq!(unsafe { run(&b, ptr::null(), &s) }.0, FT_OK);
    s[bt("SLIPPAGE_ATR_MULT")] = 0.5;
    assert_eq!(unsafe { run(&b, ptr::null(), &s) }.0, FT_ERR_INVALID_PARAM);
}

#[test]
fn performance_metrics_row_matches_core() {
    let b = bars(60);
    let (_, [_, _, _, sr, eq]) = unsafe { run(&b, ptr::null(), &defaults()) };
    let mut row = vec![0.0; N_METRICS];
    let st = unsafe {
        ft_performance_metrics(
            sr.as_ptr(),
            eq.as_ptr(),
            sr.len(),
            252.0,
            0.0,
            N_METRICS,
            row.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_OK);
    let want = compute_performance_metrics(&sr, &eq, 252.0, 0.0, None).unwrap();
    assert_eq!(row[metric("SHARPE")], want.sharpe);
    assert_eq!(row[metric("MAX_DRAWDOWN")], want.max_drawdown);
    let st = unsafe {
        ft_performance_metrics(
            sr.as_ptr(),
            eq.as_ptr(),
            sr.len(),
            252.0,
            0.0,
            N_METRICS + 1,
            row.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_ERR_INVALID_PARAM);
}
