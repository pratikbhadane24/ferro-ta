//! Signal generators, close-only engine, Monte Carlo, walk-forward, trade
//! extraction, multi-asset and streaming-backtest exports vs the core.

use ferro_ta_core::backtest::{self as core};
use ferro_ta_ffi::*;
use std::ptr;

fn prices(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 100.0 + ((i as f64) * 0.35).sin() * 5.0 + i as f64 * 0.05)
        .collect()
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

#[test]
fn crossover_signals_match_core() {
    let c = prices(120);
    let mut out = vec![0.0; c.len()];
    assert_eq!(
        unsafe { ft_sma_crossover_signals(c.as_ptr(), c.len(), 5, 20, out.as_mut_ptr()) },
        FT_OK
    );
    assert!(same(&out, &core::sma_crossover_signals(&c, 5, 20).unwrap()));
    assert_eq!(
        unsafe { ft_sma_crossover_signals(c.as_ptr(), c.len(), 20, 5, out.as_mut_ptr()) },
        FT_ERR_INVALID_PARAM
    );
    assert_eq!(
        unsafe { ft_macd_crossover_signals(c.as_ptr(), c.len(), 6, 13, 4, out.as_mut_ptr()) },
        FT_OK
    );
    assert!(same(
        &out,
        &core::macd_crossover_signals(&c, 6, 13, 4).unwrap()
    ));
}

#[test]
fn close_only_backtest_matches_core_with_settings_costs() {
    let c = prices(80);
    let sig: Vec<f64> = (0..80)
        .map(|i| if (i / 7) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut n = 0i64;
    assert_eq!(unsafe { ft_backtest_n_settings(&mut n) }, FT_OK);
    let mut s = vec![0.0; n as usize];
    assert_eq!(
        unsafe { ft_backtest_default_settings(s.len(), s.as_mut_ptr()) },
        FT_OK
    );
    let idx = |name: &str| {
        ferro_ta_ffi::enums::BACKTEST_SETTING
            .values
            .iter()
            .find(|(k, _)| *k == name)
            .unwrap()
            .1 as usize
    };
    s[idx("SLIPPAGE_BPS")] = 3.0;
    s[idx("RATE_OF_VALUE")] = 0.001;
    let mut o: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; 80]);
    let [p, br, sr, eq] = &mut o;
    let st = unsafe {
        ft_backtest_close(
            c.as_ptr(),
            sig.as_ptr(),
            80,
            s.as_ptr(),
            s.len(),
            p.as_mut_ptr(),
            br.as_mut_ptr(),
            sr.as_mut_ptr(),
            eq.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_OK);
    let cm = ferro_ta_core::commission::CommissionModel {
        rate_of_value: 0.001,
        ..Default::default()
    };
    let want = core::backtest_core(&c, &sig, Some(&cm), 3.0, 100_000.0, 0.0).unwrap();
    assert_eq!(*eq, want.equity);
    assert_eq!(*p, want.positions);
}

#[test]
fn monte_carlo_matches_core_and_is_seeded() {
    let r: Vec<f64> = (0..50)
        .map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.002)
        .collect();
    let mut out = vec![0.0; 3 * 50];
    let st = unsafe { ft_monte_carlo_bootstrap(r.as_ptr(), 50, 3, 42, 5, out.as_mut_ptr()) };
    assert_eq!(st, FT_OK);
    let want: Vec<f64> = core::monte_carlo_bootstrap(&r, 3, 42, 5)
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(out, want);
}

#[test]
fn walk_forward_fold_count_and_rows() {
    let mut n = 0i64;
    assert_eq!(
        unsafe { ft_walk_forward_n_folds(100, 40, 20, 0, 0, &mut n) },
        FT_OK
    );
    assert_eq!(n, 3);
    let mut out = vec![0i64; 12];
    assert_eq!(
        unsafe { ft_walk_forward_indices(100, 40, 20, 0, 0, 3, out.as_mut_ptr()) },
        FT_OK
    );
    assert_eq!(&out[..8], &[0, 40, 40, 60, 20, 60, 60, 80]);
    // Wrong fold count or a non-boolean `anchored` are rejected.
    assert_eq!(
        unsafe { ft_walk_forward_indices(100, 40, 20, 0, 0, 2, out.as_mut_ptr()) },
        FT_ERR_INVALID_PARAM
    );
    assert_eq!(
        unsafe { ft_walk_forward_n_folds(100, 40, 20, 2, 0, &mut n) },
        FT_ERR_INVALID_PARAM
    );
}

#[test]
fn trade_extraction_pads_to_capacity() {
    let n = 12;
    let pos = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
    let fill: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
    let high: Vec<f64> = fill.iter().map(|p| p + 1.0).collect();
    let low: Vec<f64> = fill.iter().map(|p| p - 1.0).collect();
    let mut n_trades = -7i64;
    let (mut eb, mut xb, mut dur) = (vec![0i64; n], vec![0i64; n], vec![0i64; n]);
    let mut f: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
    let [dir, ep, xp, pnl, mae, mfe] = &mut f;
    let st = unsafe {
        ft_extract_trades(
            pos.as_ptr(),
            fill.as_ptr(),
            high.as_ptr(),
            low.as_ptr(),
            n,
            &mut n_trades,
            eb.as_mut_ptr(),
            xb.as_mut_ptr(),
            dir.as_mut_ptr(),
            ep.as_mut_ptr(),
            xp.as_mut_ptr(),
            pnl.as_mut_ptr(),
            dur.as_mut_ptr(),
            mae.as_mut_ptr(),
            mfe.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_OK);
    let want = core::extract_trades_ohlcv(&pos, &fill, &high, &low).unwrap();
    assert_eq!(n_trades as usize, want.len());
    for (k, t) in want.iter().enumerate() {
        assert_eq!(
            (eb[k], xb[k], dur[k]),
            (t.entry_bar, t.exit_bar, t.duration_bars)
        );
        assert_eq!((dir[k], pnl[k]), (t.direction, t.pnl_pct));
    }
    let k = want.len();
    assert!(eb[k..].iter().all(|&v| v == -1) && pnl[k..].iter().all(|v| v.is_nan()));
}

#[test]
fn multi_asset_matches_core_layout() {
    let (n_bars, n_assets) = (40, 3);
    let close: Vec<f64> = (0..n_bars * n_assets)
        .map(|k| 100.0 + ((k / n_assets) as f64 * (0.2 + (k % n_assets) as f64 * 0.1)).sin() * 3.0)
        .collect();
    let weights: Vec<f64> = (0..n_bars * n_assets)
        .map(|k| [0.5, -0.3, 0.2][k % n_assets])
        .collect();
    let mut asset = vec![0.0; n_bars * n_assets];
    let (mut port, mut eq) = (vec![0.0; n_bars], vec![0.0; n_bars]);
    let st = unsafe {
        ft_backtest_multi_asset(
            close.as_ptr(),
            weights.as_ptr(),
            n_bars,
            n_assets,
            0.0,
            2.0,
            1.0,
            0.0,
            0.0,
            asset.as_mut_ptr(),
            port.as_mut_ptr(),
            eq.as_mut_ptr(),
        )
    };
    assert_eq!(st, FT_OK);
    let cols = |flat: &[f64]| -> Vec<Vec<f64>> {
        (0..n_assets)
            .map(|j| (0..n_bars).map(|i| flat[i * n_assets + j]).collect())
            .collect()
    };
    let want = core::backtest_multi_asset_core(
        &cols(&close),
        &cols(&weights),
        n_bars,
        n_assets,
        0.0,
        2.0,
        1.0,
        0.0,
        0.0,
    )
    .unwrap();
    assert_eq!(port, want.portfolio_returns);
    assert_eq!(eq, want.portfolio_equity);
    assert_eq!(asset[5 * n_assets + 1], want.asset_returns[1][5]);
}

#[test]
fn streaming_backtest_tracks_core() {
    let c = prices(30);
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_backtest_new(0.001, 2.0, &mut h) }, FT_OK);
    let mut want = core::StreamingBacktest::new(0.001, 2.0);
    let (mut p, mut r, mut e, mut tc, mut wr, mut aw, mut al, mut k) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut nt = 0i64;
    for (i, &close) in c.iter().enumerate() {
        let sig = if (i / 5) % 2 == 0 { 1.0 } else { -1.0 };
        let st = unsafe {
            ft_stream_backtest_update(
                h, close, sig, &mut p, &mut r, &mut e, &mut nt, &mut tc, &mut wr, &mut aw, &mut al,
                &mut k,
            )
        };
        assert_eq!(st, FT_OK);
        let bar = want.on_bar(close, sig);
        assert_eq!(
            (p, r, e, nt),
            (
                bar.position,
                bar.bar_return,
                bar.equity,
                bar.n_trades as i64
            )
        );
    }
    let s = want.summary();
    assert_eq!(
        (tc, wr, k),
        (s.total_commission, s.win_rate, s.kelly_fraction)
    );
    assert_eq!(unsafe { ft_stream_backtest_reset(h) }, FT_OK);
    unsafe { ft_stream_backtest_free(h) };
}

#[test]
fn monte_carlo_accepts_any_64_bit_seed() {
    let r: Vec<f64> = (0..20)
        .map(|i| ((i * 3 % 7) as f64 - 3.0) * 0.001)
        .collect();
    for seed in [1_700_000_000i64, -1, i64::MAX] {
        let mut out = vec![0.0; 2 * 20];
        let st = unsafe { ft_monte_carlo_bootstrap(r.as_ptr(), 20, 2, seed, 4, out.as_mut_ptr()) };
        assert_eq!(st, FT_OK, "seed {seed}");
        let want: Vec<f64> = core::monte_carlo_bootstrap(&r, 2, seed as u64, 4)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(
            out, want,
            "seed {seed} must reinterpret as u64 like Python/WASM"
        );
    }
}
