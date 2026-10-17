//! Kernel C exports for the rest of `ferro_ta_core::backtest`: signal
//! generators, the close-only engine, Monte Carlo, walk-forward, trade
//! extraction and the multi-asset backtest.

use super::backtest::config_from_settings;
use super::invalid;
use crate::marshal::{check_output, elems, kernel_check_scalar, kernel_in, kernel_out};
use crate::status::Status;
use ferro_ta_core::backtest;

/// Ordered fields of a trade record table (capacity `n_bars` rows).
struct TradeColumns {
    entry_bar: Vec<i64>,
    exit_bar: Vec<i64>,
    direction: Vec<f64>,
    entry_price: Vec<f64>,
    exit_price: Vec<f64>,
    pnl_pct: Vec<f64>,
    duration_bars: Vec<i64>,
    mae: Vec<f64>,
    mfe: Vec<f64>,
}

/// Pad each column to `capacity`: -1 for indices, NaN for values.
fn trade_columns(trades: &[backtest::TradeRecord], capacity: usize) -> TradeColumns {
    let ints = |f: fn(&backtest::TradeRecord) -> i64| {
        let mut v: Vec<i64> = trades.iter().map(f).collect();
        v.resize(capacity, -1);
        v
    };
    let floats = |f: fn(&backtest::TradeRecord) -> f64| {
        let mut v: Vec<f64> = trades.iter().map(f).collect();
        v.resize(capacity, f64::NAN);
        v
    };
    TradeColumns {
        entry_bar: ints(|t| t.entry_bar),
        exit_bar: ints(|t| t.exit_bar),
        direction: floats(|t| t.direction),
        entry_price: floats(|t| t.entry_price),
        exit_price: floats(|t| t.exit_price),
        pnl_pct: floats(|t| t.pnl_pct),
        duration_bars: ints(|t| t.duration_bars),
        mae: floats(|t| t.mae),
        mfe: floats(|t| t.mfe),
    }
}

/// Row-major `(n_bars, n_assets)` → per-asset columns (core layout).
fn columns(flat: &[f64], n_bars: usize, n_assets: usize) -> Vec<Vec<f64>> {
    (0..n_assets)
        .map(|j| (0..n_bars).map(|i| flat[i * n_assets + j]).collect())
        .collect()
}

fn flag(v: usize) -> Result<bool, Status> {
    match v {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Status::InvalidParam),
    }
}

crate::ffi_kernel_exports! {
    group: "backtest",
    /// SMA crossover signals: +1 when SMA(fast) > SMA(slow), -1 below, NaN during warm-up. Requires fast < slow.
    ft_sma_crossover_signals(
        close: in f64 [n],
        n: size usize,
        fast: param period,
        slow: param period,
        out: out f64 [n],
    ) => {
        check_output(out, n)?;
        // SAFETY: caller contract — `close` valid for n reads.
        let c = unsafe { kernel_in(close, n)? };
        let s = backtest::sma_crossover_signals(c, fast, slow).map_err(invalid)?;
        // SAFETY: checked non-null; valid for n writes.
        unsafe { kernel_out(out, &s, n) }
    };
    /// MACD crossover signals: +1 when the MACD line is above its signal line, -1 below, NaN during warm-up. Requires fastperiod < slowperiod.
    ft_macd_crossover_signals(
        close: in f64 [n],
        n: size usize,
        fastperiod: param period,
        slowperiod: param period,
        signalperiod: param period,
        out: out f64 [n],
    ) => {
        check_output(out, n)?;
        // SAFETY: caller contract — `close` valid for n reads.
        let c = unsafe { kernel_in(close, n)? };
        let s = backtest::macd_crossover_signals(c, fastperiod, slowperiod, signalperiod)
            .map_err(invalid)?;
        // SAFETY: checked non-null; valid for n writes.
        unsafe { kernel_out(out, &s, n) }
    };
    /// Close-only vectorised backtest: positions are lagged signals (NaN holds the prior position), slippage and commission are charged on position changes. Uses FT_BT_SLIPPAGE_BPS, FT_BT_INITIAL_CAPITAL and the commission entries of `settings` (same vector as ft_backtest_ohlcv); other entries must still be valid but have no effect (short-borrow costs are not modelled by this engine).
    ft_backtest_close(
        close: in f64 [n_bars],
        signals: in f64 [n_bars],
        n_bars: size usize,
        settings: in f64 [n_settings],
        n_settings: size usize,
        out_positions: out f64 [n_bars],
        out_bar_returns: out f64 [n_bars],
        out_strategy_returns: out f64 [n_bars],
        out_equity: out f64 [n_bars],
    ) => {
        for out in [out_positions, out_bar_returns, out_strategy_returns, out_equity] {
            check_output(out, n_bars)?;
        }
        // SAFETY: caller contract — inputs valid for their shapes.
        let (c, s, raw) = unsafe {
            (kernel_in(close, n_bars)?, kernel_in(signals, n_bars)?, kernel_in(settings, n_settings)?)
        };
        let cfg = config_from_settings(raw)?;
        let r = backtest::backtest_core(
            c,
            s,
            cfg.commission.as_ref(),
            cfg.slippage_bps,
            cfg.initial_capital,
            0.0,
        )
        .map_err(invalid)?;
        // SAFETY: outputs checked non-null; valid for n_bars writes each.
        unsafe {
            kernel_out(out_positions, &r.positions, n_bars)?;
            kernel_out(out_bar_returns, &r.bar_returns, n_bars)?;
            kernel_out(out_strategy_returns, &r.strategy_returns, n_bars)?;
            kernel_out(out_equity, &r.equity, n_bars)
        }
    };
    /// Block-bootstrap Monte Carlo of strategy returns: n_sims resampled equity curves, row-major (n_sims, n_bars). block_size 0 or 1 = i.i.d. resampling. Deterministic for a given seed; any 64-bit seed (reinterpreted as unsigned) reproduces the Python/WASM paths.
    ft_monte_carlo_bootstrap(
        strategy_returns: in f64 [n_bars],
        n_bars: size usize,
        n_sims: param period,
        seed: param seed,
        block_size: param count,
        out: out f64 [n_sims, n_bars],
    ) => {
        let count = elems(&[n_sims, n_bars])?;
        check_output(out, count)?;
        // SAFETY: caller contract — input valid for n_bars reads.
        let r = unsafe { kernel_in(strategy_returns, n_bars)? };
        let sims = backtest::monte_carlo_bootstrap(r, n_sims, seed, block_size)
            .map_err(invalid)?;
        let flat: Vec<f64> = sims.into_iter().flatten().collect();
        // SAFETY: checked non-null; valid for count writes.
        unsafe { kernel_out(out, &flat, count) }
    };
    /// Walk-forward fold boundaries, row-major (n_folds, 4) rows of [train_start, train_end, test_start, test_end]. anchored: 0 = rolling window, 1 = expanding from bar 0. step_bars 0 = non-overlapping test windows.
    ft_walk_forward_indices(
        n_bars: param count,
        train_bars: param period,
        test_bars: param period,
        anchored: param count,
        step_bars: param count,
        n_folds: size usize <- ft_walk_forward_n_folds(n_bars, train_bars, test_bars, anchored, step_bars),
        out: out i64 [n_folds, {4}],
    ) => {
        let folds = backtest::walk_forward_indices(n_bars, train_bars, test_bars, flag(anchored)?, step_bars)
            .map_err(invalid)?;
        if folds.len() != n_folds {
            return Err(Status::InvalidParam);
        }
        let count = elems(&[n_folds, 4])?;
        check_output(out, count)?;
        let flat: Vec<i64> = folds.iter().flatten().copied().collect();
        // SAFETY: checked non-null; valid for count writes.
        unsafe { kernel_out(out, &flat, count) }
    };
    /// Round-trip trades from an OHLCV backtest (positions and fill prices from ft_backtest_ohlcv). Writes out_n_trades rows into columns of capacity n_bars; rows past it hold -1 (bar indices) or NaN. mae/mfe are the maximum adverse/favourable excursions as fractions of the entry price.
    ft_extract_trades(
        positions: in f64 [n_bars],
        fill_prices: in f64 [n_bars],
        high: in f64 [n_bars],
        low: in f64 [n_bars],
        n_bars: size usize,
        out_n_trades: out_scalar i64,
        out_entry_bar: out i64 [n_bars],
        out_exit_bar: out i64 [n_bars],
        out_direction: out f64 [n_bars],
        out_entry_price: out f64 [n_bars],
        out_exit_price: out f64 [n_bars],
        out_pnl_pct: out f64 [n_bars],
        out_duration_bars: out i64 [n_bars],
        out_mae: out f64 [n_bars],
        out_mfe: out f64 [n_bars],
    ) => {
        kernel_check_scalar(out_n_trades)?;
        for out in [out_entry_bar, out_exit_bar, out_duration_bars] {
            check_output(out, n_bars)?;
        }
        for out in [out_direction, out_entry_price, out_exit_price, out_pnl_pct, out_mae, out_mfe] {
            check_output(out, n_bars)?;
        }
        // SAFETY: caller contract — inputs valid for n_bars reads.
        let (p, f, h, l) = unsafe {
            (
                kernel_in(positions, n_bars)?,
                kernel_in(fill_prices, n_bars)?,
                kernel_in(high, n_bars)?,
                kernel_in(low, n_bars)?,
            )
        };
        let trades = backtest::extract_trades_ohlcv(p, f, h, l).map_err(invalid)?;
        if trades.len() > n_bars {
            return Err(Status::Internal);
        }
        let t = trade_columns(&trades, n_bars);
        // SAFETY: outputs checked non-null; valid for n_bars writes each.
        unsafe {
            kernel_out(out_entry_bar, &t.entry_bar, n_bars)?;
            kernel_out(out_exit_bar, &t.exit_bar, n_bars)?;
            kernel_out(out_direction, &t.direction, n_bars)?;
            kernel_out(out_entry_price, &t.entry_price, n_bars)?;
            kernel_out(out_exit_price, &t.exit_price, n_bars)?;
            kernel_out(out_pnl_pct, &t.pnl_pct, n_bars)?;
            kernel_out(out_duration_bars, &t.duration_bars, n_bars)?;
            kernel_out(out_mae, &t.mae, n_bars)?;
            kernel_out(out_mfe, &t.mfe, n_bars)?;
            out_n_trades.write(trades.len() as i64);
        }
        Ok(())
    };
    /// Multi-asset backtest: each asset's close-only backtest on its weight column, combined into a portfolio. close and weights are row-major (n_bars, n_assets). Exposure caps (0 = off for gross/net, 1 = off for per-asset) are applied to the weights bar by bar.
    ft_backtest_multi_asset(
        close: in f64 [n_bars, n_assets],
        weights: in f64 [n_bars, n_assets],
        n_bars: size usize,
        n_assets: size usize,
        commission_per_trade: param float,
        slippage_bps: param float,
        max_asset_weight: param float,
        max_gross_exposure: param float,
        max_net_exposure: param float,
        out_asset_returns: out f64 [n_bars, n_assets],
        out_portfolio_returns: out f64 [n_bars],
        out_equity: out f64 [n_bars],
    ) => {
        let count = elems(&[n_bars, n_assets])?;
        check_output(out_asset_returns, count)?;
        check_output(out_portfolio_returns, n_bars)?;
        check_output(out_equity, n_bars)?;
        // SAFETY: caller contract — inputs valid for their shapes.
        let (c, w) = unsafe { (kernel_in(close, count)?, kernel_in(weights, count)?) };
        let r = backtest::backtest_multi_asset_core(
            &columns(c, n_bars, n_assets),
            &columns(w, n_bars, n_assets),
            n_bars,
            n_assets,
            commission_per_trade,
            slippage_bps,
            max_asset_weight,
            max_gross_exposure,
            max_net_exposure,
        )
        .map_err(invalid)?;
        let mut asset = vec![0.0; count];
        for (j, col) in r.asset_returns.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                asset[i * n_assets + j] = v;
            }
        }
        // SAFETY: outputs checked non-null; valid for their shapes.
        unsafe {
            kernel_out(out_asset_returns, &asset, count)?;
            kernel_out(out_portfolio_returns, &r.portfolio_returns, n_bars)?;
            kernel_out(out_equity, &r.portfolio_equity, n_bars)
        }
    };
}
