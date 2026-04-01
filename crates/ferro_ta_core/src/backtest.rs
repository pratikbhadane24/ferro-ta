//! Pure Rust backtest engine — no PyO3, no numpy dependency.
//!
//! This module contains all backtest logic as pure functions operating on
//! `&[f64]` slices. The PyO3 binding crate provides thin wrappers that
//! convert NumPy arrays to slices and call into this module.

use crate::commission::CommissionModel;

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Replace NaN → 0, +Inf → f64::MAX, −Inf → −f64::MAX (mirrors numpy nan_to_num defaults).
#[inline]
pub fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            f64::MAX
        } else {
            -f64::MAX
        }
    } else {
        v
    }
}

/// Kelly criterion formula: f = win_rate − (1 − win_rate) × (|avg_loss| / avg_win), clamped to [0, 1].
#[inline]
pub fn kelly_formula(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
    if avg_win <= 0.0 {
        return 0.0;
    }
    let f = win_rate - (1.0 - win_rate) * (avg_loss.abs() / avg_win);
    f.clamp(0.0, 1.0)
}

/// Deterministic LCG (Knuth MMIX).
#[inline]
pub fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005_u64)
        .wrapping_add(1_442_695_040_888_963_407_u64);
    *state
}

#[inline]
pub fn lcg_index(state: &mut u64, n: usize) -> usize {
    ((lcg_next(state) >> 11) as usize) % n
}

/// Compute commission cost as a fraction of `initial_capital` for a single execution.
#[inline]
pub fn commission_fraction(
    cm: &CommissionModel,
    fill_price: f64,
    position_size: f64,
    is_buy: bool,
    initial_capital: f64,
) -> f64 {
    if fill_price <= 0.0 || position_size <= 0.0 || initial_capital <= 0.0 {
        return 0.0;
    }
    let trade_value = position_size * fill_price * initial_capital;
    let num_lots = if cm.lot_size > 0.0 {
        (position_size * initial_capital / (cm.lot_size * fill_price)).ceil()
    } else {
        1.0
    };
    cm.cost_fraction(trade_value, num_lots, is_buy, initial_capital)
}

/// Resolve a CommissionModel from an optional reference or backward-compat scalar.
pub fn resolve_commission_model(
    commission: Option<&CommissionModel>,
    commission_per_trade: f64,
) -> CommissionModel {
    match commission {
        Some(c) => c.clone(),
        None if commission_per_trade > 0.0 => CommissionModel {
            flat_per_order: commission_per_trade,
            ..Default::default()
        },
        None => CommissionModel::default(),
    }
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Configuration for the OHLCV-aware backtester.
#[derive(Clone, Debug)]
pub struct BacktestConfig {
    pub fill_mode: String,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub trailing_stop_pct: f64,
    pub slippage_bps: f64,
    pub initial_capital: f64,
    pub commission_per_trade: f64,
    pub max_hold_bars: usize,
    pub slippage_pct_range: f64,
    pub breakeven_pct: f64,
    pub periods_per_year: f64,
    pub margin_ratio: f64,
    pub margin_call_pct: f64,
    pub daily_loss_limit: f64,
    pub total_loss_limit: f64,
    pub commission: Option<CommissionModel>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            fill_mode: "market_open".to_string(),
            stop_loss_pct: 0.0,
            take_profit_pct: 0.0,
            trailing_stop_pct: 0.0,
            slippage_bps: 0.0,
            initial_capital: 100_000.0,
            commission_per_trade: 0.0,
            max_hold_bars: 0,
            slippage_pct_range: 0.0,
            breakeven_pct: 0.0,
            periods_per_year: 252.0,
            margin_ratio: 0.0,
            margin_call_pct: 0.5,
            daily_loss_limit: 0.0,
            total_loss_limit: 0.0,
            commission: None,
        }
    }
}

/// Result of the OHLCV backtest engine.
#[derive(Clone, Debug)]
pub struct OhlcvBacktestResult {
    pub positions: Vec<f64>,
    pub fill_prices: Vec<f64>,
    pub bar_returns: Vec<f64>,
    pub strategy_returns: Vec<f64>,
    pub equity: Vec<f64>,
}

/// A single completed trade record.
#[derive(Clone, Debug)]
pub struct TradeRecord {
    pub entry_bar: i64,
    pub exit_bar: i64,
    pub direction: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl_pct: f64,
    pub duration_bars: i64,
    pub mae: f64,
    pub mfe: f64,
}

/// Comprehensive performance metrics.
#[derive(Clone, Debug)]
pub struct BacktestMetrics {
    pub total_return: f64,
    pub cagr: f64,
    pub annualized_vol: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub max_drawdown: f64,
    pub avg_drawdown: f64,
    pub max_drawdown_duration_bars: usize,
    pub avg_drawdown_duration_bars: f64,
    pub ulcer_index: f64,
    pub omega_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub r_expectancy: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub tail_ratio: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub best_bar: f64,
    pub worst_bar: f64,
    pub n_trades: usize,
    pub n_position_changes: usize,
    // Optional benchmark metrics
    pub benchmark_total_return: Option<f64>,
    pub benchmark_cagr: Option<f64>,
    pub benchmark_annualized_vol: Option<f64>,
    pub benchmark_sharpe: Option<f64>,
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub tracking_error: Option<f64>,
    pub information_ratio: Option<f64>,
}

/// Mutable state bundle for the OHLCV backtest loop.
pub struct OhlcvState {
    pub current_pos: f64,
    pub entry_price: f64,
    pub trail_high: f64,
    pub trail_low: f64,
    pub breakeven_activated: bool,
    pub breakeven_stop: f64,
    pub bars_in_trade: usize,
    pub margin_entry_price: f64,
    pub initial_margin_required: f64,
}

impl OhlcvState {
    pub fn new() -> Self {
        Self {
            current_pos: 0.0,
            entry_price: f64::NAN,
            trail_high: f64::NAN,
            trail_low: f64::NAN,
            breakeven_activated: false,
            breakeven_stop: f64::NAN,
            bars_in_trade: 0,
            margin_entry_price: f64::NAN,
            initial_margin_required: 0.0,
        }
    }

    #[inline]
    pub fn close_position(&mut self) {
        self.current_pos = 0.0;
        self.entry_price = f64::NAN;
        self.trail_high = f64::NAN;
        self.trail_low = f64::NAN;
        self.breakeven_activated = false;
        self.breakeven_stop = f64::NAN;
        self.bars_in_trade = 0;
        self.margin_entry_price = f64::NAN;
        self.initial_margin_required = 0.0;
    }
}

impl Default for OhlcvState {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateful streaming backtester — feed one bar at a time.
#[derive(Clone, Debug)]
pub struct StreamingBacktest {
    pub commission_per_trade: f64,
    pub slippage_bps: f64,
    pub position: f64,
    pub entry_price: f64,
    pub equity: f64,
    pub prev_close: f64,
    pub total_commission: f64,
    pub n_trades: usize,
    pub sum_wins: f64,
    pub n_wins: usize,
    pub sum_losses: f64,
    pub n_losses: usize,
}

/// Result of a single `StreamingBacktest::on_bar` call.
#[derive(Clone, Debug)]
pub struct StreamingBarResult {
    pub position: f64,
    pub bar_return: f64,
    pub equity: f64,
    pub n_trades: usize,
}

/// Summary statistics from StreamingBacktest.
#[derive(Clone, Debug)]
pub struct StreamingSummary {
    pub equity: f64,
    pub n_trades: usize,
    pub total_commission: f64,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub kelly_fraction: f64,
}

// ---------------------------------------------------------------------------
// Signal generators
// ---------------------------------------------------------------------------

/// RSI threshold strategy: +1 when RSI <= oversold, -1 when RSI >= overbought, 0 otherwise.
pub fn rsi_threshold_signals(
    close: &[f64],
    timeperiod: usize,
    oversold: f64,
    overbought: f64,
) -> Vec<f64> {
    let rsi = crate::momentum::rsi(close, timeperiod);
    rsi.iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else if v <= oversold {
                1.0
            } else if v >= overbought {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// SMA crossover strategy: +1 when fast SMA > slow SMA, -1 otherwise. Warm-up bars are NaN.
///
/// Returns `Err` if `fast >= slow`.
pub fn sma_crossover_signals(close: &[f64], fast: usize, slow: usize) -> Result<Vec<f64>, String> {
    if fast >= slow {
        return Err(format!("fast ({fast}) must be less than slow ({slow})"));
    }
    let sma_fast = crate::overlap::sma(close, fast);
    let sma_slow = crate::overlap::sma(close, slow);
    Ok(sma_fast
        .iter()
        .zip(sma_slow.iter())
        .map(|(&f, &s)| {
            if f.is_nan() || s.is_nan() {
                f64::NAN
            } else if f > s {
                1.0
            } else {
                -1.0
            }
        })
        .collect())
}

/// MACD crossover strategy: +1 when MACD line > signal line, -1 otherwise.
///
/// Returns `Err` if `fastperiod >= slowperiod`.
pub fn macd_crossover_signals(
    close: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> Result<Vec<f64>, String> {
    if fastperiod >= slowperiod {
        return Err(format!(
            "fastperiod ({fastperiod}) must be less than slowperiod ({slowperiod})"
        ));
    }
    let (macd_line, signal_line, _) =
        crate::overlap::macd(close, fastperiod, slowperiod, signalperiod);
    Ok(macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(&m, &s)| {
            if m.is_nan() || s.is_nan() {
                f64::NAN
            } else if m > s {
                1.0
            } else {
                -1.0
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Core backtest (close-only)
// ---------------------------------------------------------------------------

/// Backtest result from the simple close-only engine.
#[derive(Clone, Debug)]
pub struct BacktestCoreResult {
    pub positions: Vec<f64>,
    pub bar_returns: Vec<f64>,
    pub strategy_returns: Vec<f64>,
    pub equity: Vec<f64>,
}

/// Backtest core loop over close prices and strategy signals.
///
/// Uses the full `CommissionModel` if provided, otherwise falls back to
/// `commission_per_trade` as a flat per-order fee.
pub fn backtest_core(
    close: &[f64],
    signals: &[f64],
    commission: Option<&CommissionModel>,
    slippage_bps: f64,
    initial_capital: f64,
    commission_per_trade: f64,
) -> Result<BacktestCoreResult, String> {
    let n = close.len();
    if n != signals.len() {
        return Err(format!(
            "close length ({}) != signals length ({})",
            n,
            signals.len()
        ));
    }

    let mut positions = vec![0.0_f64; n];
    if n > 1 {
        for i in 1..n {
            positions[i] = nan_to_num(signals[i - 1]);
        }
    }

    let mut bar_returns = vec![0.0_f64; n];
    for i in 1..n {
        bar_returns[i] = (close[i] - close[i - 1]) / close[i - 1];
    }

    let mut strategy_returns = vec![0.0_f64; n];
    for i in 0..n {
        strategy_returns[i] = positions[i] * bar_returns[i];
    }

    let mut position_changed = vec![false; n];
    for i in 1..n {
        position_changed[i] = (positions[i] - positions[i - 1]).abs() > 1e-12;
    }

    if slippage_bps > 0.0 {
        let slip = slippage_bps / 10_000.0;
        for i in 0..n {
            if position_changed[i] {
                strategy_returns[i] -= slip;
            }
        }
    }

    let cm = resolve_commission_model(commission, commission_per_trade);

    let mut equity = vec![1.0_f64; n];
    let mut cum = 1.0_f64;
    for i in 0..n {
        cum *= 1.0 + strategy_returns[i];
        if position_changed[i] {
            let prev_pos = if i > 0 { positions[i - 1] } else { 0.0 };
            let cost = commission_fraction(
                &cm,
                if close[i] != 0.0 { close[i] } else { 1.0 },
                (positions[i] - prev_pos).abs(),
                positions[i] > prev_pos,
                initial_capital,
            );
            cum -= cost;
        }
        equity[i] = cum;
    }

    Ok(BacktestCoreResult {
        positions,
        bar_returns,
        strategy_returns,
        equity,
    })
}

/// Core single-asset loop reused by multi-asset backtest.
pub fn single_asset_backtest(
    close: &[f64],
    signals: &[f64],
    commission_per_trade: f64,
    slippage_bps: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let slip = slippage_bps / 10_000.0;

    let mut positions = vec![0.0_f64; n];
    for i in 1..n {
        positions[i] = nan_to_num(signals[i - 1]);
    }

    let mut bar_returns = vec![0.0_f64; n];
    for i in 1..n {
        if close[i - 1] != 0.0 {
            bar_returns[i] = (close[i] - close[i - 1]) / close[i - 1];
        }
    }

    let mut strategy_returns = vec![0.0_f64; n];
    for i in 0..n {
        strategy_returns[i] = positions[i] * bar_returns[i];
    }

    let mut position_changed = vec![false; n];
    for i in 1..n {
        position_changed[i] = (positions[i] - positions[i - 1]).abs() > 1e-12;
    }

    if slip > 0.0 {
        for i in 0..n {
            if position_changed[i] {
                strategy_returns[i] -= slip;
            }
        }
    }

    let mut equity = vec![1.0_f64; n];
    if commission_per_trade <= 0.0 {
        let mut g = 1.0_f64;
        for i in 0..n {
            g *= 1.0 + strategy_returns[i];
            equity[i] = g;
        }
    } else {
        let mut gross = vec![1.0_f64; n];
        let mut g = 1.0_f64;
        for i in 0..n {
            g *= 1.0 + strategy_returns[i];
            gross[i] = g;
        }
        let has_zero = gross.contains(&0.0);
        if has_zero {
            equity[0] = 1.0;
            for i in 1..n {
                equity[i] = equity[i - 1] * (1.0 + strategy_returns[i]);
                if position_changed[i] {
                    equity[i] -= commission_per_trade;
                }
            }
        } else {
            let mut disc = 0.0_f64;
            for i in 0..n {
                if position_changed[i] {
                    disc += commission_per_trade / gross[i];
                }
                equity[i] = gross[i] * (1.0 - disc);
            }
        }
    }

    (positions, strategy_returns, equity)
}

// ---------------------------------------------------------------------------
// OHLCV-aware backtest engine
// ---------------------------------------------------------------------------

/// Full OHLCV backtest with stop loss, take profit, trailing stops, breakeven,
/// margin calls, circuit breakers, limit orders, and short borrow costs.
pub fn backtest_ohlcv_core(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    signals: &[f64],
    config: &BacktestConfig,
    limit_prices: Option<&[f64]>,
) -> Result<OhlcvBacktestResult, String> {
    let n = close.len();
    if n < 2 {
        return Err("arrays must have at least 2 elements".to_string());
    }
    if open.len() != n || high.len() != n || low.len() != n || signals.len() != n {
        return Err(format!(
            "all arrays must have equal length (close={}), got open={}, high={}, low={}, signals={}",
            n,
            open.len(),
            high.len(),
            low.len(),
            signals.len()
        ));
    }

    let use_open_fill = config.fill_mode != "market_close";
    let cm = resolve_commission_model(config.commission.as_ref(), config.commission_per_trade);

    let stop_loss_pct = config.stop_loss_pct;
    let take_profit_pct = config.take_profit_pct;
    let trailing_stop_pct = config.trailing_stop_pct;
    let slippage_bps = config.slippage_bps;
    let initial_capital = config.initial_capital;
    let max_hold_bars = config.max_hold_bars;
    let slippage_pct_range = config.slippage_pct_range;
    let breakeven_pct = config.breakeven_pct;
    let periods_per_year = config.periods_per_year;
    let margin_ratio = config.margin_ratio;
    let margin_call_pct = config.margin_call_pct;
    let daily_loss_limit = config.daily_loss_limit;
    let total_loss_limit = config.total_loss_limit;

    let mut positions = vec![0.0_f64; n];
    let mut fill_prices = vec![f64::NAN; n];
    let mut bar_returns = vec![0.0_f64; n];
    let mut strategy_returns = vec![0.0_f64; n];

    let mut st = OhlcvState::new();
    let default_slip = slippage_bps / 10_000.0;

    let mut circuit_broken: bool = false;
    let mut running_equity: f64 = 1.0;

    for i in 1..n {
        let pos_start = st.current_pos;
        let desired_pos = nan_to_num(signals[i - 1]);

        // --- Margin call check (at bar open) ---
        if margin_ratio > 0.0
            && st.current_pos != 0.0
            && !st.margin_entry_price.is_nan()
            && st.initial_margin_required > 0.0
        {
            let position_pnl =
                st.current_pos * (open[i] - st.margin_entry_price) / st.margin_entry_price;
            let margin_equity = st.initial_margin_required + position_pnl;
            if margin_equity <= margin_call_pct * st.initial_margin_required {
                let mc_fill = open[i];
                let mc_ret = if close[i - 1] != 0.0 {
                    st.current_pos * (mc_fill - close[i - 1]) / close[i - 1]
                } else {
                    0.0
                };
                let comm = commission_fraction(
                    &cm,
                    mc_fill,
                    st.current_pos.abs(),
                    st.current_pos < 0.0,
                    initial_capital,
                );
                strategy_returns[i] = mc_ret - comm;
                fill_prices[i] = mc_fill;
                st.close_position();
                positions[i] = 0.0;
                continue;
            }
        }

        let slip: f64 = if slippage_pct_range > 0.0 && close[i] > 0.0 {
            slippage_pct_range * (high[i] - low[i]) / close[i]
        } else {
            default_slip
        };

        if trailing_stop_pct > 0.0 {
            if st.current_pos > 0.0 && !st.trail_high.is_nan() {
                st.trail_high = st.trail_high.max(high[i]);
            }
            if st.current_pos < 0.0 && !st.trail_low.is_nan() {
                st.trail_low = st.trail_low.min(low[i]);
            }
        }

        let close_ret = if close[i - 1] != 0.0 {
            (close[i] - close[i - 1]) / close[i - 1]
        } else {
            0.0
        };
        bar_returns[i] = close_ret;

        let mut forced_close = false;

        // --- Circuit breaker check ---
        if i > 1 {
            running_equity *= 1.0 + strategy_returns[i - 1];
        }
        if !circuit_broken {
            if daily_loss_limit > 0.0 && i > 1 && strategy_returns[i - 1] < -daily_loss_limit {
                circuit_broken = true;
            }
            if total_loss_limit > 0.0 && running_equity < 1.0 - total_loss_limit {
                circuit_broken = true;
            }
        }
        if circuit_broken && st.current_pos != 0.0 {
            let base_fill = if use_open_fill { open[i] } else { close[i] };
            let is_buy = st.current_pos < 0.0;
            let close_r = if close[i - 1] != 0.0 {
                st.current_pos * (base_fill - close[i - 1]) / close[i - 1]
            } else {
                0.0
            };
            let comm = commission_fraction(
                &cm,
                base_fill,
                st.current_pos.abs(),
                is_buy,
                initial_capital,
            );
            strategy_returns[i] = close_r - comm;
            fill_prices[i] = base_fill;
            st.close_position();
            positions[i] = 0.0;
            forced_close = true;
        }
        if circuit_broken {
            positions[i] = 0.0;
            continue;
        }

        // ---- Intrabar trailing stop check ----
        if trailing_stop_pct > 0.0 && st.current_pos != 0.0 && !st.entry_price.is_nan() {
            if st.current_pos > 0.0 && !st.trail_high.is_nan() {
                let trail_stop = st.trail_high * (1.0 - trailing_stop_pct);
                if low[i] <= trail_stop {
                    let stop_ret = if close[i - 1] != 0.0 {
                        (trail_stop - close[i - 1]) / close[i - 1]
                    } else {
                        -trailing_stop_pct
                    };
                    let comm = commission_fraction(
                        &cm,
                        trail_stop,
                        st.current_pos.abs(),
                        false,
                        initial_capital,
                    );
                    strategy_returns[i] = st.current_pos * stop_ret - slip - comm;
                    fill_prices[i] = trail_stop;
                    st.close_position();
                    positions[i] = 0.0;
                    forced_close = true;
                }
            } else if st.current_pos < 0.0 && !st.trail_low.is_nan() {
                let trail_stop = st.trail_low * (1.0 + trailing_stop_pct);
                if high[i] >= trail_stop {
                    let stop_ret = if close[i - 1] != 0.0 {
                        (trail_stop - close[i - 1]) / close[i - 1]
                    } else {
                        trailing_stop_pct
                    };
                    let comm = commission_fraction(
                        &cm,
                        trail_stop,
                        st.current_pos.abs(),
                        true,
                        initial_capital,
                    );
                    strategy_returns[i] = st.current_pos * stop_ret - slip - comm;
                    fill_prices[i] = trail_stop;
                    st.close_position();
                    positions[i] = 0.0;
                    forced_close = true;
                }
            }
        }

        // ---- Breakeven stop activation ----
        if breakeven_pct > 0.0
            && st.current_pos != 0.0
            && !st.entry_price.is_nan()
            && !st.breakeven_activated
        {
            let condition_met = if st.current_pos > 0.0 {
                high[i] >= st.entry_price * (1.0 + breakeven_pct)
            } else {
                low[i] <= st.entry_price * (1.0 - breakeven_pct)
            };
            if condition_met {
                st.breakeven_activated = true;
                st.breakeven_stop = st.entry_price;
            }
        }

        // ---- Intrabar SL/TP combined bracket check ----
        {
            let has_stop = st.breakeven_activated || stop_loss_pct > 0.0;
            let stop_long = if st.breakeven_activated {
                st.breakeven_stop
            } else {
                st.entry_price * (1.0 - stop_loss_pct)
            };
            let stop_short = if st.breakeven_activated {
                st.breakeven_stop
            } else {
                st.entry_price * (1.0 + stop_loss_pct)
            };
            let has_tp = take_profit_pct > 0.0;
            let tp_long = st.entry_price * (1.0 + take_profit_pct);
            let tp_short = st.entry_price * (1.0 - take_profit_pct);

            if !forced_close && st.current_pos != 0.0 && !st.entry_price.is_nan() {
                let (exit_price, did_exit) = if st.current_pos > 0.0 {
                    let sl_hit = has_stop && low[i] <= stop_long;
                    let tp_hit = has_tp && high[i] >= tp_long;
                    match (sl_hit, tp_hit) {
                        (true, true) => {
                            if (open[i] - stop_long).abs() < (tp_long - open[i]).abs() {
                                (stop_long, true)
                            } else {
                                (tp_long, true)
                            }
                        }
                        (true, false) => (stop_long, true),
                        (false, true) => (tp_long, true),
                        _ => (0.0, false),
                    }
                } else {
                    let sl_hit = has_stop && high[i] >= stop_short;
                    let tp_hit = has_tp && low[i] <= tp_short;
                    match (sl_hit, tp_hit) {
                        (true, true) => {
                            if (stop_short - open[i]).abs() < (open[i] - tp_short).abs() {
                                (stop_short, true)
                            } else {
                                (tp_short, true)
                            }
                        }
                        (true, false) => (stop_short, true),
                        (false, true) => (tp_short, true),
                        _ => (0.0, false),
                    }
                };

                if did_exit {
                    let exit_ret = if close[i - 1] != 0.0 {
                        (exit_price - close[i - 1]) / close[i - 1]
                    } else {
                        0.0
                    };
                    let is_buy = st.current_pos < 0.0;
                    let comm = commission_fraction(
                        &cm,
                        exit_price,
                        st.current_pos.abs(),
                        is_buy,
                        initial_capital,
                    );
                    strategy_returns[i] = st.current_pos * exit_ret - slip - comm;
                    fill_prices[i] = exit_price;
                    st.close_position();
                    positions[i] = 0.0;
                    forced_close = true;
                }
            }
        }

        // ---- Time-based exit check ----
        if !forced_close
            && max_hold_bars > 0
            && st.current_pos != 0.0
            && st.bars_in_trade >= max_hold_bars
        {
            let base_fill = if use_open_fill { open[i] } else { close[i] };
            let is_buy = st.current_pos < 0.0;
            let actual_fill = if is_buy {
                base_fill * (1.0 + slip)
            } else {
                base_fill * (1.0 - slip)
            };
            let exit_ret = if close[i - 1] != 0.0 {
                st.current_pos * (actual_fill - close[i - 1]) / close[i - 1]
            } else {
                0.0
            };
            let comm = commission_fraction(
                &cm,
                actual_fill,
                st.current_pos.abs(),
                is_buy,
                initial_capital,
            );
            strategy_returns[i] = exit_ret - comm;
            fill_prices[i] = actual_fill;
            st.close_position();
            positions[i] = 0.0;
            forced_close = true;
        }

        if !forced_close {
            // ---- Limit order check ----
            let raw_change = (desired_pos - st.current_pos).abs() > 1e-12;
            let (effective_desired_pos, limit_override_price): (f64, Option<f64>) = if raw_change {
                match limit_prices {
                    Some(lp) => {
                        let lp_val = lp[i - 1];
                        if lp_val.is_nan() {
                            (desired_pos, None)
                        } else {
                            let is_buy = desired_pos > st.current_pos;
                            if (is_buy && low[i] <= lp_val) || (!is_buy && high[i] >= lp_val) {
                                (desired_pos, Some(lp_val))
                            } else {
                                (st.current_pos, None)
                            }
                        }
                    }
                    None => (desired_pos, None),
                }
            } else {
                (desired_pos, None)
            };

            let pos_changed = (effective_desired_pos - st.current_pos).abs() > 1e-12;
            let base_fill_raw = if use_open_fill { open[i] } else { close[i] };
            let base_fill = limit_override_price.unwrap_or(base_fill_raw);

            let actual_fill = if effective_desired_pos > st.current_pos {
                base_fill * (1.0 + slip)
            } else if effective_desired_pos < st.current_pos {
                base_fill * (1.0 - slip)
            } else {
                base_fill
            };

            if pos_changed {
                fill_prices[i] = actual_fill;
                if effective_desired_pos != 0.0 {
                    st.entry_price = actual_fill;
                    if trailing_stop_pct > 0.0 {
                        if effective_desired_pos > 0.0 {
                            st.trail_high = actual_fill;
                            st.trail_low = f64::NAN;
                        } else {
                            st.trail_low = actual_fill;
                            st.trail_high = f64::NAN;
                        }
                    }
                } else {
                    st.entry_price = f64::NAN;
                    st.trail_high = f64::NAN;
                    st.trail_low = f64::NAN;
                    st.breakeven_activated = false;
                    st.breakeven_stop = f64::NAN;
                }
                if effective_desired_pos != 0.0
                    && st.current_pos != 0.0
                    && (effective_desired_pos.signum() != st.current_pos.signum())
                {
                    st.breakeven_activated = false;
                    st.breakeven_stop = f64::NAN;
                }
            }

            strategy_returns[i] = if pos_changed && use_open_fill && actual_fill != 0.0 {
                if effective_desired_pos != 0.0 && st.current_pos == 0.0 {
                    let r = effective_desired_pos * (close[i] - actual_fill) / actual_fill;
                    let comm = commission_fraction(
                        &cm,
                        actual_fill,
                        effective_desired_pos.abs(),
                        effective_desired_pos > 0.0,
                        initial_capital,
                    );
                    r - comm
                } else if effective_desired_pos == 0.0 {
                    let r = if close[i - 1] != 0.0 {
                        st.current_pos * (actual_fill - close[i - 1]) / close[i - 1]
                    } else {
                        0.0
                    };
                    let comm = commission_fraction(
                        &cm,
                        actual_fill,
                        st.current_pos.abs(),
                        st.current_pos < 0.0,
                        initial_capital,
                    );
                    r - comm
                } else {
                    let exit_r = if close[i - 1] != 0.0 {
                        st.current_pos * (actual_fill - close[i - 1]) / close[i - 1]
                    } else {
                        0.0
                    };
                    let entry_r = effective_desired_pos * (close[i] - actual_fill) / actual_fill;
                    let exit_comm = commission_fraction(
                        &cm,
                        actual_fill,
                        st.current_pos.abs(),
                        st.current_pos < 0.0,
                        initial_capital,
                    );
                    let entry_comm = commission_fraction(
                        &cm,
                        actual_fill,
                        effective_desired_pos.abs(),
                        effective_desired_pos > 0.0,
                        initial_capital,
                    );
                    exit_r + entry_r - exit_comm - entry_comm
                }
            } else {
                let r = st.current_pos * close_ret;
                if pos_changed {
                    let comm = commission_fraction(
                        &cm,
                        if close[i] != 0.0 { close[i] } else { 1.0 },
                        (effective_desired_pos - st.current_pos).abs(),
                        effective_desired_pos > st.current_pos,
                        initial_capital,
                    );
                    r - comm
                } else {
                    r
                }
            };

            if pos_changed && margin_ratio > 0.0 {
                if effective_desired_pos != 0.0 {
                    if st.current_pos == 0.0
                        || (st.current_pos.signum() != effective_desired_pos.signum())
                    {
                        st.initial_margin_required = effective_desired_pos.abs() * margin_ratio;
                        st.margin_entry_price = actual_fill;
                    }
                } else {
                    st.initial_margin_required = 0.0;
                    st.margin_entry_price = f64::NAN;
                }
            }

            st.current_pos = effective_desired_pos;
            positions[i] = st.current_pos;
        }

        // --- Short borrow cost accrual ---
        if st.current_pos < 0.0 && cm.short_borrow_rate_annual > 0.0 {
            let fill_price_for_borrow = if fill_prices[i].is_finite() && fill_prices[i] > 0.0 {
                fill_prices[i]
            } else {
                close[i]
            };
            let trade_value = st.current_pos.abs() * fill_price_for_borrow * initial_capital;
            let borrow_cost_fraction =
                cm.short_borrow_cost(trade_value, periods_per_year) / initial_capital;
            strategy_returns[i] -= borrow_cost_fraction;
        }

        // Update bars_in_trade counter
        if st.current_pos == 0.0 {
            st.bars_in_trade = 0;
        } else if pos_start == 0.0 || (pos_start.signum() != st.current_pos.signum()) {
            st.bars_in_trade = 1;
        } else {
            st.bars_in_trade += 1;
        }
    }

    // Build equity curve
    let mut equity = vec![1.0_f64; n];
    let mut cum = 1.0_f64;
    for i in 0..n {
        cum *= 1.0 + strategy_returns[i];
        equity[i] = cum;
    }

    Ok(OhlcvBacktestResult {
        positions,
        fill_prices,
        bar_returns,
        strategy_returns,
        equity,
    })
}

// ---------------------------------------------------------------------------
// Performance metrics
// ---------------------------------------------------------------------------

/// Compute all industry-standard performance metrics from strategy returns and equity.
pub fn compute_performance_metrics(
    strategy_returns: &[f64],
    equity: &[f64],
    periods_per_year: f64,
    risk_free_rate: f64,
    benchmark_returns: Option<&[f64]>,
) -> Result<BacktestMetrics, String> {
    let r = strategy_returns;
    let eq = equity;
    let n = r.len();

    if n < 2 {
        return Err("strategy_returns must have at least 2 elements".to_string());
    }
    if eq.len() != n {
        return Err("equity and strategy_returns must have equal length".to_string());
    }

    // --- Pass 1: drawdown / equity stats ---
    let mut peak = eq[0];
    let mut max_dd = 0.0_f64;
    let mut dd_sum = 0.0_f64;
    let mut dd_count = 0_usize;
    let mut ulcer_sum = 0.0_f64;
    let mut current_dd_len = 0_usize;
    let mut max_dd_len = 0_usize;
    let mut dd_len_sum = 0_usize;
    let mut dd_len_count = 0_usize;

    for &eq_val in eq.iter().take(n) {
        if eq_val > peak {
            if current_dd_len > 0 {
                dd_len_sum += current_dd_len;
                dd_len_count += 1;
                current_dd_len = 0;
            }
            peak = eq_val;
        }
        let dd = if peak != 0.0 {
            (eq_val - peak) / peak
        } else {
            0.0
        };
        if dd < 0.0 {
            dd_sum += dd;
            dd_count += 1;
            ulcer_sum += dd * dd;
            current_dd_len += 1;
            if dd < max_dd {
                max_dd = dd;
            }
            if current_dd_len > max_dd_len {
                max_dd_len = current_dd_len;
            }
        }
    }
    if current_dd_len > 0 {
        dd_len_sum += current_dd_len;
        dd_len_count += 1;
    }

    let avg_dd = if dd_count > 0 {
        dd_sum / dd_count as f64
    } else {
        0.0
    };
    let ulcer_index = (ulcer_sum / n as f64).sqrt();
    let avg_dd_duration = if dd_len_count > 0 {
        dd_len_sum as f64 / dd_len_count as f64
    } else {
        0.0
    };

    // --- Pass 2: statistical moments ---
    let rf_per_bar = risk_free_rate / periods_per_year;

    let valid_r: Vec<f64> = r.iter().copied().filter(|v| v.is_finite()).collect();
    let n_valid = valid_r.len();
    if n_valid == 0 {
        return Err("No finite values in strategy_returns".to_string());
    }

    let mean_r: f64 = valid_r.iter().sum::<f64>() / n_valid as f64;
    let variance: f64 = valid_r.iter().map(|&v| (v - mean_r).powi(2)).sum::<f64>() / n_valid as f64;
    let std_r = variance.sqrt();

    let downside_sq_sum: f64 = valid_r
        .iter()
        .filter(|&&v| v < rf_per_bar)
        .map(|&v| (v - rf_per_bar).powi(2))
        .sum();
    let downside_std = (downside_sq_sum / n_valid as f64).sqrt();

    let skewness = if std_r > 0.0 {
        valid_r
            .iter()
            .map(|&v| ((v - mean_r) / std_r).powi(3))
            .sum::<f64>()
            / n_valid as f64
    } else {
        0.0
    };
    let kurtosis = if std_r > 0.0 {
        valid_r
            .iter()
            .map(|&v| ((v - mean_r) / std_r).powi(4))
            .sum::<f64>()
            / n_valid as f64
            - 3.0
    } else {
        0.0
    };

    let total_return = if eq[0] != 0.0 {
        eq[n - 1] / eq[0] - 1.0
    } else {
        0.0
    };
    let cagr = if eq[0] != 0.0 && eq[n - 1] > 0.0 {
        (eq[n - 1] / eq[0]).powf(periods_per_year / n as f64) - 1.0
    } else {
        0.0
    };
    let annual_vol = std_r * periods_per_year.sqrt();
    let sharpe = if annual_vol > 0.0 {
        (cagr - risk_free_rate) / annual_vol
    } else {
        0.0
    };
    let sortino = if downside_std > 0.0 {
        (cagr - risk_free_rate) / (downside_std * periods_per_year.sqrt())
    } else {
        0.0
    };
    let calmar = if max_dd < 0.0 {
        cagr / max_dd.abs()
    } else {
        0.0
    };

    // Win / loss analysis — single pass with running counters
    let mut n_active = 0_usize;
    let mut n_wins = 0_usize;
    let mut n_losses = 0_usize;
    let mut win_sum = 0.0_f64;
    let mut loss_sum = 0.0_f64;
    for &v in &valid_r {
        if v != 0.0 {
            n_active += 1;
            if v > 0.0 {
                n_wins += 1;
                win_sum += v;
            } else {
                n_losses += 1;
                loss_sum += v.abs();
            }
        }
    }
    let win_rate = if n_active > 0 {
        n_wins as f64 / n_active as f64
    } else {
        0.0
    };
    let avg_win = if n_wins > 0 {
        win_sum / n_wins as f64
    } else {
        0.0
    };
    let avg_loss = if n_losses > 0 {
        -(loss_sum / n_losses as f64)
    } else {
        0.0
    };
    let profit_factor = if loss_sum > 0.0 {
        win_sum / loss_sum
    } else {
        f64::INFINITY
    };
    let loss_rate = 1.0 - win_rate;
    let r_expectancy = win_rate * avg_win - loss_rate * avg_loss.abs();

    // Omega ratio
    let omega_numer: f64 = valid_r
        .iter()
        .filter(|&&v| v > rf_per_bar)
        .map(|&v| v - rf_per_bar)
        .sum();
    let omega_denom: f64 = valid_r
        .iter()
        .filter(|&&v| v <= rf_per_bar)
        .map(|&v| rf_per_bar - v)
        .sum();
    let omega_ratio = if omega_denom > 0.0 {
        omega_numer / omega_denom
    } else {
        f64::INFINITY
    };

    // Tail ratio — use select_nth_unstable for O(n) percentile lookup
    let mut pct_r = valid_r.clone();
    let idx_5 = ((n_valid as f64 * 0.05) as usize).min(n_valid.saturating_sub(1));
    let idx_95 = ((n_valid as f64 * 0.95) as usize).min(n_valid.saturating_sub(1));
    // Find 5th percentile (also partitions so all elements below idx_5 are <=)
    pct_r.select_nth_unstable_by(idx_5, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let p5 = pct_r[idx_5];
    let worst_bar = pct_r[..=idx_5]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    // Find 95th percentile in the remaining upper partition
    pct_r[idx_5..].select_nth_unstable_by(idx_95 - idx_5, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let p95 = pct_r[idx_95];
    let best_bar = pct_r[idx_95..]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let tail_ratio = if p5.abs() > 0.0 {
        p95.abs() / p5.abs()
    } else {
        f64::INFINITY
    };

    // Position changes
    let mut n_pos_changes = 0_usize;
    for i in 1..n {
        let prev_active = r[i - 1].is_finite() && r[i - 1] != 0.0;
        let cur_active = r[i].is_finite() && r[i] != 0.0;
        if prev_active != cur_active {
            n_pos_changes += 1;
        }
    }

    // Benchmark metrics
    let (
        benchmark_total_return,
        benchmark_cagr,
        benchmark_annualized_vol,
        benchmark_sharpe,
        alpha,
        beta,
        tracking_error,
        information_ratio,
    ) = if let Some(br) = benchmark_returns {
        if br.len() == n {
            let mut s_sum = 0.0_f64;
            let mut b_sum = 0.0_f64;
            let mut nb = 0_usize;
            for i in 0..n {
                if r[i].is_finite() && br[i].is_finite() {
                    s_sum += r[i];
                    b_sum += br[i];
                    nb += 1;
                }
            }
            if nb > 1 {
                let s_mean = s_sum / nb as f64;
                let b_mean = b_sum / nb as f64;

                let mut b_var_sum = 0.0_f64;
                let mut cov_sum = 0.0_f64;
                let mut ex_sum = 0.0_f64;
                let mut ex_sq_sum = 0.0_f64;
                for i in 0..n {
                    if r[i].is_finite() && br[i].is_finite() {
                        let sd = r[i] - s_mean;
                        let bd = br[i] - b_mean;
                        b_var_sum += bd * bd;
                        cov_sum += sd * bd;
                        let ex = r[i] - br[i];
                        ex_sum += ex;
                        ex_sq_sum += ex * ex;
                    }
                }
                let b_var = b_var_sum / nb as f64;
                let b_std = b_var.sqrt();

                let mut b_eq = 1.0_f64;
                for &ret in br {
                    b_eq *= 1.0 + if ret.is_finite() { ret } else { 0.0 };
                }
                let bench_total_return = b_eq - 1.0;
                let bench_cagr = if b_eq > 0.0 {
                    b_eq.powf(periods_per_year / n as f64) - 1.0
                } else {
                    0.0
                };
                let bench_ann_vol = b_std * periods_per_year.sqrt();
                let bench_sharpe = if bench_ann_vol > 0.0 {
                    (bench_cagr - risk_free_rate) / bench_ann_vol
                } else {
                    0.0
                };

                let cov_val = cov_sum / nb as f64;
                let beta_val = if b_var > 0.0 { cov_val / b_var } else { 0.0 };
                let alpha_val = cagr - bench_cagr;

                let ex_mean = ex_sum / nb as f64;
                let ex_var = ex_sq_sum / nb as f64 - ex_mean * ex_mean;
                let te = ex_var.max(0.0).sqrt() * periods_per_year.sqrt();
                let ir = if te > 0.0 { alpha_val / te } else { 0.0 };

                (
                    Some(bench_total_return),
                    Some(bench_cagr),
                    Some(bench_ann_vol),
                    Some(bench_sharpe),
                    Some(alpha_val),
                    Some(beta_val),
                    Some(te),
                    Some(ir),
                )
            } else {
                (None, None, None, None, None, None, None, None)
            }
        } else {
            (None, None, None, None, None, None, None, None)
        }
    } else {
        (None, None, None, None, None, None, None, None)
    };

    Ok(BacktestMetrics {
        total_return,
        cagr,
        annualized_vol: annual_vol,
        sharpe,
        sortino,
        calmar,
        max_drawdown: max_dd,
        avg_drawdown: avg_dd,
        max_drawdown_duration_bars: max_dd_len,
        avg_drawdown_duration_bars: avg_dd_duration,
        ulcer_index,
        omega_ratio,
        win_rate,
        profit_factor,
        r_expectancy,
        avg_win,
        avg_loss,
        tail_ratio,
        skewness,
        kurtosis,
        best_bar,
        worst_bar,
        n_trades: n_active,
        n_position_changes: n_pos_changes,
        benchmark_total_return,
        benchmark_cagr,
        benchmark_annualized_vol,
        benchmark_sharpe,
        alpha,
        beta,
        tracking_error,
        information_ratio,
    })
}

// ---------------------------------------------------------------------------
// Trade extraction
// ---------------------------------------------------------------------------

/// Extract trade records from positions and price arrays.
pub fn extract_trades_ohlcv(
    positions: &[f64],
    fill_prices: &[f64],
    high: &[f64],
    low: &[f64],
) -> Result<Vec<TradeRecord>, String> {
    let n = positions.len();
    if fill_prices.len() != n || high.len() != n || low.len() != n {
        return Err(format!(
            "all arrays must have equal length (positions={}), got fill_prices={}, high={}, low={}",
            n,
            fill_prices.len(),
            high.len(),
            low.len()
        ));
    }

    let mut trades: Vec<TradeRecord> = Vec::new();

    let mut in_trade = false;
    let mut trade_entry_bar = 0_i64;
    let mut trade_dir = 0.0_f64;
    let mut trade_entry_price = 0.0_f64;
    let mut trade_mae = 0.0_f64;
    let mut trade_mfe = 0.0_f64;

    for i in 0..n {
        let cur_pos = positions[i];

        if !in_trade {
            if cur_pos != 0.0 {
                in_trade = true;
                trade_entry_bar = i as i64;
                trade_dir = cur_pos.signum();
                trade_entry_price = if fill_prices[i].is_finite() && fill_prices[i] > 0.0 {
                    fill_prices[i]
                } else {
                    high[i]
                };
                trade_mae = 0.0;
                trade_mfe = 0.0;
            }
        } else {
            if trade_entry_price > 0.0 {
                let unreal_high = trade_dir * (high[i] - trade_entry_price) / trade_entry_price;
                let unreal_low = trade_dir * (low[i] - trade_entry_price) / trade_entry_price;
                let bar_best = unreal_high.max(unreal_low);
                let bar_worst = unreal_high.min(unreal_low);
                if bar_best > trade_mfe {
                    trade_mfe = bar_best;
                }
                if bar_worst < trade_mae {
                    trade_mae = bar_worst;
                }
            }

            let pos_closed = cur_pos == 0.0 || cur_pos.signum() != trade_dir;

            if pos_closed {
                let exit_price = if fill_prices[i].is_finite() && fill_prices[i] > 0.0 {
                    fill_prices[i]
                } else {
                    low[i]
                };
                let pnl = if trade_entry_price > 0.0 {
                    trade_dir * (exit_price - trade_entry_price) / trade_entry_price
                } else {
                    0.0
                };

                trades.push(TradeRecord {
                    entry_bar: trade_entry_bar,
                    exit_bar: i as i64,
                    direction: trade_dir,
                    entry_price: trade_entry_price,
                    exit_price,
                    pnl_pct: pnl,
                    duration_bars: i as i64 - trade_entry_bar,
                    mae: trade_mae,
                    mfe: trade_mfe,
                });

                if cur_pos != 0.0 {
                    in_trade = true;
                    trade_entry_bar = i as i64;
                    trade_dir = cur_pos.signum();
                    trade_entry_price = if fill_prices[i].is_finite() && fill_prices[i] > 0.0 {
                        fill_prices[i]
                    } else {
                        high[i]
                    };
                    trade_mae = 0.0;
                    trade_mfe = 0.0;
                } else {
                    in_trade = false;
                }
            }
        }
    }

    // Close any open trade at last bar
    if in_trade {
        let last = n - 1;
        let exit_price = if fill_prices[last].is_finite() && fill_prices[last] > 0.0 {
            fill_prices[last]
        } else {
            high[last]
        };
        let pnl = if trade_entry_price > 0.0 {
            trade_dir * (exit_price - trade_entry_price) / trade_entry_price
        } else {
            0.0
        };
        trades.push(TradeRecord {
            entry_bar: trade_entry_bar,
            exit_bar: last as i64,
            direction: trade_dir,
            entry_price: trade_entry_price,
            exit_price,
            pnl_pct: pnl,
            duration_bars: last as i64 - trade_entry_bar,
            mae: trade_mae,
            mfe: trade_mfe,
        });
    }

    Ok(trades)
}

// ---------------------------------------------------------------------------
// Multi-asset backtest
// ---------------------------------------------------------------------------

/// Multi-asset result: per-asset strategy returns (n_bars x n_assets), portfolio returns, portfolio equity.
#[derive(Clone, Debug)]
pub struct MultiAssetBacktestResult {
    /// Shape: (n_assets, n_bars) — row-major per asset.
    pub asset_returns: Vec<Vec<f64>>,
    pub portfolio_returns: Vec<f64>,
    pub portfolio_equity: Vec<f64>,
}

/// Backtest N assets, then combine into a portfolio.
///
/// `close_2d`: row-major (n_assets, n_bars)
/// `weights_2d`: row-major (n_assets, n_bars)
///
/// Callers must transpose from (n_bars, n_assets) if needed.
#[allow(clippy::too_many_arguments)]
pub fn backtest_multi_asset_core(
    close_2d: &[Vec<f64>],
    weights_2d: &[Vec<f64>],
    n_bars: usize,
    n_assets: usize,
    commission_per_trade: f64,
    slippage_bps: f64,
    max_asset_weight: f64,
    max_gross_exposure: f64,
    max_net_exposure: f64,
) -> Result<MultiAssetBacktestResult, String> {
    if n_bars < 2 {
        return Err("n_bars must be at least 2".to_string());
    }
    if close_2d.len() != n_assets || weights_2d.len() != n_assets {
        return Err("close_2d and weights_2d must have n_assets rows".to_string());
    }

    // Apply portfolio constraints per bar
    let mut constrained: Vec<Vec<f64>> = weights_2d.to_vec();
    #[allow(clippy::needless_range_loop)]
    if max_asset_weight != 1.0 || max_gross_exposure > 0.0 || max_net_exposure > 0.0 {
        for i in 0..n_bars {
            // 1. Clamp per-asset weight
            if max_asset_weight < f64::INFINITY && max_asset_weight > 0.0 {
                for j in 0..n_assets {
                    let w = constrained[j][i];
                    if w.abs() > max_asset_weight {
                        constrained[j][i] = w.signum() * max_asset_weight;
                    }
                }
            }
            // 2. Normalize so sum(abs) <= max_gross_exposure
            if max_gross_exposure > 0.0 {
                let gross: f64 = (0..n_assets).map(|j| constrained[j][i].abs()).sum();
                if gross > max_gross_exposure {
                    let scale = max_gross_exposure / gross;
                    for j in 0..n_assets {
                        constrained[j][i] *= scale;
                    }
                }
            }
            // 3. Clamp net exposure
            if max_net_exposure > 0.0 {
                let net: f64 = (0..n_assets).map(|j| constrained[j][i]).sum();
                if net.abs() > max_net_exposure {
                    let excess = net - net.signum() * max_net_exposure;
                    let adj_per_asset = excess / n_assets as f64;
                    for j in 0..n_assets {
                        constrained[j][i] -= adj_per_asset;
                    }
                }
            }
        }
    }

    // Per-asset backtests
    let asset_strategy_returns: Vec<Vec<f64>> = (0..n_assets)
        .map(|j| {
            let (_, strat_rets, _) = single_asset_backtest(
                &close_2d[j],
                &constrained[j],
                commission_per_trade,
                slippage_bps,
            );
            strat_rets
        })
        .collect();

    // Portfolio return = sum of per-asset strategy returns
    let mut portfolio_returns = vec![0.0_f64; n_bars];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_bars {
        let mut s = 0.0_f64;
        for j in 0..n_assets {
            s += asset_strategy_returns[j][i];
        }
        portfolio_returns[i] = s;
    }

    // Portfolio equity
    let mut portfolio_equity = vec![1.0_f64; n_bars];
    let mut cum = 1.0_f64;
    for i in 0..n_bars {
        cum *= 1.0 + portfolio_returns[i];
        portfolio_equity[i] = cum;
    }

    Ok(MultiAssetBacktestResult {
        asset_returns: asset_strategy_returns,
        portfolio_returns,
        portfolio_equity,
    })
}

// ---------------------------------------------------------------------------
// Monte Carlo bootstrap
// ---------------------------------------------------------------------------

/// Bootstrap Monte Carlo simulation over strategy returns.
///
/// Returns `n_sims` equity curves, each of length `n_bars`.
pub fn monte_carlo_bootstrap(
    strategy_returns: &[f64],
    n_sims: usize,
    seed: u64,
    block_size: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let n = strategy_returns.len();
    if n < 2 {
        return Err("strategy_returns must have at least 2 elements".to_string());
    }
    if n_sims == 0 {
        return Err("n_sims must be >= 1".to_string());
    }
    let bsize = block_size.max(1).min(n);

    let mut result: Vec<Vec<f64>> = Vec::with_capacity(n_sims);

    for sim_idx in 0..n_sims {
        let mut state = seed
            .wrapping_mul(6_364_136_223_846_793_005_u64)
            .wrapping_add((sim_idx as u64).wrapping_mul(2_862_933_555_777_941_757_u64));
        lcg_next(&mut state);
        lcg_next(&mut state);

        let mut row = vec![0.0_f64; n];

        if bsize == 1 {
            for dst in row.iter_mut() {
                *dst = strategy_returns[lcg_index(&mut state, n)];
            }
        } else {
            let mut filled = 0_usize;
            while filled < n {
                let start = lcg_index(&mut state, n);
                let take = bsize.min(n - filled);
                for k in 0..take {
                    row[filled + k] = strategy_returns[(start + k) % n];
                }
                filled += take;
            }
        }

        // Convert to equity curve in-place
        let mut cum = 1.0_f64;
        for elem in row.iter_mut() {
            cum *= 1.0 + *elem;
            *elem = cum;
        }

        result.push(row);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Walk-forward indices
// ---------------------------------------------------------------------------

/// Generate train/test fold index boundaries for walk-forward analysis.
///
/// Returns a vector of (train_start, train_end, test_start, test_end) tuples.
pub fn walk_forward_indices(
    n_bars: usize,
    train_bars: usize,
    test_bars: usize,
    anchored: bool,
    step_bars: usize,
) -> Result<Vec<[i64; 4]>, String> {
    if train_bars == 0 {
        return Err("train_bars must be >= 1".to_string());
    }
    if test_bars == 0 {
        return Err("test_bars must be >= 1".to_string());
    }
    if train_bars + test_bars > n_bars {
        return Err("train_bars + test_bars must be <= n_bars".to_string());
    }

    let step = if step_bars == 0 { test_bars } else { step_bars };
    let mut folds: Vec<[i64; 4]> = Vec::new();

    let mut offset = 0_usize;
    loop {
        let train_start = if anchored { 0 } else { offset };
        let train_end = offset + train_bars;
        let test_start = train_end;
        let test_end = test_start + test_bars;

        if test_end > n_bars {
            break;
        }

        folds.push([
            train_start as i64,
            train_end as i64,
            test_start as i64,
            test_end as i64,
        ]);

        offset += step;
    }

    if folds.is_empty() {
        return Err(
            "No complete folds fit within n_bars with the given train/test sizes".to_string(),
        );
    }

    Ok(folds)
}

// ---------------------------------------------------------------------------
// Kelly criterion
// ---------------------------------------------------------------------------

/// Compute the Kelly fraction: f = win_rate - (1 - win_rate) * (|avg_loss| / avg_win), clamped to [0, 1].
pub fn kelly_fraction(win_rate: f64, avg_win: f64, avg_loss: f64) -> Result<f64, String> {
    if !(0.0..=1.0).contains(&win_rate) {
        return Err("win_rate must be in [0, 1]".to_string());
    }
    if avg_win <= 0.0 {
        return Err("avg_win must be > 0".to_string());
    }
    Ok(kelly_formula(win_rate, avg_win, avg_loss))
}

/// Half-Kelly fraction (conservative position sizing).
pub fn half_kelly_fraction(win_rate: f64, avg_win: f64, avg_loss: f64) -> Result<f64, String> {
    Ok(kelly_fraction(win_rate, avg_win, avg_loss)? / 2.0)
}

// ---------------------------------------------------------------------------
// StreamingBacktest
// ---------------------------------------------------------------------------

impl StreamingBacktest {
    pub fn new(commission_per_trade: f64, slippage_bps: f64) -> Self {
        StreamingBacktest {
            commission_per_trade,
            slippage_bps,
            position: 0.0,
            entry_price: f64::NAN,
            equity: 1.0,
            prev_close: f64::NAN,
            total_commission: 0.0,
            n_trades: 0,
            sum_wins: 0.0,
            n_wins: 0,
            sum_losses: 0.0,
            n_losses: 0,
        }
    }

    /// Process one bar. Returns position, bar_return, equity, n_trades.
    pub fn on_bar(&mut self, close: f64, signal: f64) -> StreamingBarResult {
        let slip = self.slippage_bps / 10_000.0;
        let mut bar_return = 0.0_f64;

        if self.position != 0.0 && !self.prev_close.is_nan() {
            let price_ret = (close - self.prev_close) / self.prev_close;
            bar_return = self.position * price_ret;
            self.equity *= 1.0 + bar_return;
        }

        let new_pos = if signal.is_nan() { 0.0 } else { signal };
        if (new_pos - self.position).abs() > 1e-12 {
            let direction = if new_pos > self.position { 1.0 } else { -1.0 };
            let slippage_cost = direction * slip;
            self.equity *= 1.0 - slippage_cost.abs();
            self.equity -= self.commission_per_trade;
            self.total_commission += self.commission_per_trade;

            if self.position != 0.0 && !self.entry_price.is_nan() {
                let trade_ret = self.position * (close - self.entry_price) / self.entry_price;
                if trade_ret >= 0.0 {
                    self.sum_wins += trade_ret;
                    self.n_wins += 1;
                } else {
                    self.sum_losses += trade_ret.abs();
                    self.n_losses += 1;
                }
                self.n_trades += 1;
            }

            self.position = new_pos;
            self.entry_price = if new_pos != 0.0 { close } else { f64::NAN };
        }

        self.prev_close = close;

        StreamingBarResult {
            position: self.position,
            bar_return,
            equity: self.equity,
            n_trades: self.n_trades,
        }
    }

    /// Summary statistics.
    pub fn summary(&self) -> StreamingSummary {
        let win_rate = if self.n_trades > 0 {
            self.n_wins as f64 / self.n_trades as f64
        } else {
            0.0
        };
        let avg_win = if self.n_wins > 0 {
            self.sum_wins / self.n_wins as f64
        } else {
            0.0
        };
        let avg_loss = if self.n_losses > 0 {
            self.sum_losses / self.n_losses as f64
        } else {
            0.0
        };
        let kf = kelly_formula(win_rate, avg_win, avg_loss);

        StreamingSummary {
            equity: self.equity,
            n_trades: self.n_trades,
            total_commission: self.total_commission,
            win_rate,
            avg_win,
            avg_loss,
            kelly_fraction: kf,
        }
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.position = 0.0;
        self.entry_price = f64::NAN;
        self.equity = 1.0;
        self.prev_close = f64::NAN;
        self.total_commission = 0.0;
        self.n_trades = 0;
        self.sum_wins = 0.0;
        self.n_wins = 0;
        self.sum_losses = 0.0;
        self.n_losses = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_to_num() {
        assert_eq!(nan_to_num(f64::NAN), 0.0);
        assert_eq!(nan_to_num(f64::INFINITY), f64::MAX);
        assert_eq!(nan_to_num(f64::NEG_INFINITY), -f64::MAX);
        assert_eq!(nan_to_num(42.0), 42.0);
    }

    #[test]
    fn test_kelly_formula_basic() {
        let f = kelly_formula(0.6, 1.0, 0.5);
        assert!((f - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_kelly_fraction_validation() {
        assert!(kelly_fraction(1.5, 1.0, 0.5).is_err());
        assert!(kelly_fraction(0.5, -1.0, 0.5).is_err());
        assert!(kelly_fraction(0.6, 1.0, 0.5).is_ok());
    }

    #[test]
    fn test_half_kelly() {
        let full = kelly_fraction(0.6, 1.0, 0.5).unwrap();
        let half = half_kelly_fraction(0.6, 1.0, 0.5).unwrap();
        assert!((half - full / 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_backtest_core_flat_signal() {
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let signals = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let result = backtest_core(&close, &signals, None, 0.0, 100_000.0, 0.0).unwrap();
        // With zero signals, equity should remain at 1.0
        for &e in &result.equity {
            assert!((e - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_backtest_core_long_signal() {
        let close = vec![100.0, 110.0, 120.0];
        let signals = vec![1.0, 1.0, 1.0];
        let result = backtest_core(&close, &signals, None, 0.0, 100_000.0, 0.0).unwrap();
        // Position is lagged: pos[0]=0, pos[1]=1, pos[2]=1
        // bar_returns[1] = 0.1, bar_returns[2] ≈ 0.0909
        // strategy_returns[1] = 1*0.1 = 0.1, strategy_returns[2] = 1*0.0909
        assert!((result.equity[2] - 1.1 * (1.0 + 10.0 / 110.0)).abs() < 1e-10);
    }

    #[test]
    fn test_walk_forward_indices_basic() {
        let folds = walk_forward_indices(100, 50, 25, false, 0).unwrap();
        assert_eq!(folds.len(), 2);
        assert_eq!(folds[0], [0, 50, 50, 75]);
        assert_eq!(folds[1], [25, 75, 75, 100]);
    }

    #[test]
    fn test_walk_forward_anchored() {
        let folds = walk_forward_indices(100, 50, 25, true, 0).unwrap();
        assert!(folds.len() >= 2);
        // Anchored: train always starts at 0
        for fold in &folds {
            assert_eq!(fold[0], 0);
        }
    }

    #[test]
    fn test_monte_carlo_basic() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let result = monte_carlo_bootstrap(&returns, 10, 42, 1).unwrap();
        assert_eq!(result.len(), 10);
        for curve in &result {
            assert_eq!(curve.len(), 5);
            // Equity curves should be positive
            assert!(curve.last().unwrap() > &0.0);
        }
    }

    #[test]
    fn test_extract_trades_empty() {
        let positions = vec![0.0, 0.0, 0.0];
        let fill_prices = vec![f64::NAN, f64::NAN, f64::NAN];
        let high = vec![100.0, 101.0, 102.0];
        let low = vec![99.0, 100.0, 101.0];
        let trades = extract_trades_ohlcv(&positions, &fill_prices, &high, &low).unwrap();
        assert!(trades.is_empty());
    }

    #[test]
    fn test_extract_trades_single_roundtrip() {
        let positions = vec![0.0, 1.0, 1.0, 0.0];
        let fill_prices = vec![f64::NAN, 100.0, f64::NAN, 110.0];
        let high = vec![100.0, 105.0, 115.0, 112.0];
        let low = vec![98.0, 99.0, 100.0, 108.0];
        let trades = extract_trades_ohlcv(&positions, &fill_prices, &high, &low).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].entry_bar, 1);
        assert_eq!(trades[0].exit_bar, 3);
        assert!((trades[0].entry_price - 100.0).abs() < 1e-10);
        assert!((trades[0].exit_price - 110.0).abs() < 1e-10);
        assert!(trades[0].pnl_pct > 0.0);
    }

    #[test]
    fn test_ohlcv_backtest_basic() {
        let n = 10;
        let open: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let high: Vec<f64> = open.iter().map(|&v| v + 2.0).collect();
        let low: Vec<f64> = open.iter().map(|&v| v - 2.0).collect();
        let close: Vec<f64> = open.iter().map(|&v| v + 1.0).collect();
        let signals: Vec<f64> = vec![0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0];

        let config = BacktestConfig::default();
        let result =
            backtest_ohlcv_core(&open, &high, &low, &close, &signals, &config, None).unwrap();
        assert_eq!(result.equity.len(), n);
        // Equity should be positive
        assert!(*result.equity.last().unwrap() > 0.0);
    }

    #[test]
    fn test_streaming_backtest() {
        let mut engine = StreamingBacktest::new(0.0, 0.0);
        let closes = vec![100.0, 105.0, 103.0, 110.0];
        let signals = vec![1.0, 1.0, -1.0, 0.0];

        for (&c, &s) in closes.iter().zip(signals.iter()) {
            let _r = engine.on_bar(c, s);
        }
        assert!(engine.equity > 0.0);
        let summary = engine.summary();
        assert!(summary.n_trades > 0);
    }

    #[test]
    fn test_compute_performance_metrics_basic() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.003, 0.008];
        let mut equity = vec![1.0_f64; returns.len()];
        let mut cum = 1.0;
        for (i, &r) in returns.iter().enumerate() {
            cum *= 1.0 + r;
            equity[i] = cum;
        }
        let metrics = compute_performance_metrics(&returns, &equity, 252.0, 0.0, None).unwrap();
        assert!(metrics.total_return > 0.0);
        assert!(metrics.sharpe != 0.0);
        assert!(metrics.n_trades > 0);
    }

    #[test]
    fn test_multi_asset_basic() {
        let n_bars = 5;
        let close1 = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let close2 = vec![200.0, 198.0, 201.0, 203.0, 205.0];
        let weights1 = vec![0.0, 0.5, 0.5, 0.5, 0.0];
        let weights2 = vec![0.0, 0.5, 0.5, 0.5, 0.0];

        let result = backtest_multi_asset_core(
            &[close1, close2],
            &[weights1, weights2],
            n_bars,
            2,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        .unwrap();

        assert_eq!(result.portfolio_returns.len(), n_bars);
        assert_eq!(result.portfolio_equity.len(), n_bars);
        assert_eq!(result.asset_returns.len(), 2);
    }

    #[test]
    fn test_sma_crossover_signals() {
        let close: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let signals = sma_crossover_signals(&close, 5, 10).unwrap();
        assert_eq!(signals.len(), close.len());
        // First 9 bars should be NaN (slow SMA warm-up)
        for i in 0..9 {
            assert!(signals[i].is_nan(), "bar {} should be NaN", i);
        }
    }

    #[test]
    fn test_sma_crossover_invalid() {
        let close = vec![1.0; 20];
        assert!(sma_crossover_signals(&close, 10, 5).is_err());
    }

    #[test]
    fn test_rsi_threshold_signals() {
        let close: Vec<f64> = (1..=30).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let signals = rsi_threshold_signals(&close, 14, 30.0, 70.0);
        assert_eq!(signals.len(), close.len());
    }
}
