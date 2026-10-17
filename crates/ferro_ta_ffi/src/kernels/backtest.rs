//! Kernel C exports for the OHLCV backtest engine and performance metrics.
//!
//! The engine's ~30 knobs cross the ABI as a **settings vector**: a `double`
//! array indexed by the `FT_BT_*` constants (see
//! [`crate::enums::BACKTEST_SETTING`]). Unlike a C struct it can grow without
//! breaking callers: a shorter vector (from an older header) gets defaults for
//! the missing tail; a longer one (from a newer header) is rejected rather
//! than silently ignored.

use super::invalid;
use crate::marshal::{check_output, kernel_in, kernel_in_opt, kernel_out, MAX_PERIOD};
use crate::status::Status;
use ferro_ta_core::backtest::{
    backtest_ohlcv_volume_core, compute_performance_metrics_with_positions,
    resolve_commission_model, BacktestConfig, BacktestMetrics,
};
use ferro_ta_core::commission::CommissionModel;

/// Settings-vector indices; must match `FT_BT_*` (checked by a test).
mod key {
    pub const FILL_MODE: usize = 0;
    pub const STOP_LOSS_PCT: usize = 1;
    pub const TAKE_PROFIT_PCT: usize = 2;
    pub const TRAILING_STOP_PCT: usize = 3;
    pub const BREAKEVEN_PCT: usize = 4;
    pub const MAX_HOLD_BARS: usize = 5;
    pub const SLIPPAGE_BPS: usize = 6;
    pub const SLIPPAGE_PCT_RANGE: usize = 7;
    pub const SLIPPAGE_ATR_MULT: usize = 8;
    pub const SLIPPAGE_ATR_PERIOD: usize = 9;
    pub const MAX_PARTICIPATION: usize = 10;
    pub const INITIAL_CAPITAL: usize = 11;
    pub const PERIODS_PER_YEAR: usize = 12;
    pub const MARGIN_RATIO: usize = 13;
    pub const MARGIN_CALL_PCT: usize = 14;
    pub const DAILY_LOSS_LIMIT: usize = 15;
    pub const TOTAL_LOSS_LIMIT: usize = 16;
    pub const FLAT_PER_ORDER: usize = 17;
    pub const RATE_OF_VALUE: usize = 18;
    pub const PER_LOT: usize = 19;
    pub const MAX_BROKERAGE: usize = 20;
    pub const SPREAD_BPS: usize = 21;
    pub const STT_RATE: usize = 22;
    pub const STT_ON_BUY: usize = 23;
    pub const STT_ON_SELL: usize = 24;
    pub const EXCHANGE_CHARGES_RATE: usize = 25;
    pub const REGULATORY_CHARGES_RATE: usize = 26;
    pub const GST_RATE: usize = 27;
    pub const STAMP_DUTY_RATE: usize = 28;
    pub const LOT_SIZE: usize = 29;
    pub const SHORT_BORROW_RATE_ANNUAL: usize = 30;
}

/// Length of the settings vector this library understands.
pub const N_SETTINGS: usize = 31;
const N_METRICS: usize = BacktestMetrics::N_SCALAR_FIELDS;
/// Columns of `out_benchmark` (`FT_BENCH_*`).
pub const N_BENCHMARK_METRICS: usize = 8;

fn flag(on: bool) -> f64 {
    if on {
        1.0
    } else {
        0.0
    }
}

/// Encode a config (commission included) as a settings vector.
pub fn settings_from_config(cfg: &BacktestConfig) -> [f64; N_SETTINGS] {
    let cm = resolve_commission_model(cfg.commission.as_ref(), cfg.commission_per_trade);
    let mut s = [0.0; N_SETTINGS];
    s[key::FILL_MODE] = flag(cfg.fill_mode == "market_close");
    s[key::STOP_LOSS_PCT] = cfg.stop_loss_pct;
    s[key::TAKE_PROFIT_PCT] = cfg.take_profit_pct;
    s[key::TRAILING_STOP_PCT] = cfg.trailing_stop_pct;
    s[key::BREAKEVEN_PCT] = cfg.breakeven_pct;
    s[key::MAX_HOLD_BARS] = cfg.max_hold_bars as f64;
    s[key::SLIPPAGE_BPS] = cfg.slippage_bps;
    s[key::SLIPPAGE_PCT_RANGE] = cfg.slippage_pct_range;
    s[key::SLIPPAGE_ATR_MULT] = cfg.slippage_atr_mult;
    s[key::SLIPPAGE_ATR_PERIOD] = cfg.slippage_atr_period as f64;
    s[key::MAX_PARTICIPATION] = cfg.max_participation;
    s[key::INITIAL_CAPITAL] = cfg.initial_capital;
    s[key::PERIODS_PER_YEAR] = cfg.periods_per_year;
    s[key::MARGIN_RATIO] = cfg.margin_ratio;
    s[key::MARGIN_CALL_PCT] = cfg.margin_call_pct;
    s[key::DAILY_LOSS_LIMIT] = cfg.daily_loss_limit;
    s[key::TOTAL_LOSS_LIMIT] = cfg.total_loss_limit;
    s[key::FLAT_PER_ORDER] = cm.flat_per_order;
    s[key::RATE_OF_VALUE] = cm.rate_of_value;
    s[key::PER_LOT] = cm.per_lot;
    s[key::MAX_BROKERAGE] = cm.max_brokerage;
    s[key::SPREAD_BPS] = cm.spread_bps;
    s[key::STT_RATE] = cm.stt_rate;
    s[key::STT_ON_BUY] = flag(cm.stt_on_buy);
    s[key::STT_ON_SELL] = flag(cm.stt_on_sell);
    s[key::EXCHANGE_CHARGES_RATE] = cm.exchange_charges_rate;
    s[key::REGULATORY_CHARGES_RATE] = cm.regulatory_charges_rate;
    s[key::GST_RATE] = cm.gst_rate;
    s[key::STAMP_DUTY_RATE] = cm.stamp_duty_rate;
    s[key::LOT_SIZE] = cm.lot_size;
    s[key::SHORT_BORROW_RATE_ANNUAL] = cm.short_borrow_rate_annual;
    s
}

fn whole(v: f64, min: f64) -> Result<usize, Status> {
    if v.fract() != 0.0 || v < min || v > MAX_PERIOD as f64 {
        return Err(Status::InvalidParam);
    }
    Ok(v as usize)
}

fn boolean(v: f64) -> Result<bool, Status> {
    match v {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(Status::InvalidParam),
    }
}

/// Decode a (possibly shorter) settings vector over the defaults.
///
/// Every value must be finite; enum, flag and integer entries must be valid.
/// Range rules beyond that (e.g. `max_participation >= 0`) are the engine's.
pub fn config_from_settings(given: &[f64]) -> Result<BacktestConfig, Status> {
    if given.len() > N_SETTINGS || given.iter().any(|v| !v.is_finite()) {
        return Err(Status::InvalidParam);
    }
    let mut s = settings_from_config(&BacktestConfig::default());
    s[..given.len()].copy_from_slice(given);
    let fill_mode = match s[key::FILL_MODE] {
        0.0 => "market_open",
        1.0 => "market_close",
        _ => return Err(Status::InvalidParam),
    };
    let commission = CommissionModel {
        flat_per_order: s[key::FLAT_PER_ORDER],
        rate_of_value: s[key::RATE_OF_VALUE],
        per_lot: s[key::PER_LOT],
        max_brokerage: s[key::MAX_BROKERAGE],
        spread_bps: s[key::SPREAD_BPS],
        stt_rate: s[key::STT_RATE],
        stt_on_buy: boolean(s[key::STT_ON_BUY])?,
        stt_on_sell: boolean(s[key::STT_ON_SELL])?,
        exchange_charges_rate: s[key::EXCHANGE_CHARGES_RATE],
        regulatory_charges_rate: s[key::REGULATORY_CHARGES_RATE],
        gst_rate: s[key::GST_RATE],
        stamp_duty_rate: s[key::STAMP_DUTY_RATE],
        lot_size: s[key::LOT_SIZE],
        short_borrow_rate_annual: s[key::SHORT_BORROW_RATE_ANNUAL],
    };
    Ok(BacktestConfig {
        fill_mode: fill_mode.to_string(),
        stop_loss_pct: s[key::STOP_LOSS_PCT],
        take_profit_pct: s[key::TAKE_PROFIT_PCT],
        trailing_stop_pct: s[key::TRAILING_STOP_PCT],
        slippage_bps: s[key::SLIPPAGE_BPS],
        initial_capital: s[key::INITIAL_CAPITAL],
        commission_per_trade: 0.0,
        max_hold_bars: whole(s[key::MAX_HOLD_BARS], 0.0)?,
        slippage_pct_range: s[key::SLIPPAGE_PCT_RANGE],
        breakeven_pct: s[key::BREAKEVEN_PCT],
        periods_per_year: s[key::PERIODS_PER_YEAR],
        margin_ratio: s[key::MARGIN_RATIO],
        margin_call_pct: s[key::MARGIN_CALL_PCT],
        daily_loss_limit: s[key::DAILY_LOSS_LIMIT],
        total_loss_limit: s[key::TOTAL_LOSS_LIMIT],
        commission: Some(commission),
        max_participation: s[key::MAX_PARTICIPATION],
        slippage_atr_mult: s[key::SLIPPAGE_ATR_MULT],
        // 0 is fine while ATR slippage is off; the engine rejects it otherwise.
        slippage_atr_period: whole(s[key::SLIPPAGE_ATR_PERIOD], 0.0)?,
    })
}

crate::ffi_kernel_exports! {
    group: "backtest",
    /// Write the first `n_settings` entries of the default settings vector for ft_backtest_ohlcv (index with FT_BT_*): market-open fills, no stops, no costs, 100000 initial capital, 252 periods per year. Pass your array's own length (at most ft_backtest_n_settings()), so code built against an older header stays in bounds.
    ft_backtest_default_settings(
        n_settings: size usize <- ft_backtest_n_settings(),
        out: out f64 [n_settings],
    ) => {
        if n_settings > N_SETTINGS {
            return Err(Status::InvalidParam);
        }
        check_output(out, n_settings)?;
        let defaults = settings_from_config(&BacktestConfig::default());
        // SAFETY: checked non-null; valid for n_settings writes.
        unsafe { kernel_out(out, &defaults[..n_settings], n_settings) }
    };
    /// OHLCV backtest engine: lagged signals fill at the bar open (or close), with stop-loss, take-profit, trailing and breakeven stops, max-hold exits, limit orders, leverage/margin calls, loss circuit breakers, the full commission model, ATR slippage and volume-capped partial fills. `settings` is indexed by FT_BT_* and may be shorter than ft_backtest_n_settings() (missing entries take defaults). `volume` (required when FT_BT_MAX_PARTICIPATION > 0) and `limit_prices` (NaN = market order) may be NULL. Equity starts at 1.0.
    ft_backtest_ohlcv(
        open: in f64 [n_bars],
        high: in f64 [n_bars],
        low: in f64 [n_bars],
        close: in f64 [n_bars],
        volume: in_opt f64 [n_bars],
        signals: in f64 [n_bars],
        limit_prices: in_opt f64 [n_bars],
        n_bars: size usize,
        settings: in f64 [n_settings],
        n_settings: size usize,
        out_positions: out f64 [n_bars],
        out_fill_prices: out f64 [n_bars],
        out_bar_returns: out f64 [n_bars],
        out_strategy_returns: out f64 [n_bars],
        out_equity: out f64 [n_bars],
    ) => {
        for out in [out_positions, out_fill_prices, out_bar_returns, out_strategy_returns, out_equity] {
            check_output(out, n_bars)?;
        }
        // SAFETY: caller contract — every input valid for its shape.
        let (o, h, l, c, s, cfg_raw, vol, lp) = unsafe {
            (
                kernel_in(open, n_bars)?,
                kernel_in(high, n_bars)?,
                kernel_in(low, n_bars)?,
                kernel_in(close, n_bars)?,
                kernel_in(signals, n_bars)?,
                kernel_in(settings, n_settings)?,
                kernel_in_opt(volume, n_bars),
                kernel_in_opt(limit_prices, n_bars),
            )
        };
        let cfg = config_from_settings(cfg_raw)?;
        let r = backtest_ohlcv_volume_core(o, h, l, c, vol, s, &cfg, lp).map_err(invalid)?;
        // SAFETY: outputs checked non-null; valid for n_bars writes each.
        unsafe {
            kernel_out(out_positions, &r.positions, n_bars)?;
            kernel_out(out_fill_prices, &r.fill_prices, n_bars)?;
            kernel_out(out_bar_returns, &r.bar_returns, n_bars)?;
            kernel_out(out_strategy_returns, &r.strategy_returns, n_bars)?;
            kernel_out(out_equity, &r.equity, n_bars)
        }
    };
    /// Performance metrics of a strategy-return / equity series (e.g. from ft_backtest_ohlcv), written as one row indexed by FT_METRIC_*. With `positions` (optional) trades are counted from position changes; otherwise inferred from non-zero return runs. With `benchmark_returns` (optional, per-bar returns) out_benchmark gets the FT_BENCH_* metrics; without it, NaN. n_metric_cols / n_benchmark_cols must equal ft_sweep_n_metrics() / ft_benchmark_n_metrics().
    ft_performance_metrics(
        strategy_returns: in f64 [n_bars],
        equity: in f64 [n_bars],
        benchmark_returns: in_opt f64 [n_bars],
        positions: in_opt f64 [n_bars],
        n_bars: size usize,
        periods_per_year: param float,
        risk_free_rate: param float,
        n_metric_cols: size usize <- ft_sweep_n_metrics(),
        n_benchmark_cols: size usize <- ft_benchmark_n_metrics(),
        out_metrics: out f64 [n_metric_cols],
        out_benchmark: out f64 [n_benchmark_cols],
    ) => {
        if n_metric_cols != N_METRICS || n_benchmark_cols != N_BENCHMARK_METRICS {
            return Err(Status::InvalidParam);
        }
        check_output(out_metrics, n_metric_cols)?;
        check_output(out_benchmark, n_benchmark_cols)?;
        // SAFETY: caller contract — inputs valid for n_bars reads (optional ones may be NULL).
        let (r, eq, bench, pos) = unsafe {
            (
                kernel_in(strategy_returns, n_bars)?,
                kernel_in(equity, n_bars)?,
                kernel_in_opt(benchmark_returns, n_bars),
                kernel_in_opt(positions, n_bars),
            )
        };
        let m = compute_performance_metrics_with_positions(
            r, eq, periods_per_year, risk_free_rate, bench, pos,
        )
        .map_err(invalid)?;
        let row = m.scalar_fields().map(|(_, v)| v);
        let bench_row = [
            m.benchmark_total_return,
            m.benchmark_cagr,
            m.benchmark_annualized_vol,
            m.benchmark_sharpe,
            m.alpha,
            m.beta,
            m.tracking_error,
            m.information_ratio,
        ]
        .map(|v| v.unwrap_or(f64::NAN));
        // SAFETY: checked non-null; valid for their lengths.
        unsafe {
            kernel_out(out_metrics, &row, n_metric_cols)?;
            kernel_out(out_benchmark, &bench_row, n_benchmark_cols)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::BACKTEST_SETTING;

    #[test]
    fn key_table_matches_c_enum() {
        assert_eq!(BACKTEST_SETTING.values.len(), N_SETTINGS);
        let cfg = BacktestConfig {
            stop_loss_pct: 0.123,
            slippage_atr_period: 7,
            ..Default::default()
        };
        let s = settings_from_config(&cfg);
        let idx = |name: &str| {
            BACKTEST_SETTING
                .values
                .iter()
                .find(|(n, _)| *n == name)
                .unwrap()
                .1 as usize
        };
        assert_eq!(s[idx("STOP_LOSS_PCT")], 0.123);
        assert_eq!(s[idx("SLIPPAGE_ATR_PERIOD")], 7.0);
        assert_eq!(s[idx("LOT_SIZE")], 1.0);
        assert_eq!(s[idx("INITIAL_CAPITAL")], 100_000.0);
    }

    #[test]
    fn settings_round_trip() {
        let cfg = BacktestConfig {
            fill_mode: "market_close".into(),
            max_hold_bars: 5,
            max_participation: 0.1,
            commission: Some(CommissionModel {
                rate_of_value: 0.001,
                stt_on_sell: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let s = settings_from_config(&cfg);
        assert_eq!(settings_from_config(&config_from_settings(&s).unwrap()), s);
    }

    #[test]
    fn short_vectors_take_defaults_long_and_bad_ones_are_rejected() {
        let defaults = settings_from_config(&BacktestConfig::default());
        let short = config_from_settings(&[0.0, 0.05]).unwrap();
        let mut want = defaults;
        want[key::STOP_LOSS_PCT] = 0.05;
        assert_eq!(settings_from_config(&short), want);
        assert!(config_from_settings(&[0.0; N_SETTINGS + 1]).is_err());
        let mut bad = defaults;
        bad[key::FILL_MODE] = 2.0;
        assert!(config_from_settings(&bad).is_err());
        bad = defaults;
        bad[key::MAX_HOLD_BARS] = 2.5;
        assert!(config_from_settings(&bad).is_err());
        bad = defaults;
        bad[key::STT_ON_BUY] = 0.5;
        assert!(config_from_settings(&bad).is_err());
        bad = defaults;
        bad[key::SLIPPAGE_BPS] = f64::NAN;
        assert!(config_from_settings(&bad).is_err());
    }
}
