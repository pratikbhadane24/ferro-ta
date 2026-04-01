//! Thin PyO3 wrappers delegating to `ferro_ta_core::backtest`.

pub mod commission;
pub mod currency;

use commission::PyCommissionModel;
use currency::PyCurrency;
use ferro_ta_core::backtest as core_bt;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::validation;

// ---------------------------------------------------------------------------
// BacktestConfig pyclass wrapping core struct
// ---------------------------------------------------------------------------

#[pyclass(name = "BacktestConfig")]
#[derive(Clone)]
pub struct BacktestConfig {
    #[pyo3(get, set)]
    pub fill_mode: String,
    #[pyo3(get, set)]
    pub stop_loss_pct: f64,
    #[pyo3(get, set)]
    pub take_profit_pct: f64,
    #[pyo3(get, set)]
    pub trailing_stop_pct: f64,
    #[pyo3(get, set)]
    pub slippage_bps: f64,
    #[pyo3(get, set)]
    pub initial_capital: f64,
    #[pyo3(get, set)]
    pub commission_per_trade: f64,
    #[pyo3(get, set)]
    pub max_hold_bars: usize,
    #[pyo3(get, set)]
    pub slippage_pct_range: f64,
    #[pyo3(get, set)]
    pub breakeven_pct: f64,
    #[pyo3(get, set)]
    pub periods_per_year: f64,
    #[pyo3(get, set)]
    pub margin_ratio: f64,
    #[pyo3(get, set)]
    pub margin_call_pct: f64,
    #[pyo3(get, set)]
    pub daily_loss_limit: f64,
    #[pyo3(get, set)]
    pub total_loss_limit: f64,
    #[pyo3(get, set)]
    pub commission: Option<PyCommissionModel>,
}

#[pymethods]
impl BacktestConfig {
    #[new]
    #[pyo3(signature = (
        fill_mode = "market_open",
        stop_loss_pct = 0.0,
        take_profit_pct = 0.0,
        trailing_stop_pct = 0.0,
        slippage_bps = 0.0,
        initial_capital = 100_000.0,
        commission_per_trade = 0.0,
        max_hold_bars = 0,
        slippage_pct_range = 0.0,
        breakeven_pct = 0.0,
        periods_per_year = 252.0,
        margin_ratio = 0.0,
        margin_call_pct = 0.5,
        daily_loss_limit = 0.0,
        total_loss_limit = 0.0,
        commission = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fill_mode: &str,
        stop_loss_pct: f64,
        take_profit_pct: f64,
        trailing_stop_pct: f64,
        slippage_bps: f64,
        initial_capital: f64,
        commission_per_trade: f64,
        max_hold_bars: usize,
        slippage_pct_range: f64,
        breakeven_pct: f64,
        periods_per_year: f64,
        margin_ratio: f64,
        margin_call_pct: f64,
        daily_loss_limit: f64,
        total_loss_limit: f64,
        commission: Option<PyCommissionModel>,
    ) -> Self {
        BacktestConfig {
            fill_mode: fill_mode.to_string(),
            stop_loss_pct,
            take_profit_pct,
            trailing_stop_pct,
            slippage_bps,
            initial_capital,
            commission_per_trade,
            max_hold_bars,
            slippage_pct_range,
            breakeven_pct,
            periods_per_year,
            margin_ratio,
            margin_call_pct,
            daily_loss_limit,
            total_loss_limit,
            commission,
        }
    }
}

// ---------------------------------------------------------------------------
// Signal generators
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14, oversold = 30.0, overbought = 70.0))]
pub fn rsi_threshold_signals<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    oversold: f64,
    overbought: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let out = core_bt::rsi_threshold_signals(prices, timeperiod, oversold, overbought);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, fast = 10, slow = 30))]
pub fn sma_crossover_signals<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fast: usize,
    slow: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fast, "fast", 1)?;
    validation::validate_timeperiod(slow, "slow", 1)?;
    let prices = close.as_slice()?;
    let out = core_bt::sma_crossover_signals(prices, fast, slow).map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, slowperiod = 26, signalperiod = 9))]
pub fn macd_crossover_signals<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    validation::validate_timeperiod(signalperiod, "signalperiod", 1)?;
    let prices = close.as_slice()?;
    let out = core_bt::macd_crossover_signals(prices, fastperiod, slowperiod, signalperiod)
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Backtest core (close-only)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    close, signals,
    commission = None,
    slippage_bps = 0.0,
    initial_capital = 100_000.0,
    commission_per_trade = 0.0,
))]
#[allow(clippy::type_complexity)]
pub fn backtest_core<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    signals: PyReadonlyArray1<'py, f64>,
    commission: Option<PyRef<'py, PyCommissionModel>>,
    slippage_bps: f64,
    initial_capital: f64,
    commission_per_trade: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let c = close.as_slice()?;
    let s = signals.as_slice()?;
    validation::validate_equal_length(&[(c.len(), "close"), (s.len(), "signals")])?;

    let cm = commission.as_ref().map(|c| &c.inner);
    let result = core_bt::backtest_core(
        c,
        s,
        cm,
        slippage_bps,
        initial_capital,
        commission_per_trade,
    )
    .map_err(PyValueError::new_err)?;

    Ok((
        result.positions.into_pyarray(py),
        result.bar_returns.into_pyarray(py),
        result.strategy_returns.into_pyarray(py),
        result.equity.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// OHLCV backtest
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    open, high, low, close, signals,
    fill_mode = "market_open",
    stop_loss_pct = 0.0,
    take_profit_pct = 0.0,
    trailing_stop_pct = 0.0,
    commission = None,
    slippage_bps = 0.0,
    initial_capital = 100_000.0,
    commission_per_trade = 0.0,
    limit_prices = None,
    max_hold_bars = 0,
    slippage_pct_range = 0.0,
    breakeven_pct = 0.0,
    periods_per_year = 252.0,
    margin_ratio = 0.0,
    margin_call_pct = 0.5,
    daily_loss_limit = 0.0,
    total_loss_limit = 0.0,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn backtest_ohlcv_core<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    signals: PyReadonlyArray1<'py, f64>,
    fill_mode: &str,
    stop_loss_pct: f64,
    take_profit_pct: f64,
    trailing_stop_pct: f64,
    commission: Option<PyRef<'py, PyCommissionModel>>,
    slippage_bps: f64,
    initial_capital: f64,
    commission_per_trade: f64,
    limit_prices: Option<PyReadonlyArray1<'py, f64>>,
    max_hold_bars: usize,
    slippage_pct_range: f64,
    breakeven_pct: f64,
    periods_per_year: f64,
    margin_ratio: f64,
    margin_call_pct: f64,
    daily_loss_limit: f64,
    total_loss_limit: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let s = signals.as_slice()?;
    let n = c.len();

    validation::validate_equal_length(&[
        (n, "close"),
        (o.len(), "open"),
        (h.len(), "high"),
        (l.len(), "low"),
        (s.len(), "signals"),
    ])?;

    let config = core_bt::BacktestConfig {
        fill_mode: fill_mode.to_string(),
        stop_loss_pct,
        take_profit_pct,
        trailing_stop_pct,
        slippage_bps,
        initial_capital,
        commission_per_trade,
        max_hold_bars,
        slippage_pct_range,
        breakeven_pct,
        periods_per_year,
        margin_ratio,
        margin_call_pct,
        daily_loss_limit,
        total_loss_limit,
        commission: commission.as_ref().map(|c| c.inner.clone()),
    };

    let lp_opt: Option<&[f64]> = limit_prices.as_ref().and_then(|lp| lp.as_slice().ok());

    let result = core_bt::backtest_ohlcv_core(o, h, l, c, s, &config, lp_opt)
        .map_err(PyValueError::new_err)?;

    Ok((
        result.positions.into_pyarray(py),
        result.fill_prices.into_pyarray(py),
        result.bar_returns.into_pyarray(py),
        result.strategy_returns.into_pyarray(py),
        result.equity.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// Performance metrics
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (strategy_returns, equity, periods_per_year = 252.0, risk_free_rate = 0.0, benchmark_returns = None))]
pub fn compute_performance_metrics<'py>(
    py: Python<'py>,
    strategy_returns: PyReadonlyArray1<'py, f64>,
    equity: PyReadonlyArray1<'py, f64>,
    periods_per_year: f64,
    risk_free_rate: f64,
    benchmark_returns: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = strategy_returns.as_slice()?;
    let eq = equity.as_slice()?;
    let br = benchmark_returns.as_ref().and_then(|b| b.as_slice().ok());

    let metrics = core_bt::compute_performance_metrics(r, eq, periods_per_year, risk_free_rate, br)
        .map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("total_return", metrics.total_return)?;
    dict.set_item("cagr", metrics.cagr)?;
    dict.set_item("annualized_vol", metrics.annualized_vol)?;
    dict.set_item("sharpe", metrics.sharpe)?;
    dict.set_item("sortino", metrics.sortino)?;
    dict.set_item("calmar", metrics.calmar)?;
    dict.set_item("max_drawdown", metrics.max_drawdown)?;
    dict.set_item("avg_drawdown", metrics.avg_drawdown)?;
    dict.set_item(
        "max_drawdown_duration_bars",
        metrics.max_drawdown_duration_bars as i64,
    )?;
    dict.set_item(
        "avg_drawdown_duration_bars",
        metrics.avg_drawdown_duration_bars,
    )?;
    dict.set_item("ulcer_index", metrics.ulcer_index)?;
    dict.set_item("omega_ratio", metrics.omega_ratio)?;
    dict.set_item("win_rate", metrics.win_rate)?;
    dict.set_item("profit_factor", metrics.profit_factor)?;
    dict.set_item("r_expectancy", metrics.r_expectancy)?;
    dict.set_item("avg_win", metrics.avg_win)?;
    dict.set_item("avg_loss", metrics.avg_loss)?;
    dict.set_item("tail_ratio", metrics.tail_ratio)?;
    dict.set_item("skewness", metrics.skewness)?;
    dict.set_item("kurtosis", metrics.kurtosis)?;
    dict.set_item("best_bar", metrics.best_bar)?;
    dict.set_item("worst_bar", metrics.worst_bar)?;
    dict.set_item("n_trades", metrics.n_trades as i64)?;
    dict.set_item("n_position_changes", metrics.n_position_changes as i64)?;

    if let Some(v) = metrics.benchmark_total_return {
        dict.set_item("benchmark_total_return", v)?;
    }
    if let Some(v) = metrics.benchmark_cagr {
        dict.set_item("benchmark_cagr", v)?;
    }
    if let Some(v) = metrics.benchmark_annualized_vol {
        dict.set_item("benchmark_annualized_vol", v)?;
    }
    if let Some(v) = metrics.benchmark_sharpe {
        dict.set_item("benchmark_sharpe", v)?;
    }
    if let Some(v) = metrics.alpha {
        dict.set_item("alpha", v)?;
    }
    if let Some(v) = metrics.beta {
        dict.set_item("beta", v)?;
    }
    if let Some(v) = metrics.tracking_error {
        dict.set_item("tracking_error", v)?;
    }
    if let Some(v) = metrics.information_ratio {
        dict.set_item("information_ratio", v)?;
    }

    Ok(dict)
}

// ---------------------------------------------------------------------------
// Trade extraction
// ---------------------------------------------------------------------------

#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn extract_trades_ohlcv<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<'py, f64>,
    fill_prices: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let pos = positions.as_slice()?;
    let fp = fill_prices.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;

    validation::validate_equal_length(&[
        (pos.len(), "positions"),
        (fp.len(), "fill_prices"),
        (h.len(), "high"),
        (l.len(), "low"),
    ])?;

    let trades = core_bt::extract_trades_ohlcv(pos, fp, h, l).map_err(PyValueError::new_err)?;

    let mut entry_bars: Vec<i64> = Vec::with_capacity(trades.len());
    let mut exit_bars: Vec<i64> = Vec::with_capacity(trades.len());
    let mut directions: Vec<f64> = Vec::with_capacity(trades.len());
    let mut entry_prices: Vec<f64> = Vec::with_capacity(trades.len());
    let mut exit_prices: Vec<f64> = Vec::with_capacity(trades.len());
    let mut pnl_pcts: Vec<f64> = Vec::with_capacity(trades.len());
    let mut duration_bars_vec: Vec<i64> = Vec::with_capacity(trades.len());
    let mut maes: Vec<f64> = Vec::with_capacity(trades.len());
    let mut mfes: Vec<f64> = Vec::with_capacity(trades.len());

    for t in &trades {
        entry_bars.push(t.entry_bar);
        exit_bars.push(t.exit_bar);
        directions.push(t.direction);
        entry_prices.push(t.entry_price);
        exit_prices.push(t.exit_price);
        pnl_pcts.push(t.pnl_pct);
        duration_bars_vec.push(t.duration_bars);
        maes.push(t.mae);
        mfes.push(t.mfe);
    }

    Ok((
        entry_bars.into_pyarray(py),
        exit_bars.into_pyarray(py),
        directions.into_pyarray(py),
        entry_prices.into_pyarray(py),
        exit_prices.into_pyarray(py),
        pnl_pcts.into_pyarray(py),
        duration_bars_vec.into_pyarray(py),
        maes.into_pyarray(py),
        mfes.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// Multi-asset backtest
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    close_2d, weights_2d,
    commission_per_trade = 0.0,
    slippage_bps = 0.0,
    parallel = true,
    max_asset_weight = 1.0,
    max_gross_exposure = 0.0,
    max_net_exposure = 0.0,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn backtest_multi_asset_core<'py>(
    py: Python<'py>,
    close_2d: PyReadonlyArray2<'py, f64>,
    weights_2d: PyReadonlyArray2<'py, f64>,
    commission_per_trade: f64,
    slippage_bps: f64,
    parallel: bool,
    max_asset_weight: f64,
    max_gross_exposure: f64,
    max_net_exposure: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let c_arr = close_2d.as_array();
    let w_arr = weights_2d.as_array();
    let (n_bars, n_assets) = c_arr.dim();

    if w_arr.dim() != (n_bars, n_assets) {
        return Err(PyValueError::new_err(format!(
            "weights_2d shape {:?} must match close_2d shape {:?}",
            w_arr.dim(),
            c_arr.dim()
        )));
    }

    // Transpose to (n_assets, n_bars) for the core function
    let mut close_cm: Vec<Vec<f64>> = vec![vec![0.0; n_bars]; n_assets];
    let mut weights_cm: Vec<Vec<f64>> = vec![vec![0.0; n_bars]; n_assets];
    for j in 0..n_assets {
        for i in 0..n_bars {
            close_cm[j][i] = c_arr[[i, j]];
            weights_cm[j][i] = w_arr[[i, j]];
        }
    }

    // For parallel execution, use rayon directly on the core's single_asset_backtest.
    // Apply portfolio constraints first via the core function's logic.

    // Apply constraints
    #[allow(clippy::needless_range_loop)]
    if max_asset_weight != 1.0 || max_gross_exposure > 0.0 || max_net_exposure > 0.0 {
        for i in 0..n_bars {
            if max_asset_weight < f64::INFINITY && max_asset_weight > 0.0 {
                for j in 0..n_assets {
                    let w = weights_cm[j][i];
                    if w.abs() > max_asset_weight {
                        weights_cm[j][i] = w.signum() * max_asset_weight;
                    }
                }
            }
            if max_gross_exposure > 0.0 {
                let gross: f64 = (0..n_assets).map(|j| weights_cm[j][i].abs()).sum();
                if gross > max_gross_exposure {
                    let scale = max_gross_exposure / gross;
                    for j in 0..n_assets {
                        weights_cm[j][i] *= scale;
                    }
                }
            }
            if max_net_exposure > 0.0 {
                let net: f64 = (0..n_assets).map(|j| weights_cm[j][i]).sum();
                if net.abs() > max_net_exposure {
                    let excess = net - net.signum() * max_net_exposure;
                    let adj_per_asset = excess / n_assets as f64;
                    for j in 0..n_assets {
                        weights_cm[j][i] -= adj_per_asset;
                    }
                }
            }
        }
    }

    // Run per-asset backtests (parallel or serial)
    let asset_strategy_returns: Vec<Vec<f64>> = py.allow_threads(|| {
        let run_asset = |j: usize| -> Vec<f64> {
            let (_, strat_rets, _) = core_bt::single_asset_backtest(
                &close_cm[j],
                &weights_cm[j],
                commission_per_trade,
                slippage_bps,
            );
            strat_rets
        };

        if parallel {
            (0..n_assets).into_par_iter().map(run_asset).collect()
        } else {
            (0..n_assets).map(run_asset).collect()
        }
    });

    // Assemble asset_returns 2D array (n_bars, n_assets)
    let mut asset_ret_arr = Array2::<f64>::zeros((n_bars, n_assets));
    for j in 0..n_assets {
        for i in 0..n_bars {
            asset_ret_arr[[i, j]] = asset_strategy_returns[j][i];
        }
    }

    // Portfolio returns
    let mut portfolio_returns = vec![0.0_f64; n_bars];
    for i in 0..n_bars {
        let mut s = 0.0_f64;
        for j in 0..n_assets {
            s += asset_ret_arr[[i, j]];
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

    Ok((
        asset_ret_arr.into_pyarray(py),
        portfolio_returns.into_pyarray(py),
        portfolio_equity.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// Monte Carlo bootstrap
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (strategy_returns, n_sims = 1000, seed = 42, block_size = 1))]
pub fn monte_carlo_bootstrap<'py>(
    py: Python<'py>,
    strategy_returns: PyReadonlyArray1<'py, f64>,
    n_sims: usize,
    seed: u64,
    block_size: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let r = strategy_returns.as_slice()?;
    let n = r.len();

    // Use rayon for parallel Monte Carlo (preserving the original parallel behavior)
    if n < 2 {
        return Err(PyValueError::new_err(
            "strategy_returns must have at least 2 elements",
        ));
    }
    if n_sims == 0 {
        return Err(PyValueError::new_err("n_sims must be >= 1"));
    }
    let bsize = block_size.max(1).min(n);

    let mut result = Array2::<f64>::zeros((n_sims, n));

    py.allow_threads(|| {
        result
            .as_slice_mut()
            .unwrap()
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(sim_idx, row)| {
                let mut state = seed
                    .wrapping_mul(6_364_136_223_846_793_005_u64)
                    .wrapping_add((sim_idx as u64).wrapping_mul(2_862_933_555_777_941_757_u64));
                core_bt::lcg_next(&mut state);
                core_bt::lcg_next(&mut state);

                if bsize == 1 {
                    for dst in row.iter_mut() {
                        *dst = r[core_bt::lcg_index(&mut state, n)];
                    }
                } else {
                    let mut filled = 0_usize;
                    while filled < n {
                        let start = core_bt::lcg_index(&mut state, n);
                        let take = bsize.min(n - filled);
                        for k in 0..take {
                            row[filled + k] = r[(start + k) % n];
                        }
                        filled += take;
                    }
                }

                let mut cum = 1.0_f64;
                for elem in row.iter_mut().take(n) {
                    cum *= 1.0 + *elem;
                    *elem = cum;
                }
            });
    });

    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Walk-forward indices
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (n_bars, train_bars, test_bars, anchored = false, step_bars = 0))]
pub fn walk_forward_indices<'py>(
    py: Python<'py>,
    n_bars: usize,
    train_bars: usize,
    test_bars: usize,
    anchored: bool,
    step_bars: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let folds = core_bt::walk_forward_indices(n_bars, train_bars, test_bars, anchored, step_bars)
        .map_err(PyValueError::new_err)?;

    let n_folds = folds.len();
    let mut arr = Array2::<i64>::zeros((n_folds, 4));
    for (i, fold) in folds.iter().enumerate() {
        for j in 0..4 {
            arr[[i, j]] = fold[j];
        }
    }

    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Kelly criterion
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn kelly_fraction(win_rate: f64, avg_win: f64, avg_loss: f64) -> PyResult<f64> {
    core_bt::kelly_fraction(win_rate, avg_win, avg_loss).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn half_kelly_fraction(win_rate: f64, avg_win: f64, avg_loss: f64) -> PyResult<f64> {
    core_bt::half_kelly_fraction(win_rate, avg_win, avg_loss).map_err(PyValueError::new_err)
}

// ---------------------------------------------------------------------------
// StreamingBacktest
// ---------------------------------------------------------------------------

#[pyclass(name = "StreamingBacktest")]
pub struct StreamingBacktest {
    inner: core_bt::StreamingBacktest,
}

#[pymethods]
impl StreamingBacktest {
    #[new]
    #[pyo3(signature = (commission_per_trade=0.0, slippage_bps=0.0))]
    pub fn new(commission_per_trade: f64, slippage_bps: f64) -> Self {
        StreamingBacktest {
            inner: core_bt::StreamingBacktest::new(commission_per_trade, slippage_bps),
        }
    }

    pub fn on_bar<'py>(
        &mut self,
        py: Python<'py>,
        close: f64,
        signal: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.on_bar(close, signal);
        let d = PyDict::new(py);
        d.set_item("position", result.position)?;
        d.set_item("bar_return", result.bar_return)?;
        d.set_item("equity", result.equity)?;
        d.set_item("n_trades", result.n_trades)?;
        Ok(d)
    }

    #[getter]
    pub fn equity(&self) -> f64 {
        self.inner.equity
    }

    #[getter]
    pub fn position(&self) -> f64 {
        self.inner.position
    }

    #[getter]
    pub fn n_trades(&self) -> usize {
        self.inner.n_trades
    }

    pub fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.summary();
        let d = PyDict::new(py);
        d.set_item("equity", s.equity)?;
        d.set_item("n_trades", s.n_trades)?;
        d.set_item("total_commission", s.total_commission)?;
        d.set_item("win_rate", s.win_rate)?;
        d.set_item("avg_win", s.avg_win)?;
        d.set_item("avg_loss", s.avg_loss)?;
        d.set_item("kelly_fraction", s.kelly_fraction)?;
        Ok(d)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rsi_threshold_signals, m)?)?;
    m.add_function(wrap_pyfunction!(sma_crossover_signals, m)?)?;
    m.add_function(wrap_pyfunction!(macd_crossover_signals, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_core, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_ohlcv_core, m)?)?;
    m.add_function(wrap_pyfunction!(compute_performance_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(extract_trades_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_multi_asset_core, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(walk_forward_indices, m)?)?;
    m.add_function(wrap_pyfunction!(kelly_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(half_kelly_fraction, m)?)?;
    m.add_class::<BacktestConfig>()?;
    m.add_class::<StreamingBacktest>()?;
    m.add_class::<PyCommissionModel>()?;
    m.add_class::<PyCurrency>()?;
    Ok(())
}
