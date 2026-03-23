pub mod aggregation;
pub mod alerts;
pub mod attribution;
pub mod batch;
pub mod chunked;
pub mod crypto;
pub mod cycle;
pub mod extended;
pub mod futures;
pub mod math_ops;
pub mod momentum;
pub mod options;
pub mod overlap;
pub mod pattern;
pub mod portfolio;
pub mod price_transform;
pub mod regime;
pub mod resampling;
pub mod signals;
pub mod statistic;
pub mod streaming;
pub mod validation;
pub mod volatility;
pub mod volume;

use pyo3::prelude::*;

/// ferro_ta — A fast Technical Analysis library powered by Rust.
///
/// Indicators are organized into modules matching the TA-Lib category structure:
/// - **overlap**        : Overlap Studies (SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MACD, BBANDS, SAR, MA, MAVP, MAMA, SAREXT, MACDEXT, …)
/// - **momentum**       : Momentum Indicators (RSI, STOCH, ADX, CCI, WILLR, AROON, MFI, …)
/// - **volume**         : Volume Indicators (AD, ADOSC, OBV)
/// - **volatility**     : Volatility Indicators (ATR, NATR, TRANGE)
/// - **statistic**      : Statistic Functions (STDDEV, VAR, LINEARREG, BETA, CORREL, …)
/// - **price_transform**: Price Transformations (AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE)
/// - **pattern**        : Pattern Recognition (CDLDOJI, CDLENGULFING, CDLHAMMER, …)
/// - **cycle**          : Cycle Indicators (HT_TRENDLINE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE)
/// - **batch**          : Batch Execution (batch_sma, batch_ema, batch_rsi — 2-D array input)
/// - **streaming**      : Streaming Indicators (StreamingSMA, StreamingEMA, … — bar-by-bar PyO3 classes)
/// - **extended**       : Extended Indicators (VWAP, SUPERTREND, DONCHIAN, ICHIMOKU, …)
/// - **math_ops**       : Rolling Math Operators (rolling_sum, rolling_max, rolling_min, …)
/// - **resampling**     : OHLCV resampling helpers (volume_bars, ohlcv_agg)
/// - **aggregation**    : Tick/trade aggregation pipeline (aggregate_tick_bars, aggregate_volume_bars_ticks, aggregate_time_bars)
/// - **portfolio**      : Portfolio analytics (portfolio_volatility, beta_full, rolling_beta, drawdown_series, correlation_matrix, relative_strength, spread, zscore_series, compose_weighted)
/// - **signals**        : Signal helpers (rank_series, top_n_indices, bottom_n_indices)
#[pymodule]
fn _ferro_ta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    overlap::register(m)?;
    momentum::register(m)?;
    volume::register(m)?;
    volatility::register(m)?;
    statistic::register(m)?;
    price_transform::register(m)?;
    pattern::register(m)?;
    cycle::register(m)?;
    batch::register(m)?;
    streaming::register(m)?;
    extended::register(m)?;
    math_ops::register(m)?;
    options::register(m)?;
    futures::register(m)?;
    resampling::register(m)?;
    aggregation::register(m)?;
    portfolio::register(m)?;
    signals::register(m)?;
    alerts::register(m)?;
    crypto::register(m)?;
    chunked::register(m)?;
    regime::register(m)?;
    attribution::register(m)?;
    Ok(())
}
