//! Statistic and options-volatility streams: rolling moments, regression,
//! beta / correlation, IV rank / percentile / z-score and the realized-vol
//! estimators. All are windowed replays of their batch kernels.

/// Registry entries for the statistic and options-volatility streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_stats {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                stddev => StreamingStddev, doc "Streaming rolling population standard deviation over `timeperiod` bars, scaled by `nbdev`. Recomputes each window exactly; on long series the batch function accumulates rounding drift, so values can differ beyond 1e-9 (the stream is the exact one).",
                    params(timeperiod: period = 5, nbdev: float = 1.0), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::statistic::stddev(real, timeperiod, nbdev));
                var => StreamingVar, doc "Streaming rolling population variance over `timeperiod` bars, scaled by `nbdev**2`. Recomputes each window exactly; on long series the batch function accumulates rounding drift, so values can differ beyond 1e-9 (the stream is the exact one).",
                    params(timeperiod: period = 5, nbdev: float = 1.0), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::statistic::var(real, timeperiod, nbdev));
                linearreg => StreamingLinearreg, doc "Streaming linear regression fitted value at the last bar of the window.",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::statistic::linearreg(close, timeperiod));
                linearreg_slope => StreamingLinearregSlope, doc "Streaming slope of the rolling linear regression line. Recomputes each window exactly; on long series the batch function accumulates rounding drift, so values can differ beyond 1e-9 (the stream is the exact one).",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::statistic::linearreg_slope(close, timeperiod));
                linearreg_intercept => StreamingLinearregIntercept, doc "Streaming intercept of the rolling linear regression line.",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::statistic::linearreg_intercept(close, timeperiod));
                linearreg_angle => StreamingLinearregAngle, doc "Streaming angle of the rolling regression line in degrees. Recomputes each window exactly; on long series the batch function accumulates rounding drift, so values can differ beyond 1e-9 (the stream is the exact one).",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::statistic::linearreg_angle(close, timeperiod));
                tsf => StreamingTsf, doc "Streaming time series forecast: the regression line extrapolated one bar ahead.",
                    params(timeperiod: period = 14), inputs(close), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::statistic::tsf(close, timeperiod));
                beta => StreamingBeta, doc "Streaming rolling beta of `real1` returns on `real0` returns over `timeperiod` returns.",
                    params(timeperiod: period = 5), inputs(real0, real1), outputs(out: f64),
                    windowed(lookback = timeperiod + 1, batch = $crate::statistic::beta(real0, real1, timeperiod));
                correl => StreamingCorrel, doc "Streaming rolling Pearson correlation of two series over `timeperiod` bars. Recomputes each window exactly; on long series the batch function accumulates rounding drift, so values can differ beyond 1e-9 (the stream is the exact one).",
                    params(timeperiod: period = 30), inputs(real0, real1), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::statistic::correl(real0, real1, timeperiod));
                iv_rank => StreamingIvRank, doc "Streaming IV rank: position of the latest IV between the window's min and max.",
                    params(window: period = 252), inputs(iv_series), outputs(out: f64),
                    windowed(lookback = window, batch = $crate::options::iv::iv_rank(iv_series, window));
                iv_percentile => StreamingIvPercentile, doc "Streaming IV percentile: share of window values at or below the latest IV.",
                    params(window: period = 252), inputs(iv_series), outputs(out: f64),
                    windowed(lookback = window, batch = $crate::options::iv::iv_percentile(iv_series, window));
                iv_zscore => StreamingIvZscore, doc "Streaming IV z-score of the latest IV against the window mean and deviation.",
                    params(window: period = 252), inputs(iv_series), outputs(out: f64),
                    windowed(lookback = window, batch = $crate::options::iv::iv_zscore(iv_series, window));
                close_to_close_vol => StreamingCloseToCloseVol, doc "Streaming annualised close-to-close realized volatility (first value after `window` returns).",
                    params(window: period = 20, trading_days: float = 252.0), inputs(close), outputs(out: f64),
                    windowed(lookback = window + 1, batch = $crate::options::realized_vol::close_to_close_vol(close, window, trading_days));
                parkinson_vol => StreamingParkinsonVol, doc "Streaming annualised Parkinson high-low realized volatility.",
                    params(window: period = 20, trading_days: float = 252.0), inputs(high, low), outputs(out: f64),
                    windowed(lookback = window, batch = $crate::options::realized_vol::parkinson_vol(high, low, window, trading_days));
                garman_klass_vol => StreamingGarmanKlassVol, doc "Streaming annualised Garman-Klass OHLC realized volatility.",
                    params(window: period = 20, trading_days: float = 252.0), inputs(open, high, low, close), outputs(out: f64),
                    windowed(lookback = window, batch = $crate::options::realized_vol::garman_klass_vol(open, high, low, close, window, trading_days));
                rogers_satchell_vol => StreamingRogersSatchellVol, doc "Streaming annualised Rogers-Satchell OHLC realized volatility.",
                    params(window: period = 20, trading_days: float = 252.0), inputs(open, high, low, close), outputs(out: f64),
                    windowed(lookback = window, batch = $crate::options::realized_vol::rogers_satchell_vol(open, high, low, close, window, trading_days));
                yang_zhang_vol => StreamingYangZhangVol, doc "Streaming annualised Yang-Zhang OHLC realized volatility (first value after `window` bars plus the prior close).",
                    params(window: period = 20, trading_days: float = 252.0), inputs(open, high, low, close), outputs(out: f64),
                    windowed(lookback = window + 1, batch = $crate::options::realized_vol::yang_zhang_vol(open, high, low, close, window, trading_days));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_stats! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}
