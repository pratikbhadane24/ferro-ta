//! C exports for `ferro_ta_core::statistic`.

crate::ffi_exports! {
    group: "statistic",
    /// Compute the rolling population standard deviation, scaled by `nbdev`.
    ft_stddev = statistic::stddev(real)[timeperiod: period, nbdev: float]
        -> [out: f64];
    /// Rolling population variance, scaled by `nbdev**2`.
    ft_var = statistic::var(real)[timeperiod: period, nbdev: float]
        -> [out: f64];
    /// Linear regression fitted value at the last point of the window.
    ft_linearreg = statistic::linearreg(close)[timeperiod: period]
        -> [out: f64];
    /// Slope of the rolling linear regression line.
    ft_linearreg_slope = statistic::linearreg_slope(close)[timeperiod: period]
        -> [out: f64];
    /// Intercept of the rolling linear regression line.
    ft_linearreg_intercept = statistic::linearreg_intercept(close)[timeperiod: period]
        -> [out: f64];
    /// Angle of the regression line in degrees.
    ft_linearreg_angle = statistic::linearreg_angle(close)[timeperiod: period]
        -> [out: f64];
    /// Time Series Forecast: linear regression extrapolated one period ahead.
    ft_tsf = statistic::tsf(close)[timeperiod: period]
        -> [out: f64];
    /// Rolling beta: regression of real1 daily returns on real0 daily returns.
    ft_beta = statistic::beta(real0, real1)[timeperiod: period]
        -> [out: f64];
    /// Rolling Pearson correlation coefficient between two series.
    ft_correl = statistic::correl(real0, real1)[timeperiod: period]
        -> [out: f64];
}
