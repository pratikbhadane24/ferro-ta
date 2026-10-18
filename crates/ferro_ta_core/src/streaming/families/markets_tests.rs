//! Native market streams against their batch kernels on inputs the registry
//! fixture never produces (NaN, flat runs, many parameter sets).
use super::*;
use crate::streaming::fixtures::input_column;

const N: usize = 600;

/// `series` with a NaN, a flat run and a spike, so every CUSUM skip rule fires.
fn awkward_series() -> Vec<f64> {
    let mut series = input_column("series", N);
    series[70] = f64::NAN;
    for v in &mut series[120..150] {
        *v = 100.0;
    }
    series[200] += 40.0;
    series
}

#[test]
fn cusum_matches_batch_with_nan_and_flat_runs_across_params() {
    let series = awkward_series();
    for (window, threshold, slack) in [(2, 3.0, 0.5), (5, 1.0, 0.0), (20, 3.0, 0.5), (40, 8.0, 1.0)]
    {
        let want = crate::regime::detect_breaks_cusum(&series, window, threshold, slack);
        let mut s = StreamingDetectBreaksCusum::new(window, threshold, slack).unwrap();
        let got: Vec<i8> = series.iter().map(|&x| s.update(x)).collect();
        assert_eq!(
            got, want,
            "window={window} threshold={threshold} slack={slack}"
        );
        assert!(want.contains(&1), "no break to compare for window={window}");
    }
}

#[test]
fn cusum_rejects_invalid_params() {
    assert!(StreamingDetectBreaksCusum::new(1, 3.0, 0.5).is_err());
    assert!(StreamingDetectBreaksCusum::new(5, f64::NAN, 0.5).is_err());
    assert!(StreamingDetectBreaksCusum::new(5, 3.0, f64::INFINITY).is_err());
}

#[test]
fn rsi_threshold_signals_match_batch_for_several_bands() {
    let close = input_column("close", N);
    for (period, oversold, overbought) in [
        (14, 30.0, 70.0),
        (2, 10.0, 90.0),
        (5, 50.0, 50.0),
        (30, 45.0, 55.0),
    ] {
        let want = crate::backtest::rsi_threshold_signals(&close, period, oversold, overbought);
        let mut s = StreamingRsiThresholdSignals::new(period, oversold, overbought).unwrap();
        for (i, (&x, &w)) in close.iter().zip(&want).enumerate() {
            let g = s.update(x);
            assert!(
                g == w || (g.is_nan() && w.is_nan()),
                "p={period} bar {i}: {g} != {w}"
            );
        }
        assert!(want.contains(&1.0) && want.contains(&-1.0) || oversold == overbought);
    }
}

#[test]
fn relative_strength_and_funding_match_batch() {
    let asset = input_column("asset_returns", N);
    let bench = input_column("benchmark_returns", N);
    let want = crate::portfolio::relative_strength(&asset, &bench);
    let mut rs = StreamingRelativeStrength::new().unwrap();
    for (i, w) in want.iter().enumerate() {
        assert_eq!(rs.update(asset[i], bench[i]), *w, "bar {i}");
    }
    // A benchmark return of -100% zeroes the product: NaN, as in batch.
    let mut rs = StreamingRelativeStrength::new().unwrap();
    assert!(rs.update(0.01, -1.0).is_nan());

    let pos = input_column("position_size", N);
    let rate = input_column("funding_rate", N);
    let want = crate::crypto::funding_cumulative_pnl(&pos, &rate);
    let mut f = StreamingFundingCumulativePnl::new().unwrap();
    for (i, w) in want.iter().enumerate() {
        assert_eq!(f.update(pos[i], rate[i]), *w, "bar {i}");
    }
}
