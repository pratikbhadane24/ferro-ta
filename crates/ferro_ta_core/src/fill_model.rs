//! Execution-realism helpers for the OHLCV backtest engine.
//!
//! - **ATR slippage**: slippage proportional to recent volatility instead of
//!   a fixed number of basis points.
//! - **Volume participation cap**: a signal-driven position change may trade
//!   at most `max_participation × bar volume` shares; the remainder carries
//!   to later bars as a partial fill.

/// Per-bar slippage fraction for bar `i`.
///
/// Precedence: ATR (`atr_mult × ATR[i-1] / close[i-1]`, previous bar so there
/// is no look-ahead) when enabled and warmed up, then the high-low range
/// model, then the flat `default_slip`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bar_slippage(
    i: usize,
    atr: Option<&[f64]>,
    atr_mult: f64,
    range_pct: f64,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    default_slip: f64,
) -> f64 {
    if let Some(a) = atr {
        let prev_atr = a[i - 1];
        if atr_mult > 0.0 && prev_atr.is_finite() && close[i - 1] > 0.0 {
            return atr_mult * prev_atr / close[i - 1];
        }
    }
    if range_pct > 0.0 && close[i] > 0.0 {
        return range_pct * (high[i] - low[i]) / close[i];
    }
    default_slip
}

/// Largest position-weight change fillable on a bar.
///
/// Notional of a weight change is approximated as `|Δw| × initial_capital`
/// (the same convention commission uses), so the cap in weight units is
/// `max_participation × volume × price / initial_capital`. Missing or
/// non-positive volume means nothing can trade.
pub(crate) fn participation_cap(
    max_participation: f64,
    volume: f64,
    price: f64,
    initial_capital: f64,
) -> f64 {
    if !(volume.is_finite() && volume > 0.0 && price > 0.0 && initial_capital > 0.0) {
        return 0.0;
    }
    max_participation * volume * price / initial_capital
}

/// Move from `current` toward `target` by at most `cap` (weight units).
pub(crate) fn step_toward(current: f64, target: f64, cap: f64) -> f64 {
    let delta = target - current;
    if delta.abs() <= cap {
        target
    } else {
        current + delta.signum() * cap
    }
}

/// True when both weights are non-zero and on the same side.
pub(crate) fn same_side(a: f64, b: f64) -> bool {
    a != 0.0 && b != 0.0 && a.signum() == b.signum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::{backtest_ohlcv_core, backtest_ohlcv_volume_core, BacktestConfig};
    use crate::commission::CommissionModel;

    struct Bars {
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
    }

    /// Flat market with a fixed 2.0 high-low range: ATR converges to 2.0.
    fn flat_bars(n: usize, price: f64) -> Bars {
        Bars {
            open: vec![price; n],
            high: vec![price + 1.0; n],
            low: vec![price - 1.0; n],
            close: vec![price; n],
        }
    }

    fn run(
        b: &Bars,
        volume: Option<&[f64]>,
        signals: &[f64],
        cfg: &BacktestConfig,
    ) -> crate::backtest::OhlcvBacktestResult {
        backtest_ohlcv_volume_core(
            &b.open, &b.high, &b.low, &b.close, volume, signals, cfg, None,
        )
        .unwrap()
    }

    #[test]
    fn helpers() {
        assert_eq!(step_toward(0.0, 1.0, 0.25), 0.25);
        assert_eq!(step_toward(0.5, -1.0, 0.25), 0.25);
        assert_eq!(step_toward(0.9, 1.0, 0.25), 1.0);
        assert_eq!(participation_cap(0.1, 1000.0, 50.0, 100_000.0), 0.05);
        assert_eq!(participation_cap(0.1, f64::NAN, 50.0, 100_000.0), 0.0);
        assert!(same_side(0.5, 1.0) && !same_side(-0.5, 1.0) && !same_side(0.0, 1.0));
    }

    /// Golden outputs captured from the engine *before* partial fills and
    /// ATR slippage existed (stops, TP, trailing, breakeven, max-hold and
    /// commission all active). With both new knobs off, results must be
    /// bit-identical.
    #[test]
    fn defaults_reproduce_pre_fill_model_goldens() {
        let n = 40;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.7).sin() * 6.0 + i as f64 * 0.2)
            .collect();
        let open: Vec<f64> = (0..n)
            .map(|i| {
                if i == 0 {
                    close[0]
                } else {
                    close[i - 1] + ((i as f64) * 1.3).cos()
                }
            })
            .collect();
        let high: Vec<f64> = (0..n).map(|i| open[i].max(close[i]) + 1.5).collect();
        let low: Vec<f64> = (0..n).map(|i| open[i].min(close[i]) - 1.5).collect();
        let signals: Vec<f64> = (0..n)
            .map(|i| match i % 9 {
                0..=3 => 1.0,
                4 => f64::NAN,
                5..=7 => -1.0,
                _ => 0.0,
            })
            .collect();
        let cfg = BacktestConfig {
            stop_loss_pct: 0.03,
            take_profit_pct: 0.05,
            trailing_stop_pct: 0.04,
            slippage_bps: 3.0,
            breakeven_pct: 0.02,
            max_hold_bars: 6,
            commission: Some(CommissionModel {
                rate_of_value: 0.0005,
                flat_per_order: 2.0,
                ..Default::default()
            }),
            ..Default::default()
        };
        let r = backtest_ohlcv_core(&open, &high, &low, &close, &signals, &cfg, None).unwrap();
        let expected_pos: Vec<f64> = vec![
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0,
            -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let expected_ret: Vec<f64> = vec![
            0.0,
            0.0370454834321836,
            -0.0062503946218384066,
            0.0010025948341406414,
            -0.02295472024347372,
            -0.0,
            0.029292360425764018,
            0.019987974406987165,
            -0.031037394195433225,
            -0.010021791117723226,
            0.02965086610034194,
            -0.007409181769289598,
            0.0028450406783180238,
            -0.030729357850450525,
            -0.0,
            0.03543561564195653,
            0.01374819269518965,
            -0.03571372544527504,
            -0.005462874504149047,
            0.028851962115196212,
            -0.008532299323998129,
            -0.0018018972070057886,
            -0.03596562147528926,
            -0.0,
            0.03607703209518005,
            0.01309210983496173,
            -0.03284871437097251,
            -0.008251049356892007,
            0.034281947895058104,
            0.013866918724744425,
            -0.009773468430818779,
            -0.031853700941131505,
            -0.0,
            0.03034323726438347,
            -0.0227416294569186,
            -0.024877219823661027,
            -0.016090825326961416,
            0.0414120172958691,
            0.0069232908963386824,
            -0.015761707090527016,
        ];
        let expected_fill: Vec<f64> = vec![
            f64::NAN,
            100.29757907827316,
            103.50019044473353,
            105.61844210555334,
            103.43786195634587,
            f64::NAN,
            98.9195712776086,
            93.97359271372817,
            94.91581677680674,
            98.71244944787901,
            102.83917268341158,
            105.24384902713152,
            107.16697222761161,
            104.31151217727987,
            f64::NAN,
            101.3665211122441,
            96.2981950566319,
            96.30195283853459,
            100.15403095207597,
            104.74097074844734,
            106.98589836880245,
            109.41473943574556,
            105.43396144118482,
            f64::NAN,
            103.2533644927889,
            98.09069626814946,
            98.3901751947803,
            102.32578220257152,
            106.00268190791319,
            111.30281600330885,
            112.05684833978559,
            107.57457440619416,
            f64::NAN,
            104.4526218439383,
            103.44797769855829,
            100.99197117839232,
            105.03165002552801,
            107.07511253858824,
            112.42886816551766,
            114.5092843751644,
        ];
        assert_eq!(r.positions, expected_pos);
        assert_eq!(r.strategy_returns, expected_ret);
        assert!(r
            .fill_prices
            .iter()
            .zip(&expected_fill)
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan())));
    }

    #[test]
    fn atr_slippage_scales_entry_fill() {
        let b = flat_bars(20, 100.0);
        let mut signals = vec![0.0; 20];
        signals[10] = 1.0;
        let cfg = BacktestConfig {
            slippage_atr_mult: 0.5,
            slippage_atr_period: 3,
            slippage_bps: 1_000.0, // must be overridden once ATR is warm
            ..Default::default()
        };
        let res = run(&b, None, &signals, &cfg);
        // ATR of a constant 2.0 range is 2.0 → slip = 0.5 * 2 / 100 = 1%.
        assert!(
            (res.fill_prices[11] - 101.0).abs() < 1e-9,
            "{}",
            res.fill_prices[11]
        );
    }

    #[test]
    fn atr_slippage_falls_back_during_warmup() {
        let b = flat_bars(10, 100.0);
        let mut signals = vec![0.0; 10];
        signals[0] = 1.0;
        let cfg = BacktestConfig {
            slippage_atr_mult: 0.5,
            slippage_atr_period: 5,
            slippage_bps: 100.0,
            ..Default::default()
        };
        let res = run(&b, None, &signals, &cfg);
        assert!((res.fill_prices[1] - 101.0).abs() < 1e-9); // 100 bps fallback
    }

    #[test]
    fn volume_cap_fills_partially_across_bars() {
        let b = flat_bars(8, 100.0);
        // Cap = 0.1 * 250 * 100 / 100_000 = 0.025 ... use bigger volume:
        // 0.1 * 2_500 * 100 / 100_000 = 0.25 weight per bar.
        let vol = vec![2_500.0; 8];
        let mut signals = vec![f64::NAN; 8];
        signals[0] = 1.0; // one signal, then NaN = hold the target
        let cfg = BacktestConfig {
            max_participation: 0.1,
            ..Default::default()
        };
        let res = run(&b, Some(&vol), &signals, &cfg);
        assert_eq!(res.positions[1..6], [0.25, 0.5, 0.75, 1.0, 1.0]);
    }

    #[test]
    fn volume_cap_partial_exit_and_flip() {
        let b = flat_bars(8, 100.0);
        let vol = vec![5_000.0; 8]; // 0.5 weight per bar at 10%
        let signals = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let cfg = BacktestConfig {
            max_participation: 0.1,
            ..Default::default()
        };
        let res = run(&b, Some(&vol), &signals, &cfg);
        assert_eq!(
            res.positions,
            vec![0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5, -1.0]
        );
    }

    #[test]
    fn zero_volume_bar_blocks_trading() {
        let b = flat_bars(5, 100.0);
        let vol = vec![1e9, 0.0, 1e9, 1e9, 1e9];
        let signals = vec![1.0; 5];
        let cfg = BacktestConfig {
            max_participation: 0.1,
            ..Default::default()
        };
        let res = run(&b, Some(&vol), &signals, &cfg);
        assert_eq!(res.positions, vec![0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn scale_in_charges_commission_on_traded_amount_only() {
        let b = flat_bars(6, 100.0);
        let vol = vec![5_000.0; 6];
        let signals = vec![1.0; 6];
        let cfg = BacktestConfig {
            max_participation: 0.1,
            commission: Some(CommissionModel {
                rate_of_value: 0.001,
                ..Default::default()
            }),
            ..Default::default()
        };
        let res = run(&b, Some(&vol), &signals, &cfg);
        // Flat prices: each bar's return is pure cost on the traded 0.5 weight.
        assert!(
            (res.strategy_returns[1] + 0.0005).abs() < 1e-12,
            "{:?}",
            res.strategy_returns
        );
        assert!(
            (res.strategy_returns[2] + 0.0005).abs() < 1e-12,
            "{:?}",
            res.strategy_returns
        );
        assert_eq!(res.strategy_returns[3], 0.0);
    }

    #[test]
    fn scale_in_averages_entry_price_for_stops() {
        // Enter 0.5 @100, then 0.5 @110 → avg entry 105; 10% stop at 94.5.
        let open = vec![100.0, 100.0, 110.0, 110.0, 96.0, 90.0];
        let close = open.clone();
        let high: Vec<f64> = open.iter().map(|p| p + 0.1).collect();
        let low = vec![99.9, 99.9, 109.9, 109.9, 95.0, 89.0];
        let vol = vec![5_000.0; 6];
        let cfg = BacktestConfig {
            max_participation: 0.1, // 0.5 weight @100, then the last 0.5 @110
            stop_loss_pct: 0.1,
            ..Default::default()
        };
        let res = backtest_ohlcv_volume_core(
            &open,
            &high,
            &low,
            &close,
            Some(&vol),
            &[1.0; 6],
            &cfg,
            None,
        )
        .unwrap();
        // Stop at 94.5 is not hit on bar 4 (low 95) but is on bar 5 (low 89).
        assert!(res.positions[4] > 0.0);
        assert_eq!(res.positions[5], 0.0);
        assert!(
            (res.fill_prices[5] - 94.5).abs() < 1e-6,
            "{:?}",
            res.fill_prices
        );
    }

    #[test]
    fn scale_in_resizes_margin_baseline() {
        // 0.5 + 0.5 fills @100 → full 1.0 position needs 0.5 margin at 50%
        // ratio; a 15% gap down leaves 0.35 equity > 0.25 call level.
        let open = vec![100.0, 100.0, 100.0, 100.0, 85.0, 85.0];
        let close = open.clone();
        let high: Vec<f64> = open.iter().map(|p| p + 0.5).collect();
        let low: Vec<f64> = open.iter().map(|p| p - 0.5).collect();
        let cfg = BacktestConfig {
            max_participation: 0.1,
            margin_ratio: 0.5,
            margin_call_pct: 0.5,
            ..Default::default()
        };
        let res = backtest_ohlcv_volume_core(
            &open,
            &high,
            &low,
            &close,
            Some(&[5_000.0; 6]),
            &[1.0; 6],
            &cfg,
            None,
        )
        .unwrap();
        assert_eq!(res.positions[1..3], [0.5, 1.0]);
        assert_eq!(
            res.positions[4], 1.0,
            "no margin call expected: {:?}",
            res.positions
        );
    }

    #[test]
    fn participation_requires_volume() {
        let b = flat_bars(5, 100.0);
        let cfg = BacktestConfig {
            max_participation: 0.1,
            ..Default::default()
        };
        assert!(backtest_ohlcv_volume_core(
            &b.open, &b.high, &b.low, &b.close, None, &[1.0; 5], &cfg, None
        )
        .is_err());
        let short = vec![1.0; 3];
        assert!(backtest_ohlcv_volume_core(
            &b.open,
            &b.high,
            &b.low,
            &b.close,
            Some(&short),
            &[1.0; 5],
            &cfg,
            None
        )
        .is_err());
    }

    #[test]
    fn rejects_bad_execution_params() {
        let b = flat_bars(5, 100.0);
        for cfg in [
            BacktestConfig {
                max_participation: -0.1,
                ..Default::default()
            },
            BacktestConfig {
                slippage_atr_mult: -1.0,
                ..Default::default()
            },
            BacktestConfig {
                slippage_atr_mult: 1.0,
                slippage_atr_period: 0,
                ..Default::default()
            },
        ] {
            assert!(run_result(&b, &cfg).is_err());
        }
    }

    fn run_result(
        b: &Bars,
        cfg: &BacktestConfig,
    ) -> Result<crate::backtest::OhlcvBacktestResult, String> {
        backtest_ohlcv_volume_core(
            &b.open, &b.high, &b.low, &b.close, None, &[1.0; 5], cfg, None,
        )
    }
}
