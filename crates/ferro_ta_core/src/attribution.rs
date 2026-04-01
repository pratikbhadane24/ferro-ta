//! Performance attribution and trade analysis — pure Rust, no PyO3.
//!
//! Functions
//! ---------
//! - `trade_stats`          — win rate, avg win/loss, profit factor, avg hold
//! - `monthly_contribution` — group bar returns by month index and sum
//! - `signal_attribution`   — group bar returns by signal label and sum
//! - `extract_trades`       — extract trade pnl and hold durations from positions

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// trade_stats
// ---------------------------------------------------------------------------

/// Compute trade-level statistics from trade PnL and hold durations.
///
/// Returns `(win_rate, avg_win, avg_loss, profit_factor, avg_hold_bars)`.
///
/// - **win_rate**      : fraction of trades with PnL > 0
/// - **avg_win**       : mean PnL of winning trades (0 if none)
/// - **avg_loss**      : mean PnL of losing trades (negative; 0 if none)
/// - **profit_factor** : gross profit / |gross loss| (inf if no losses)
/// - **avg_hold_bars** : mean hold duration across all trades
///
/// # Panics
/// Panics if `pnl` is empty or `pnl.len() != hold_bars.len()`.
pub fn trade_stats(pnl: &[f64], hold_bars: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = pnl.len();
    assert!(n > 0, "pnl must be non-empty");
    assert_eq!(
        n,
        hold_bars.len(),
        "pnl and hold_bars must have equal length"
    );

    let mut wins: Vec<f64> = Vec::new();
    let mut losses: Vec<f64> = Vec::new();
    for &v in pnl.iter() {
        if v > 0.0 {
            wins.push(v);
        } else if v < 0.0 {
            losses.push(v);
        }
    }

    let win_rate = wins.len() as f64 / n as f64;
    let avg_win = if wins.is_empty() {
        0.0
    } else {
        wins.iter().sum::<f64>() / wins.len() as f64
    };
    let avg_loss = if losses.is_empty() {
        0.0
    } else {
        losses.iter().sum::<f64>() / losses.len() as f64
    };

    let gross_profit: f64 = wins.iter().sum();
    let gross_loss: f64 = losses.iter().map(|v| v.abs()).sum();
    let profit_factor = if gross_loss == 0.0 {
        f64::INFINITY
    } else {
        gross_profit / gross_loss
    };

    let avg_hold = hold_bars.iter().sum::<f64>() / n as f64;

    (win_rate, avg_win, avg_loss, profit_factor, avg_hold)
}

// ---------------------------------------------------------------------------
// monthly_contribution
// ---------------------------------------------------------------------------

/// Group per-bar returns by month index and sum each month's contribution.
///
/// Returns `(months, contributions)` where `months` is sorted unique month
/// indices and `contributions` is the corresponding total return per month.
/// NaN returns are skipped.
///
/// # Panics
/// Panics if `bar_returns.len() != month_index.len()`.
pub fn monthly_contribution(bar_returns: &[f64], month_index: &[i64]) -> (Vec<i64>, Vec<f64>) {
    let n = bar_returns.len();
    assert_eq!(
        n,
        month_index.len(),
        "bar_returns and month_index must have equal length"
    );

    let mut map: HashMap<i64, f64> = HashMap::new();
    for i in 0..n {
        if !bar_returns[i].is_nan() {
            *map.entry(month_index[i]).or_insert(0.0) += bar_returns[i];
        }
    }

    let mut months: Vec<i64> = map.keys().copied().collect();
    months.sort_unstable();
    let contributions: Vec<f64> = months.iter().map(|m| map[m]).collect();

    (months, contributions)
}

// ---------------------------------------------------------------------------
// signal_attribution
// ---------------------------------------------------------------------------

/// Attribute per-bar returns to each signal label.
///
/// Returns `(labels, contributions)` where `labels` is sorted unique signal
/// labels and `contributions` is the corresponding total return per label.
/// NaN returns are skipped.
///
/// # Panics
/// Panics if `bar_returns.len() != signal_labels.len()`.
pub fn signal_attribution(bar_returns: &[f64], signal_labels: &[i64]) -> (Vec<i64>, Vec<f64>) {
    let n = bar_returns.len();
    assert_eq!(
        n,
        signal_labels.len(),
        "bar_returns and signal_labels must have equal length"
    );

    let mut map: HashMap<i64, f64> = HashMap::new();
    for i in 0..n {
        if !bar_returns[i].is_nan() {
            *map.entry(signal_labels[i]).or_insert(0.0) += bar_returns[i];
        }
    }

    let mut labels: Vec<i64> = map.keys().copied().collect();
    labels.sort_unstable();
    let contributions: Vec<f64> = labels.iter().map(|l| map[l]).collect();

    (labels, contributions)
}

// ---------------------------------------------------------------------------
// extract_trades
// ---------------------------------------------------------------------------

/// Extract trade-level PnL and hold durations from positions and strategy returns.
///
/// A trade is a maximal contiguous run of non-zero position values with the
/// same sign/magnitude. Returns `(pnl, hold_durations)`.
///
/// # Panics
/// Panics if `positions.len() != strategy_returns.len()`.
pub fn extract_trades(positions: &[f64], strategy_returns: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = positions.len();
    assert_eq!(
        n,
        strategy_returns.len(),
        "positions and strategy_returns must have equal length"
    );

    let mut pnl = Vec::<f64>::new();
    let mut hold = Vec::<f64>::new();

    let mut i = 0usize;
    while i < n {
        if positions[i] == 0.0 {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < n && positions[j] == positions[i] {
            j += 1;
        }
        let mut trade_pnl = 0.0_f64;
        for v in strategy_returns.iter().take(j).skip(i) {
            trade_pnl += *v;
        }
        pnl.push(trade_pnl);
        hold.push((j - i) as f64);
        i = j;
    }

    (pnl, hold)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- trade_stats ---------------------------------------------------------

    #[test]
    fn test_trade_stats_basic() {
        let pnl = [100.0, -50.0, 200.0, -30.0, 150.0];
        let hold = [5.0, 3.0, 7.0, 2.0, 6.0];
        let (wr, aw, al, pf, ah) = trade_stats(&pnl, &hold);

        // 3 wins out of 5
        assert!((wr - 0.6).abs() < 1e-10);
        // avg win = (100+200+150)/3
        assert!((aw - 150.0).abs() < 1e-10);
        // avg loss = (-50 + -30)/2 = -40
        assert!((al - (-40.0)).abs() < 1e-10);
        // profit_factor = 450 / 80
        assert!((pf - 5.625).abs() < 1e-10);
        // avg hold = (5+3+7+2+6)/5 = 4.6
        assert!((ah - 4.6).abs() < 1e-10);
    }

    #[test]
    fn test_trade_stats_all_wins() {
        let pnl = [10.0, 20.0];
        let hold = [1.0, 2.0];
        let (wr, _aw, al, pf, _ah) = trade_stats(&pnl, &hold);
        assert!((wr - 1.0).abs() < 1e-10);
        assert!((al - 0.0).abs() < 1e-10);
        assert!(pf.is_infinite());
    }

    #[test]
    fn test_trade_stats_all_losses() {
        let pnl = [-10.0, -20.0];
        let hold = [1.0, 2.0];
        let (wr, aw, _al, pf, _ah) = trade_stats(&pnl, &hold);
        assert!((wr - 0.0).abs() < 1e-10);
        assert!((aw - 0.0).abs() < 1e-10);
        assert!((pf - 0.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "pnl must be non-empty")]
    fn test_trade_stats_empty() {
        trade_stats(&[], &[]);
    }

    // -- monthly_contribution ------------------------------------------------

    #[test]
    fn test_monthly_contribution_basic() {
        let returns = [0.01, 0.02, -0.01, 0.03, -0.02];
        let months = [0, 0, 1, 1, 2];
        let (m, c) = monthly_contribution(&returns, &months);
        assert_eq!(m, vec![0, 1, 2]);
        assert!((c[0] - 0.03).abs() < 1e-10);
        assert!((c[1] - 0.02).abs() < 1e-10);
        assert!((c[2] - (-0.02)).abs() < 1e-10);
    }

    #[test]
    fn test_monthly_contribution_nan_skipped() {
        let returns = [0.01, f64::NAN, 0.03];
        let months = [0, 0, 1];
        let (m, c) = monthly_contribution(&returns, &months);
        assert_eq!(m, vec![0, 1]);
        assert!((c[0] - 0.01).abs() < 1e-10);
        assert!((c[1] - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_monthly_contribution_empty() {
        let (m, c) = monthly_contribution(&[], &[]);
        assert!(m.is_empty());
        assert!(c.is_empty());
    }

    // -- signal_attribution --------------------------------------------------

    #[test]
    fn test_signal_attribution_basic() {
        let returns = [0.05, -0.02, 0.03, 0.01];
        let labels = [1, -1, 2, 1];
        let (l, c) = signal_attribution(&returns, &labels);
        assert_eq!(l, vec![-1, 1, 2]);
        assert!((c[0] - (-0.02)).abs() < 1e-10);
        assert!((c[1] - 0.06).abs() < 1e-10); // 0.05 + 0.01
        assert!((c[2] - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_signal_attribution_nan_skipped() {
        let returns = [0.05, f64::NAN];
        let labels = [1, 2];
        let (l, c) = signal_attribution(&returns, &labels);
        assert_eq!(l, vec![1]);
        assert!((c[0] - 0.05).abs() < 1e-10);
    }

    // -- extract_trades ------------------------------------------------------

    #[test]
    fn test_extract_trades_basic() {
        // positions: flat, long, long, flat, short, short
        let positions = [0.0, 1.0, 1.0, 0.0, -1.0, -1.0];
        let strat_ret = [0.0, 0.01, 0.02, 0.0, -0.01, 0.03];
        let (pnl, hold) = extract_trades(&positions, &strat_ret);
        assert_eq!(pnl.len(), 2);
        assert_eq!(hold.len(), 2);
        // First trade: bars 1..3 => 0.01 + 0.02 = 0.03
        assert!((pnl[0] - 0.03).abs() < 1e-10);
        assert!((hold[0] - 2.0).abs() < 1e-10);
        // Second trade: bars 4..6 => -0.01 + 0.03 = 0.02
        assert!((pnl[1] - 0.02).abs() < 1e-10);
        assert!((hold[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_trades_all_flat() {
        let positions = [0.0, 0.0, 0.0];
        let strat_ret = [0.01, 0.02, 0.03];
        let (pnl, hold) = extract_trades(&positions, &strat_ret);
        assert!(pnl.is_empty());
        assert!(hold.is_empty());
    }

    #[test]
    fn test_extract_trades_empty() {
        let (pnl, hold) = extract_trades(&[], &[]);
        assert!(pnl.is_empty());
        assert!(hold.is_empty());
    }

    #[test]
    fn test_extract_trades_single_bar_trade() {
        let positions = [0.0, 1.0, 0.0];
        let strat_ret = [0.0, 0.05, 0.0];
        let (pnl, hold) = extract_trades(&positions, &strat_ret);
        assert_eq!(pnl.len(), 1);
        assert!((pnl[0] - 0.05).abs() < 1e-10);
        assert!((hold[0] - 1.0).abs() < 1e-10);
    }
}
