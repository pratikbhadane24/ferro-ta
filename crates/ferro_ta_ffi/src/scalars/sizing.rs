//! Scalar C exports for position sizing (`ferro_ta_core::backtest`).

use crate::status::Status;
use ferro_ta_core::backtest;

crate::ffi_scalar_exports! {
    group: "sizing",
    /// Compute the Kelly fraction: f = win_rate - (1 - win_rate) * (|avg_loss| / avg_win), clamped to [0, 1].
    ft_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> [out_value: f64]
        = backtest::kelly_fraction(win_rate, avg_win, avg_loss).map_err(|_| Status::InvalidParam)?;
    /// Half-Kelly fraction (conservative position sizing).
    ft_half_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> [out_value: f64]
        = backtest::half_kelly_fraction(win_rate, avg_win, avg_loss).map_err(|_| Status::InvalidParam)?;
}
