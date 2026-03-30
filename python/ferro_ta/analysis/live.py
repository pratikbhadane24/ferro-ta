"""
Paper trading bridge — event-driven bar-by-bar simulation.

PaperTrader
    Simulates live order execution using the same logic as the backtester,
    but processes one bar at a time. Maintains live state (position, equity, trades).

Usage:
    from ferro_ta.analysis.live import PaperTrader

    trader = PaperTrader(initial_capital=100_000)
    for bar in streaming_bars:
        signal = my_strategy(bar)
        result = trader.on_bar(
            open_=bar.open, high=bar.high, low=bar.low, close=bar.close,
            signal=signal
        )
        if result.filled:
            print(f"Order filled at {result.fill_price}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class BarResult:
    """Result of processing one bar through PaperTrader."""

    bar_index: int
    filled: bool  # whether an order was executed this bar
    fill_price: float  # NaN if no fill
    position: float  # position after this bar
    equity: float  # equity after this bar (normalized, initial = 1.0)
    equity_abs: float  # absolute equity in currency units
    pnl_bar: float  # P&L this bar as fraction of initial capital
    regime: Optional[int] = None  # regime label if regime detection is enabled


@dataclass
class TradeRecord:
    """Record of a completed round-trip trade."""

    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    position: float  # +1 long, -1 short
    pnl_pct: float  # P&L as fraction of initial capital
    pnl_abs: float  # P&L in currency units


class PaperTrader:
    """Event-driven paper trading simulator.

    Processes bars one at a time, maintaining live state.
    Supports stop-loss, take-profit, trailing stop, and breakeven stop.

    Parameters
    ----------
    initial_capital : float
        Starting capital in base currency.
    stop_loss_pct : float
        Stop-loss distance from entry (fraction). 0 = disabled.
    take_profit_pct : float
        Take-profit distance from entry (fraction). 0 = disabled.
    trailing_stop_pct : float
        Trailing stop distance (fraction). 0 = disabled.
    breakeven_pct : float
        Move stop to breakeven when this profit is reached. 0 = disabled.
    slippage_bps : float
        Slippage in basis points per fill.
    commission_model : optional CommissionModel
        Full commission model. None = zero commission.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        trailing_stop_pct: float = 0.0,
        breakeven_pct: float = 0.0,
        slippage_bps: float = 0.0,
        commission_model=None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.trailing_stop_pct = float(trailing_stop_pct)
        self.breakeven_pct = float(breakeven_pct)
        self.slippage_bps = float(slippage_bps)
        self.commission_model = commission_model

        # Live state
        self._position: float = 0.0
        self._entry_price: float = float("nan")
        self._equity: float = 1.0  # normalized
        self._prev_close: float = float("nan")
        self._bar_index: int = 0
        self._trail_high: float = float("nan")
        self._trail_low: float = float("nan")
        self._breakeven_activated: bool = False
        self._breakeven_stop: float = float("nan")
        self._trades: list[TradeRecord] = []
        self._equity_history: list[float] = []

        # One-bar-lag signal state
        self._pending_signal: float = 0.0
        self._first_bar: bool = True

    def _close_position(self) -> None:
        """Reset all trade-tracking state to flat (mirrors Rust OhlcvState.close_position)."""
        self._position = 0.0
        self._entry_price = float("nan")
        self._trail_high = float("nan")
        self._trail_low = float("nan")
        self._breakeven_activated = False
        self._breakeven_stop = float("nan")

    def _commission_cost(self, fill_price: float, pos_size: float) -> float:
        """Compute commission cost as fraction of initial capital."""
        if self.commission_model is None:
            return 0.0
        try:
            trade_value = abs(pos_size) * fill_price * self.initial_capital
            if hasattr(self.commission_model, "cost_fraction"):
                return self.commission_model.cost_fraction(
                    trade_value, 1.0, pos_size > 0, self.initial_capital
                )
        except Exception:
            pass
        return 0.0

    def on_bar(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        signal: float,
    ) -> BarResult:
        """Process one bar and return a BarResult.

        signal : float
            Desired position (+1, -1, or 0). Applied next bar (standard bar-by-bar logic).
            For this bar, the signal from the PREVIOUS bar is acted upon.
        """
        nan = float("nan")
        slip = self.slippage_bps / 10_000.0

        bar_idx = self._bar_index
        self._bar_index += 1

        # On the very first bar: record signal, no action (no prev signal yet)
        if self._first_bar:
            self._pending_signal = signal
            self._first_bar = False
            self._prev_close = close
            self._equity_history.append(self._equity)
            return BarResult(
                bar_index=bar_idx,
                filled=False,
                fill_price=nan,
                position=self._position,
                equity=self._equity,
                equity_abs=self._equity * self.initial_capital,
                pnl_bar=0.0,
            )

        # The signal to act on this bar is from the previous call
        desired_pos = (
            self._pending_signal if not math.isnan(self._pending_signal) else 0.0
        )
        # Store current bar's signal for next bar
        self._pending_signal = signal

        prev_close = self._prev_close
        self._prev_close = close

        strategy_return = 0.0
        fill_price_this_bar = nan
        filled = False
        forced_close = False

        # ---- Update trailing stop water marks ----
        if self.trailing_stop_pct > 0.0:
            if self._position > 0.0 and not math.isnan(self._trail_high):
                self._trail_high = max(self._trail_high, high)
            if self._position < 0.0 and not math.isnan(self._trail_low):
                self._trail_low = min(self._trail_low, low)

        close_ret = (close - prev_close) / prev_close if prev_close != 0.0 else 0.0

        # ---- Trailing stop check ----
        if (
            self.trailing_stop_pct > 0.0
            and self._position != 0.0
            and not math.isnan(self._entry_price)
        ):
            if self._position > 0.0 and not math.isnan(self._trail_high):
                trail_stop = self._trail_high * (1.0 - self.trailing_stop_pct)
                if low <= trail_stop:
                    stop_ret = (
                        (trail_stop - prev_close) / prev_close
                        if prev_close != 0.0
                        else -self.trailing_stop_pct
                    )
                    comm = self._commission_cost(trail_stop, self._position)
                    strategy_return = self._position * stop_ret - slip - comm
                    fill_price_this_bar = trail_stop
                    filled = True
                    self._record_trade(bar_idx, trail_stop)
                    self._close_position()
                    forced_close = True

            elif self._position < 0.0 and not math.isnan(self._trail_low):
                trail_stop = self._trail_low * (1.0 + self.trailing_stop_pct)
                if high >= trail_stop:
                    stop_ret = (
                        (trail_stop - prev_close) / prev_close
                        if prev_close != 0.0
                        else self.trailing_stop_pct
                    )
                    comm = self._commission_cost(trail_stop, self._position)
                    strategy_return = self._position * stop_ret - slip - comm
                    fill_price_this_bar = trail_stop
                    filled = True
                    self._record_trade(bar_idx, trail_stop)
                    self._close_position()
                    forced_close = True

        # ---- Breakeven stop activation ----
        if (
            self.breakeven_pct > 0.0
            and self._position != 0.0
            and not math.isnan(self._entry_price)
            and not self._breakeven_activated
        ):
            if self._position > 0.0 and high >= self._entry_price * (
                1.0 + self.breakeven_pct
            ):
                self._breakeven_activated = True
                self._breakeven_stop = self._entry_price
            elif self._position < 0.0 and low <= self._entry_price * (
                1.0 - self.breakeven_pct
            ):
                self._breakeven_activated = True
                self._breakeven_stop = self._entry_price

        # ---- SL/TP combined bracket check ----
        if (
            not forced_close
            and self._position != 0.0
            and not math.isnan(self._entry_price)
        ):
            entry = self._entry_price
            has_stop = self._breakeven_activated or self.stop_loss_pct > 0.0
            stop_long = (
                self._breakeven_stop
                if self._breakeven_activated
                else entry * (1.0 - self.stop_loss_pct)
            )
            stop_short = (
                self._breakeven_stop
                if self._breakeven_activated
                else entry * (1.0 + self.stop_loss_pct)
            )
            has_tp = self.take_profit_pct > 0.0
            tp_long = entry * (1.0 + self.take_profit_pct)
            tp_short = entry * (1.0 - self.take_profit_pct)

            if self._position > 0.0:
                sl_triggered = has_stop and low <= stop_long
                tp_triggered = has_tp and high >= tp_long

                if sl_triggered and tp_triggered:
                    sl_dist = abs(open_ - stop_long)
                    tp_dist = abs(tp_long - open_)
                    if sl_dist <= tp_dist:
                        # SL first
                        sr = (
                            (stop_long - prev_close) / prev_close
                            if prev_close != 0.0
                            else -self.stop_loss_pct
                        )
                        comm = self._commission_cost(stop_long, self._position)
                        strategy_return = self._position * sr - slip - comm
                        fill_price_this_bar = stop_long
                    else:
                        sr = (
                            (tp_long - prev_close) / prev_close
                            if prev_close != 0.0
                            else self.take_profit_pct
                        )
                        comm = self._commission_cost(tp_long, self._position)
                        strategy_return = self._position * sr - slip - comm
                        fill_price_this_bar = tp_long
                    filled = True
                    self._record_trade(bar_idx, fill_price_this_bar)
                    self._close_position()
                    forced_close = True

                elif sl_triggered:
                    sr = (
                        (stop_long - prev_close) / prev_close
                        if prev_close != 0.0
                        else -self.stop_loss_pct
                    )
                    comm = self._commission_cost(stop_long, self._position)
                    strategy_return = self._position * sr - slip - comm
                    fill_price_this_bar = stop_long
                    filled = True
                    self._record_trade(bar_idx, stop_long)
                    self._close_position()
                    forced_close = True

                elif tp_triggered:
                    sr = (
                        (tp_long - prev_close) / prev_close
                        if prev_close != 0.0
                        else self.take_profit_pct
                    )
                    comm = self._commission_cost(tp_long, self._position)
                    strategy_return = self._position * sr - slip - comm
                    fill_price_this_bar = tp_long
                    filled = True
                    self._record_trade(bar_idx, tp_long)
                    self._close_position()
                    forced_close = True

            elif self._position < 0.0:
                sl_triggered = has_stop and high >= stop_short
                tp_triggered = has_tp and low <= tp_short

                if sl_triggered and tp_triggered:
                    sl_dist = abs(stop_short - open_)
                    tp_dist = abs(open_ - tp_short)
                    if sl_dist <= tp_dist:
                        sr = (
                            (stop_short - prev_close) / prev_close
                            if prev_close != 0.0
                            else self.stop_loss_pct
                        )
                        comm = self._commission_cost(stop_short, self._position)
                        strategy_return = self._position * sr - slip - comm
                        fill_price_this_bar = stop_short
                    else:
                        sr = (
                            (tp_short - prev_close) / prev_close
                            if prev_close != 0.0
                            else -self.take_profit_pct
                        )
                        comm = self._commission_cost(tp_short, self._position)
                        strategy_return = self._position * sr - slip - comm
                        fill_price_this_bar = tp_short
                    filled = True
                    self._record_trade(bar_idx, fill_price_this_bar)
                    self._close_position()
                    forced_close = True

                elif sl_triggered:
                    sr = (
                        (stop_short - prev_close) / prev_close
                        if prev_close != 0.0
                        else self.stop_loss_pct
                    )
                    comm = self._commission_cost(stop_short, self._position)
                    strategy_return = self._position * sr - slip - comm
                    fill_price_this_bar = stop_short
                    filled = True
                    self._record_trade(bar_idx, stop_short)
                    self._close_position()
                    forced_close = True

                elif tp_triggered:
                    sr = (
                        (tp_short - prev_close) / prev_close
                        if prev_close != 0.0
                        else -self.take_profit_pct
                    )
                    comm = self._commission_cost(tp_short, self._position)
                    strategy_return = self._position * sr - slip - comm
                    fill_price_this_bar = tp_short
                    filled = True
                    self._record_trade(bar_idx, tp_short)
                    self._close_position()
                    forced_close = True

        # ---- Normal signal execution ----
        if not forced_close:
            pos_changed = abs(desired_pos - self._position) > 1e-12
            # Fill at open (market_open mode, same as Rust default)
            base_fill = open_
            if desired_pos > self._position:
                actual_fill = base_fill * (1.0 + slip)
            elif desired_pos < self._position:
                actual_fill = base_fill * (1.0 - slip)
            else:
                actual_fill = base_fill

            if pos_changed:
                fill_price_this_bar = actual_fill
                filled = True

                old_pos = self._position

                if desired_pos != 0.0 and old_pos == 0.0:
                    r = (
                        desired_pos * (close - actual_fill) / actual_fill
                        if actual_fill != 0.0
                        else 0.0
                    )
                    comm = self._commission_cost(actual_fill, desired_pos)
                    strategy_return = r - comm
                    self._set_entry(bar_idx, actual_fill, desired_pos)
                elif desired_pos == 0.0:
                    r = (
                        old_pos * (actual_fill - prev_close) / prev_close
                        if prev_close != 0.0
                        else 0.0
                    )
                    comm = self._commission_cost(actual_fill, old_pos)
                    strategy_return = r - comm
                    self._record_trade(bar_idx, actual_fill)
                    self._close_position()
                else:
                    exit_r = (
                        old_pos * (actual_fill - prev_close) / prev_close
                        if prev_close != 0.0
                        else 0.0
                    )
                    entry_r = (
                        desired_pos * (close - actual_fill) / actual_fill
                        if actual_fill != 0.0
                        else 0.0
                    )
                    exit_comm = self._commission_cost(actual_fill, old_pos)
                    entry_comm = self._commission_cost(actual_fill, desired_pos)
                    strategy_return = exit_r + entry_r - exit_comm - entry_comm
                    if old_pos != 0.0:
                        self._record_trade(bar_idx, actual_fill)
                    self._set_entry(bar_idx, actual_fill, desired_pos)

                self._position = desired_pos

            else:
                # Hold: full bar return (close-to-close on existing position)
                strategy_return = self._position * close_ret

        # Update equity
        prev_equity = self._equity
        self._equity = self._equity * (1.0 + strategy_return)
        pnl_bar = self._equity - prev_equity

        self._equity_history.append(self._equity)

        return BarResult(
            bar_index=bar_idx,
            filled=filled,
            fill_price=fill_price_this_bar,
            position=self._position,
            equity=self._equity,
            equity_abs=self._equity * self.initial_capital,
            pnl_bar=pnl_bar,
        )

    def _record_trade(self, exit_bar: int, exit_price: float) -> None:
        """Record a completed round-trip trade."""
        if math.isnan(self._entry_price):
            return
        entry_price = self._entry_price
        pos = self._position
        # P&L = position * (exit - entry) / entry  as fraction
        if entry_price != 0.0:
            pnl_pct = pos * (exit_price - entry_price) / entry_price
        else:
            pnl_pct = 0.0
        pnl_abs = pnl_pct * self.initial_capital

        self._trades.append(
            TradeRecord(
                entry_bar=getattr(self, "_trade_entry_bar", 0),
                exit_bar=exit_bar,
                entry_price=entry_price,
                exit_price=exit_price,
                position=pos,
                pnl_pct=pnl_pct,
                pnl_abs=pnl_abs,
            )
        )

    def _set_entry(self, bar_idx: int, fill_price: float, pos: float) -> None:
        """Set entry state — call after position changes to new non-zero position."""
        self._entry_price = fill_price
        self._trade_entry_bar = bar_idx
        self._trail_high = fill_price if pos > 0.0 else float("nan")
        self._trail_low = fill_price if pos < 0.0 else float("nan")
        self._breakeven_activated = False
        self._breakeven_stop = float("nan")

    @property
    def position(self) -> float:
        """Current open position."""
        return self._position

    @property
    def equity(self) -> float:
        """Current normalized equity."""
        return self._equity

    @property
    def equity_abs(self) -> float:
        """Current absolute equity in base currency."""
        return self._equity * self.initial_capital

    @property
    def trades(self) -> list[TradeRecord]:
        """List of completed trades."""
        return list(self._trades)

    @property
    def equity_curve(self) -> list[float]:
        """Equity history (normalized)."""
        return list(self._equity_history)

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._position = 0.0
        self._entry_price = float("nan")
        self._equity = 1.0
        self._prev_close = float("nan")
        self._bar_index = 0
        self._trail_high = float("nan")
        self._trail_low = float("nan")
        self._breakeven_activated = False
        self._breakeven_stop = float("nan")
        self._trades = []
        self._equity_history = []
        self._pending_signal = 0.0
        self._first_bar = True
