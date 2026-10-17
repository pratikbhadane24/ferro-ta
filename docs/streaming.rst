Streaming API
=============

The :mod:`ferro_ta.streaming` module provides stateful, bar-by-bar indicator computation
for live/real-time trading. Each class maintains an internal buffer and returns ``NaN``
during the warmup period.

Live bars: committing vs. previewing
------------------------------------

Exchanges deliver many ticks per bar. ``update(..., closed=True)`` (the
default) commits a finished bar. ``update(..., closed=False)`` previews the
still-forming bar from the last committed state *without changing it*, so each
tick of the forming bar replaces the previous preview:

.. code-block:: python

    from ferro_ta.data.streaming import StreamingRSI

    rsi = StreamingRSI(14)
    for bar in history:                      # closed bars
        rsi.update(bar.close)

    for tick in live_ticks:                  # forming bar
        value = rsi.update(tick.price, closed=tick.bar_closed)

Committed output is bit-identical whether or not previews were interleaved.
``closed`` is keyword-only. Previews are allocation-free for every indicator
and cost about the same as a commit.

Building bars from ticks
------------------------

:class:`BarBuilder` turns raw ticks into OHLCV bars on a session-aligned grid
(``session_open + k * interval``; the last bar is cut at ``session_close``).
Defaults are NSE hours, 09:15–15:30 IST, so 5-minute bars start at 09:15, 09:20, …

.. code-block:: python

    from ferro_ta.data.streaming import BarBuilder, StreamingRSI

    bars, rsi = BarBuilder(300), StreamingRSI(14)
    for tick in feed:                                   # ts in UTC epoch ms
        status, closed = bars.on_tick(tick.ts_ms, tick.price, tick.qty)
        if closed is not None:
            rsi.update(closed.close)                    # commit on close
        if bars.forming is not None:
            live_rsi = rsi.update(bars.forming.close, closed=False)
    # from a timer, so bars close on time even when the market is quiet:
    if (closed := bars.on_time(now_ms)) is not None:
        rsi.update(closed.close)

- A tick exactly on a boundary opens the next bar.
- ``on_tick`` returns a status: ``"accepted"``, ``"late"`` (older than the last
  accepted tick or inside a closed bar), ``"duplicate"`` (same timestamp, price
  and qty as the last tick), ``"out_of_session"`` or ``"invalid"``. Only
  accepted ticks change state.
- Gaps produce **no** bars, so no holiday calendar is needed.
- Other sessions: ``BarBuilder(3600, session_open="09:30", session_close="16:00",
  utc_offset_minutes=-300)``.

Conditions
----------

Event helpers emit ``1`` / ``-1`` / ``0`` and support previews like any indicator:

- ``StreamingCross(dead_band)`` — cross of ``a`` over ``b`` with hysteresis:
  inside ``±dead_band`` the side is unchanged, so noise around a level fires at
  most once per real crossing (``dead_band=0`` matches batch ``check_cross``).
- ``StreamingStreak(n)`` — ``1`` while the last ``n``+ inputs were all non-zero.
- ``StreamingCooldown(bars)`` — passes a signal, then mutes the next ``bars`` bars.

.. code-block:: python

    cross, cooldown = StreamingCross(dead_band=0.5), StreamingCooldown(10)
    if cooldown.update(cross.update(rsi_value, 30.0)) == 1:
        alert("RSI crossed above 30")

Option IV and greeks
--------------------

``StreamingOptionGreeks(strike, option_type="call", model="bsm", rate=0.0,
carry=0.0)`` takes ``(option_price, underlying, time_to_expiry_years)`` per tick
and returns ``(iv, delta, gamma, vega, theta, rho)``. The IV solver starts from
the previous tick's IV; a price no volatility can reproduce returns all-NaN
and keeps that starting point.

Introspection and snapshots
---------------------------

- ``count`` — number of committed bars (previews and NaN/±inf ticks excluded).
- ``is_ready`` — ``True`` once the last committed bar produced a non-NaN value.
- ``copy.copy(ind)`` / ``copy.deepcopy(ind)`` — independent snapshot of the
  full state, e.g. to evaluate a what-if path without disturbing the live one.
- ``ind.to_json()`` / ``Class.from_json(s)`` and ``pickle`` — save state and
  restore it in another process; the restored object continues bit-identically.
  Snapshots are ``{"format": 1, "kind": "rsi", "state": {...}}``. Loading a
  different ``kind`` or an unsupported ``format`` raises ``ValueError``. The
  format number is bumped whenever a saved layout changes, so old snapshots
  fail loudly instead of loading wrong.

Non-finite ticks
----------------

A tick with any NaN or ±inf input (bad print, feed gap) returns NaN and is otherwise
**skipped**: indicator state, ``count`` and the last value are untouched, so
the next clean bar continues as if the bad tick never arrived. Previously a
single NaN poisoned running sums (SMA, Bollinger) and recursive averages (EMA,
RSI, ATR) permanently; infinity did the same via ``inf - inf``.

Other bindings
--------------

- **WASM**: ``update(...)`` commits, ``peek(...)`` previews; ``count`` and
  ``isReady`` getters.
  ``WasmBarBuilder`` (``onTick`` returns ``{status, closed}``). ``saveState`` /
  ``loadState`` need the opt-in ``snapshot`` cargo feature, which keeps the
  default bundle size unchanged.
- **C ABI / Go**: ``ft_stream_*_peek`` / ``Stream*.Peek`` mirror ``update``
  without committing; ``ft_stream_*_save`` / ``_load`` and ``Save`` /
  ``LoadStream*`` snapshot state; ``ft_bar_builder_*`` / ``BarBuilder``.

API reference
-------------

.. automodule:: ferro_ta.streaming
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
