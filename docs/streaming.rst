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
``closed`` is keyword-only.

Introspection and snapshots
---------------------------

- ``count`` — number of committed bars (previews and NaN/±inf ticks excluded).
- ``is_ready`` — ``True`` once the last committed bar produced a non-NaN value.
- ``copy.copy(ind)`` / ``copy.deepcopy(ind)`` — independent snapshot of the
  full state, e.g. to evaluate a what-if path without disturbing the live one.

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
- **C ABI / Go**: ``ft_stream_*_peek`` / ``Stream*.Peek`` mirror ``update``
  without committing.

API reference
-------------

.. automodule:: ferro_ta.streaming
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
