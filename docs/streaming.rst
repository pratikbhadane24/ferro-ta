Streaming API
=============

The :mod:`ferro_ta.streaming` module provides stateful, bar-by-bar indicator computation
for live/real-time trading. Each class maintains an internal buffer and returns ``NaN``
during the warmup period.

What's new: the streaming universe
----------------------------------

Besides the 14 hand-written classes (``StreamingSMA``, ``StreamingRSI``, ...)
there are now **215 more streams**, one for nearly every batch indicator,
generated from a single declarative registry. They share the same live-bar,
snapshot and non-finite semantics described below and exist in Python, WASM,
the C ABI and Go. The full list is the :ref:`stream-catalog`.

**Parity contract.** On finite input a stream equals the batch function at
every bar: identical NaN mask, values within 1e-9 relative, integer outputs
exact.

Naming per surface:

.. code-block:: python

    from ferro_ta.data.streaming import StreamingKeltnerChannels

    kc = StreamingKeltnerChannels(timeperiod=20, atr_period=10, multiplier=2.0)
    upper, middle, lower = kc.update(high, low, close)

.. code-block:: javascript

    const kc = new WasmStreamingKeltnerChannels(20, 10, 2.0);
    const [upper, middle, lower] = kc.update(high, low, close);  // peek(...) previews

.. code-block:: c

    FtStreamWillr *w = NULL;
    ft_stream_willr_new(14, &w);              /* _new _update _peek _reset _save _load _free */
    double out; ft_stream_willr_update(w, high, low, close, &out);
    ft_stream_willr_free(w);

.. code-block:: go

    willr, err := ferrota.NewStreamWillr(14)  // also Update, Peek, Save, LoadStreamWillr, Close
    defer willr.Close()
    value, err := willr.Update(high, low, close)

Python classes are ``Streaming<UpperCamel>`` (``StreamingWillr``), WASM classes
``WasmStreaming<UpperCamel>``, C functions ``ft_stream_<name>_*`` and Go
wrappers ``NewStream<UpperCamel>`` / ``Stream<UpperCamel>``.

Notes and exclusions
~~~~~~~~~~~~~~~~~~~~

- Streams whose warm-up value is 0 (candlestick patterns, ``ht_trendmode``) are
  ``is_ready`` from the first bar, because the batch function emits 0 there.
- ``regime_adx`` / ``regime_combined`` batch functions emit ``-1`` on NaN input;
  streams skip NaN ticks, so they never emit ``-1``.
- The ``ichimoku`` stream omits ``chikou`` (it needs future bars).
- No stream, by design: ``williams_fractals`` (centred window needs future
  bars), ``back_adjusted_continuous`` / ``ratio_adjusted_continuous`` (rewrite
  past bars), ``rank_values`` (whole-series rank) and ``forward_fill_nan``
  (its input is NaN, which streams skip).
- The 9 original streams are unchanged, so batch-only extras (``bbands`` /
  ``stoch`` ``matype``, rolling ``vwap(timeperiod>0)``) have no stream.
- Limits: ``mavp`` periods are capped at 4096; period and count parameters at
  2^24 (same as the C ABI).
- Performance: see ``benchmarks/bench_streaming_universe.py``.

Keyword names
~~~~~~~~~~~~~

Stream parameters take the C ABI names (``ft_<name>``), which match the Python
batch function everywhere except these three:

.. list-table::
   :header-rows: 1

   * - Stream
     - Stream keyword(s)
     - Python batch keyword(s)
   * - ``StreamingAc``
     - ``timeperiod``
     - ``smoothperiod``
   * - ``StreamingCoppock``
     - ``roc1``, ``roc2``
     - ``roc1_period``, ``roc2_period``
   * - ``StreamingEmv``
     - ``scale``
     - ``divisor``

So ``StreamingCoppock(roc1=14, roc2=11, wma_period=10)``, not
``roc1_period=14``. Positional arguments are in the same order on both.

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
  utc_offset_minutes=-240)`` for New York in summer.
- ``utc_offset_minutes`` is a **fixed** offset; there is no time-zone or DST
  handling. For a market that observes daylight saving time, create a new
  builder with the new offset when the clocks change (between sessions, so no
  forming bar is lost). With the wrong offset, the first or last hour of the
  session is rejected as ``"out_of_session"``. NSE (IST) has no DST.

Conditions
----------

Event helpers emit ``1`` / ``-1`` / ``0`` and support previews like any indicator:

- ``StreamingCross(dead_band)`` — cross of ``a`` over ``b`` with hysteresis:
  inside ``±dead_band`` the side is unchanged, so noise around a level fires at
  most once per real crossing (``dead_band=0`` matches batch ``check_cross``).
- ``StreamingStreak(n)`` — ``1`` while the last ``n`` (or more) inputs were all non-zero.
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


.. _stream-catalog:

Stream catalog
--------------

Kind: ``windowed`` streams replay the batch kernel over the last ``lookback``
bars; ``native`` streams update incremental state; ``legacy`` are the original
hand-written classes. Regenerate with
``python3 scripts/build_streaming_python.py --docs``.

.. BEGIN GENERATED STREAM CATALOG (scripts/build_streaming_python.py)

.. rubric:: cycle (6)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``ht_dcperiod``
     - ``StreamingHtDcperiod``
     - close
     - -
     - value
     - native
   * - ``ht_dcphase``
     - ``StreamingHtDcphase``
     - close
     - -
     - value
     - native
   * - ``ht_phasor``
     - ``StreamingHtPhasor``
     - close
     - -
     - inphase, quadrature
     - native
   * - ``ht_sine``
     - ``StreamingHtSine``
     - close
     - -
     - sine, leadsine
     - native
   * - ``ht_trendline``
     - ``StreamingHtTrendline``
     - close
     - -
     - value
     - native
   * - ``ht_trendmode``
     - ``StreamingHtTrendmode``
     - close
     - -
     - value
     - native

.. rubric:: extended (57)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``ac``
     - ``StreamingAc``
     - high, low
     - fastperiod=5, slowperiod=34, timeperiod=5
     - value
     - windowed
   * - ``alligator``
     - ``StreamingAlligator``
     - high, low
     - jaw_period=13, jaw_shift=8, teeth_period=8, teeth_shift=5, lips_period=5, lips_shift=3
     - jaw, teeth, lips
     - native
   * - ``alma``
     - ``StreamingAlma``
     - close
     - timeperiod=21, offset=0.85, sigma=6.0
     - value
     - windowed
   * - ``ao``
     - ``StreamingAo``
     - high, low
     - fastperiod=5, slowperiod=34
     - value
     - windowed
   * - ``bbpercent``
     - ``StreamingBbpercent``
     - close
     - timeperiod=5, nbdevup=2.0, nbdevdn=2.0
     - value
     - windowed
   * - ``bbwidth``
     - ``StreamingBbwidth``
     - close
     - timeperiod=5, nbdevup=2.0, nbdevdn=2.0
     - value
     - windowed
   * - ``chaikin_vol``
     - ``StreamingChaikinVol``
     - high, low
     - timeperiod=10, rocperiod=10
     - value
     - native
   * - ``chande_kroll_stop``
     - ``StreamingChandeKrollStop``
     - high, low, close
     - timeperiod=10, multiplier=1.0, stop_period=9
     - long_stop, short_stop
     - native
   * - ``chandelier_exit``
     - ``StreamingChandelierExit``
     - high, low, close
     - timeperiod=22, multiplier=3.0
     - long_exit, short_exit
     - native
   * - ``cho``
     - ``StreamingCho``
     - high, low, close, volume
     - fastperiod=3, slowperiod=10
     - value
     - native
   * - ``choppiness_index``
     - ``StreamingChoppinessIndex``
     - high, low, close
     - timeperiod=14
     - value
     - windowed
   * - ``cmf``
     - ``StreamingCmf``
     - high, low, close, volume
     - timeperiod=20
     - value
     - native
   * - ``coppock``
     - ``StreamingCoppock``
     - close
     - wma_period=10, roc1=14, roc2=11
     - value
     - windowed
   * - ``crsi``
     - ``StreamingCrsi``
     - close
     - timeperiod=3, streakperiod=2, rankperiod=100
     - value
     - native
   * - ``dmi``
     - ``StreamingDmi``
     - high, low, close
     - timeperiod=14
     - plus_di, minus_di, adx
     - native
   * - ``donchian``
     - ``StreamingDonchian``
     - high, low
     - timeperiod=20
     - upper, middle, lower
     - windowed
   * - ``dpo``
     - ``StreamingDpo``
     - close
     - timeperiod=20
     - value
     - windowed
   * - ``elder_ray``
     - ``StreamingElderRay``
     - high, low, close
     - timeperiod=13
     - bull, bear
     - native
   * - ``emv``
     - ``StreamingEmv``
     - high, low, volume
     - timeperiod=14, scale=10000.0
     - value
     - windowed
   * - ``fisher``
     - ``StreamingFisher``
     - high, low
     - timeperiod=9
     - fisher, signal
     - native
   * - ``force_index``
     - ``StreamingForceIndex``
     - close, volume
     - timeperiod=13
     - value
     - native
   * - ``frama``
     - ``StreamingFrama``
     - close
     - timeperiod=16
     - value
     - native
   * - ``gator``
     - ``StreamingGator``
     - high, low
     - jaw_period=13, jaw_shift=8, teeth_period=8, teeth_shift=5, lips_period=5, lips_shift=3
     - upper, lower
     - native
   * - ``historical_volatility``
     - ``StreamingHistoricalVolatility``
     - close
     - timeperiod=20, annual=252.0
     - value
     - windowed
   * - ``hull_ma``
     - ``StreamingHullMa``
     - close
     - timeperiod=16
     - value
     - windowed
   * - ``ichimoku``
     - ``StreamingIchimoku``
     - high, low, close
     - tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26
     - tenkan, kijun, senkou_a, senkou_b
     - native
   * - ``keltner_channels``
     - ``StreamingKeltnerChannels``
     - high, low, close
     - timeperiod=20, atr_period=10, multiplier=2.0
     - upper, middle, lower
     - native
   * - ``kst``
     - ``StreamingKst``
     - close
     - roc1=10, roc2=15, roc3=20, roc4=30, sma1=10, sma2=10, sma3=10, sma4=15, signalperiod=9
     - kst, signal
     - windowed
   * - ``kvo``
     - ``StreamingKvo``
     - high, low, close, volume
     - fastperiod=34, slowperiod=55, signalperiod=13
     - kvo, signal
     - native
   * - ``ma_envelopes``
     - ``StreamingMaEnvelopes``
     - close
     - timeperiod=20, percent=2.5, matype=0
     - upper, middle, lower
     - native
   * - ``mass``
     - ``StreamingMass``
     - high, low
     - timeperiod=9, sumperiod=25
     - value
     - native
   * - ``mcginley``
     - ``StreamingMcginley``
     - close
     - timeperiod=14
     - value
     - native
   * - ``median``
     - ``StreamingMedian``
     - real
     - timeperiod=3
     - value
     - windowed
   * - ``median_bands``
     - ``StreamingMedianBands``
     - high, low, close
     - timeperiod=3, atr_period=14, multiplier=2.0
     - median, upper, lower, median_ema
     - native
   * - ``mode``
     - ``StreamingMode``
     - real
     - timeperiod=20, bins=10
     - value
     - windowed
   * - ``nvi``
     - ``StreamingNvi``
     - close, volume
     - -
     - value
     - native
   * - ``nvi_with_ema``
     - ``StreamingNviWithEma``
     - close, volume
     - timeperiod=255
     - nvi, signal
     - native
   * - ``obv_smoothed``
     - ``StreamingObvSmoothed``
     - close, volume
     - timeperiod=20, matype=1
     - value
     - native
   * - ``po``
     - ``StreamingPo``
     - close
     - fastperiod=10, slowperiod=21
     - value
     - windowed
   * - ``pvi``
     - ``StreamingPvi``
     - close, volume
     - -
     - value
     - native
   * - ``pvi_with_signal``
     - ``StreamingPviWithSignal``
     - close, volume
     - timeperiod=255, matype=1
     - pvi, signal
     - native
   * - ``pvt``
     - ``StreamingPvt``
     - close, volume
     - -
     - value
     - native
   * - ``rvi``
     - ``StreamingRvi``
     - open, high, low, close
     - timeperiod=10
     - rvi, signal
     - native
   * - ``rvol``
     - ``StreamingRvol``
     - volume
     - timeperiod=20
     - value
     - native
   * - ``rwi``
     - ``StreamingRwi``
     - high, low, close
     - timeperiod=14
     - rwi_high, rwi_low
     - windowed
   * - ``starc``
     - ``StreamingStarc``
     - high, low, close
     - timeperiod=15, atr_period=15, multiplier=2.0
     - upper, middle, lower
     - native
   * - ``stc``
     - ``StreamingStc``
     - close
     - fastperiod=23, slowperiod=50, cycleperiod=10, d1=3, d2=3
     - value
     - native
   * - ``supertrend``
     - ``StreamingSupertrend``
     - high, low, close
     - period=7, multiplier=...
     - line, direction
     - legacy
   * - ``tsi``
     - ``StreamingTsi``
     - close
     - longperiod=25, shortperiod=13, signalperiod=13
     - tsi, signal
     - native
   * - ``ulcer_index``
     - ``StreamingUlcerIndex``
     - close
     - timeperiod=14
     - value
     - native
   * - ``vidya``
     - ``StreamingVidya``
     - close
     - timeperiod=14, cmo_period=9
     - value
     - native
   * - ``volosc``
     - ``StreamingVolosc``
     - volume
     - fastperiod=5, slowperiod=10
     - value
     - native
   * - ``vortex``
     - ``StreamingVortex``
     - high, low, close
     - timeperiod=14
     - plus_vi, minus_vi
     - windowed
   * - ``vroc``
     - ``StreamingVroc``
     - volume
     - timeperiod=25
     - value
     - windowed
   * - ``vwap``
     - ``StreamingVWAP``
     - high, low, close, volume
     - -
     - value
     - legacy
   * - ``vwma``
     - ``StreamingVwma``
     - close, volume
     - timeperiod=20
     - value
     - windowed
   * - ``zlema``
     - ``StreamingZlema``
     - close
     - timeperiod=14
     - value
     - native

.. rubric:: futures (1)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``weighted_continuous``
     - ``StreamingWeightedContinuous``
     - front, next, next_weights
     - -
     - value
     - windowed

.. rubric:: math (9)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``add``
     - ``StreamingAdd``
     - a, b
     - -
     - value
     - windowed
   * - ``div``
     - ``StreamingDiv``
     - a, b
     - -
     - value
     - windowed
   * - ``max``
     - ``StreamingMax``
     - real
     - timeperiod=30
     - value
     - windowed
   * - ``min``
     - ``StreamingMin``
     - real
     - timeperiod=30
     - value
     - windowed
   * - ``mult``
     - ``StreamingMult``
     - a, b
     - -
     - value
     - windowed
   * - ``sliding_max``
     - ``StreamingSlidingMax``
     - real
     - timeperiod=30
     - value
     - windowed
   * - ``sliding_min``
     - ``StreamingSlidingMin``
     - real
     - timeperiod=30
     - value
     - windowed
   * - ``sub``
     - ``StreamingSub``
     - a, b
     - -
     - value
     - windowed
   * - ``sum``
     - ``StreamingSum``
     - real
     - timeperiod=30
     - value
     - windowed

.. rubric:: math_ops (5)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``rolling_max``
     - ``StreamingRollingMax``
     - real
     - timeperiod=30
     - value
     - windowed
   * - ``rolling_maxindex``
     - ``StreamingRollingMaxindex``
     - real
     - timeperiod=30
     - value
     - native
   * - ``rolling_min``
     - ``StreamingRollingMin``
     - real
     - timeperiod=30
     - value
     - windowed
   * - ``rolling_minindex``
     - ``StreamingRollingMinindex``
     - real
     - timeperiod=30
     - value
     - native
   * - ``rolling_sum``
     - ``StreamingRollingSum``
     - real
     - timeperiod=30
     - value
     - windowed

.. rubric:: momentum (26)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``adx``
     - ``StreamingAdx``
     - high, low, close
     - timeperiod=14
     - value
     - native
   * - ``adxr``
     - ``StreamingAdxr``
     - high, low, close
     - timeperiod=14
     - value
     - native
   * - ``apo``
     - ``StreamingApo``
     - close
     - fastperiod=12, slowperiod=26, matype=1
     - value
     - native
   * - ``aroon``
     - ``StreamingAroon``
     - high, low
     - timeperiod=14
     - aroondown, aroonup
     - windowed
   * - ``aroonosc``
     - ``StreamingAroonosc``
     - high, low
     - timeperiod=14
     - value
     - windowed
   * - ``bop``
     - ``StreamingBop``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cci``
     - ``StreamingCci``
     - high, low, close
     - timeperiod=14
     - value
     - windowed
   * - ``cmo``
     - ``StreamingCmo``
     - close
     - timeperiod=14
     - value
     - native
   * - ``dx``
     - ``StreamingDx``
     - high, low, close
     - timeperiod=14
     - value
     - native
   * - ``minus_di``
     - ``StreamingMinusDi``
     - high, low, close
     - timeperiod=14
     - value
     - native
   * - ``minus_dm``
     - ``StreamingMinusDm``
     - high, low
     - timeperiod=14
     - value
     - native
   * - ``mom``
     - ``StreamingMom``
     - close
     - timeperiod=10
     - value
     - windowed
   * - ``plus_di``
     - ``StreamingPlusDi``
     - high, low, close
     - timeperiod=14
     - value
     - native
   * - ``plus_dm``
     - ``StreamingPlusDm``
     - high, low
     - timeperiod=14
     - value
     - native
   * - ``ppo``
     - ``StreamingPpo``
     - close
     - fastperiod=12, slowperiod=26, signalperiod=9, matype=1
     - ppo, signal, hist
     - native
   * - ``roc``
     - ``StreamingRoc``
     - close
     - timeperiod=10
     - value
     - windowed
   * - ``rocp``
     - ``StreamingRocp``
     - close
     - timeperiod=10
     - value
     - windowed
   * - ``rocr``
     - ``StreamingRocr``
     - close
     - timeperiod=10
     - value
     - windowed
   * - ``rocr100``
     - ``StreamingRocr100``
     - close
     - timeperiod=10
     - value
     - windowed
   * - ``rsi``
     - ``StreamingRSI``
     - value
     - period=14
     - value
     - legacy
   * - ``stoch``
     - ``StreamingStoch``
     - high, low, close
     - fastk_period=..., slowk_period=..., slowd_period=...
     - slowk, slowd
     - legacy
   * - ``stochf``
     - ``StreamingStochf``
     - high, low, close
     - fastk_period=5, fastd_period=3, fastd_matype=0
     - fastk, fastd
     - native
   * - ``stochrsi``
     - ``StreamingStochrsi``
     - close
     - timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
     - fastk, fastd
     - native
   * - ``trix``
     - ``StreamingTrix``
     - close
     - timeperiod=30
     - value
     - native
   * - ``ultosc``
     - ``StreamingUltosc``
     - high, low, close
     - timeperiod1=7, timeperiod2=14, timeperiod3=28
     - value
     - windowed
   * - ``willr``
     - ``StreamingWillr``
     - high, low, close
     - timeperiod=14
     - value
     - windowed

.. rubric:: options (8)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``close_to_close_vol``
     - ``StreamingCloseToCloseVol``
     - close
     - window=20, trading_days=252.0
     - value
     - windowed
   * - ``garman_klass_vol``
     - ``StreamingGarmanKlassVol``
     - open, high, low, close
     - window=20, trading_days=252.0
     - value
     - windowed
   * - ``iv_percentile``
     - ``StreamingIvPercentile``
     - iv_series
     - window=252
     - value
     - windowed
   * - ``iv_rank``
     - ``StreamingIvRank``
     - iv_series
     - window=252
     - value
     - windowed
   * - ``iv_zscore``
     - ``StreamingIvZscore``
     - iv_series
     - window=252
     - value
     - windowed
   * - ``parkinson_vol``
     - ``StreamingParkinsonVol``
     - high, low
     - window=20, trading_days=252.0
     - value
     - windowed
   * - ``rogers_satchell_vol``
     - ``StreamingRogersSatchellVol``
     - open, high, low, close
     - window=20, trading_days=252.0
     - value
     - windowed
   * - ``yang_zhang_vol``
     - ``StreamingYangZhangVol``
     - open, high, low, close
     - window=20, trading_days=252.0
     - value
     - windowed

.. rubric:: overlap (19)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``bbands``
     - ``StreamingBBands``
     - value
     - period=20, nbdevup=..., nbdevdn=...
     - upper, middle, lower
     - legacy
   * - ``dema``
     - ``StreamingDema``
     - close
     - timeperiod=30
     - value
     - native
   * - ``ema``
     - ``StreamingEMA``
     - value
     - period
     - value
     - legacy
   * - ``kama``
     - ``StreamingKama``
     - close
     - timeperiod=30
     - value
     - native
   * - ``ma``
     - ``StreamingMa``
     - close
     - timeperiod=30, matype=0
     - value
     - native
   * - ``macd``
     - ``StreamingMACD``
     - value
     - fastperiod=..., slowperiod=..., signalperiod=...
     - macd, signal, hist
     - legacy
   * - ``macdext``
     - ``StreamingMacdext``
     - close
     - fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9, signalmatype=1
     - macd, signal, hist
     - native
   * - ``macdfix``
     - ``StreamingMacdfix``
     - close
     - signalperiod=9
     - macd, signal, hist
     - native
   * - ``mama``
     - ``StreamingMama``
     - close
     - fastlimit=0.5, slowlimit=0.05
     - mama, fama
     - native
   * - ``mavp``
     - ``StreamingMavp``
     - close, periods
     - minperiod=2, maxperiod=30, matype=0
     - value
     - native
   * - ``midpoint``
     - ``StreamingMidpoint``
     - close
     - timeperiod=14
     - value
     - windowed
   * - ``midprice``
     - ``StreamingMidprice``
     - high, low
     - timeperiod=14
     - value
     - windowed
   * - ``sar``
     - ``StreamingSar``
     - high, low
     - acceleration=0.02, maximum=0.2
     - value
     - native
   * - ``sarext``
     - ``StreamingSarext``
     - high, low
     - startvalue=0.0, offsetonreverse=0.0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2
     - value
     - native
   * - ``sma``
     - ``StreamingSMA``
     - value
     - period
     - value
     - legacy
   * - ``t3``
     - ``StreamingT3``
     - close
     - timeperiod=5, vfactor=0.7
     - value
     - native
   * - ``tema``
     - ``StreamingTema``
     - close
     - timeperiod=30
     - value
     - native
   * - ``trima``
     - ``StreamingTrima``
     - close
     - timeperiod=30
     - value
     - windowed
   * - ``wma``
     - ``StreamingWma``
     - close
     - timeperiod=30
     - value
     - windowed

.. rubric:: pattern (61)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``cdl2crows``
     - ``StreamingCdl2crows``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdl3blackcrows``
     - ``StreamingCdl3blackcrows``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdl3inside``
     - ``StreamingCdl3inside``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdl3linestrike``
     - ``StreamingCdl3linestrike``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdl3outside``
     - ``StreamingCdl3outside``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdl3starsinsouth``
     - ``StreamingCdl3starsinsouth``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdl3whitesoldiers``
     - ``StreamingCdl3whitesoldiers``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlabandonedbaby``
     - ``StreamingCdlabandonedbaby``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdladvanceblock``
     - ``StreamingCdladvanceblock``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlbelthold``
     - ``StreamingCdlbelthold``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlbreakaway``
     - ``StreamingCdlbreakaway``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlclosingmarubozu``
     - ``StreamingCdlclosingmarubozu``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlconcealbabyswall``
     - ``StreamingCdlconcealbabyswall``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlcounterattack``
     - ``StreamingCdlcounterattack``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdldarkcloudcover``
     - ``StreamingCdldarkcloudcover``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdldoji``
     - ``StreamingCdldoji``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdldojistar``
     - ``StreamingCdldojistar``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdldragonflydoji``
     - ``StreamingCdldragonflydoji``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlengulfing``
     - ``StreamingCdlengulfing``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdleveningdojistar``
     - ``StreamingCdleveningdojistar``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdleveningstar``
     - ``StreamingCdleveningstar``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlgapsidesidewhite``
     - ``StreamingCdlgapsidesidewhite``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlgravestonedoji``
     - ``StreamingCdlgravestonedoji``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlhammer``
     - ``StreamingCdlhammer``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlhangingman``
     - ``StreamingCdlhangingman``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlharami``
     - ``StreamingCdlharami``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlharamicross``
     - ``StreamingCdlharamicross``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlhighwave``
     - ``StreamingCdlhighwave``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlhikkake``
     - ``StreamingCdlhikkake``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlhikkakemod``
     - ``StreamingCdlhikkakemod``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlhomingpigeon``
     - ``StreamingCdlhomingpigeon``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlidentical3crows``
     - ``StreamingCdlidentical3crows``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlinneck``
     - ``StreamingCdlinneck``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlinvertedhammer``
     - ``StreamingCdlinvertedhammer``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlkicking``
     - ``StreamingCdlkicking``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlkickingbylength``
     - ``StreamingCdlkickingbylength``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlladderbottom``
     - ``StreamingCdlladderbottom``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdllongleggeddoji``
     - ``StreamingCdllongleggeddoji``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdllongline``
     - ``StreamingCdllongline``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlmarubozu``
     - ``StreamingCdlmarubozu``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlmatchinglow``
     - ``StreamingCdlmatchinglow``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlmathold``
     - ``StreamingCdlmathold``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlmorningdojistar``
     - ``StreamingCdlmorningdojistar``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlmorningstar``
     - ``StreamingCdlmorningstar``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlonneck``
     - ``StreamingCdlonneck``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlpiercing``
     - ``StreamingCdlpiercing``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlrickshawman``
     - ``StreamingCdlrickshawman``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlrisefall3methods``
     - ``StreamingCdlrisefall3methods``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlseparatinglines``
     - ``StreamingCdlseparatinglines``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlshootingstar``
     - ``StreamingCdlshootingstar``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlshortline``
     - ``StreamingCdlshortline``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlspinningtop``
     - ``StreamingCdlspinningtop``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlstalledpattern``
     - ``StreamingCdlstalledpattern``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlsticksandwich``
     - ``StreamingCdlsticksandwich``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdltakuri``
     - ``StreamingCdltakuri``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdltasukigap``
     - ``StreamingCdltasukigap``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlthrusting``
     - ``StreamingCdlthrusting``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdltristar``
     - ``StreamingCdltristar``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlunique3river``
     - ``StreamingCdlunique3river``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlupsidegap2crows``
     - ``StreamingCdlupsidegap2crows``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``cdlxsidegap3methods``
     - ``StreamingCdlxsidegap3methods``
     - open, high, low, close
     - -
     - value
     - windowed

.. rubric:: portfolio (5)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``ratio``
     - ``StreamingRatio``
     - a, b
     - -
     - value
     - windowed
   * - ``relative_strength``
     - ``StreamingRelativeStrength``
     - asset_returns, benchmark_returns
     - -
     - value
     - native
   * - ``rolling_beta``
     - ``StreamingRollingBeta``
     - asset, benchmark
     - window=20
     - value
     - windowed
   * - ``spread``
     - ``StreamingSpread``
     - a, b
     - hedge=1.0
     - value
     - windowed
   * - ``zscore_series``
     - ``StreamingZscoreSeries``
     - x
     - window=20
     - value
     - windowed

.. rubric:: price_transform (4)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``avgprice``
     - ``StreamingAvgprice``
     - open, high, low, close
     - -
     - value
     - windowed
   * - ``medprice``
     - ``StreamingMedprice``
     - high, low
     - -
     - value
     - windowed
   * - ``typprice``
     - ``StreamingTypprice``
     - high, low, close
     - -
     - value
     - windowed
   * - ``wclprice``
     - ``StreamingWclprice``
     - high, low, close
     - -
     - value
     - windowed

.. rubric:: regime (4)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``detect_breaks_cusum``
     - ``StreamingDetectBreaksCusum``
     - series
     - window=20, threshold=3.0, slack=0.5
     - value
     - native
   * - ``regime_adx``
     - ``StreamingRegimeAdx``
     - adx
     - threshold=25.0
     - value
     - windowed
   * - ``regime_combined``
     - ``StreamingRegimeCombined``
     - adx, atr, close
     - adx_threshold=25.0, atr_pct_threshold=0.005
     - value
     - windowed
   * - ``rolling_variance_break``
     - ``StreamingRollingVarianceBreak``
     - series
     - short_window=10, long_window=50, threshold=2.0
     - value
     - windowed

.. rubric:: statistic (9)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``beta``
     - ``StreamingBeta``
     - real0, real1
     - timeperiod=5
     - value
     - windowed
   * - ``correl``
     - ``StreamingCorrel``
     - real0, real1
     - timeperiod=30
     - value
     - windowed
   * - ``linearreg``
     - ``StreamingLinearreg``
     - close
     - timeperiod=14
     - value
     - windowed
   * - ``linearreg_angle``
     - ``StreamingLinearregAngle``
     - close
     - timeperiod=14
     - value
     - windowed
   * - ``linearreg_intercept``
     - ``StreamingLinearregIntercept``
     - close
     - timeperiod=14
     - value
     - windowed
   * - ``linearreg_slope``
     - ``StreamingLinearregSlope``
     - close
     - timeperiod=14
     - value
     - windowed
   * - ``stddev``
     - ``StreamingStddev``
     - real
     - timeperiod=5, nbdev=1.0
     - value
     - windowed
   * - ``tsf``
     - ``StreamingTsf``
     - close
     - timeperiod=14
     - value
     - windowed
   * - ``var``
     - ``StreamingVar``
     - real
     - timeperiod=5, nbdev=1.0
     - value
     - windowed

.. rubric:: streaming-only (5)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``backtest``
     - ``StreamingBacktest``
     - close, signal
     - commission_per_trade=..., slippage_bps=...
     - position, bar_return, equity, n_trades, total_commission, win_rate, avg_win, avg_loss, kelly_fraction
     - legacy
   * - ``cooldown``
     - ``StreamingCooldown``
     - signal
     - bars
     - value
     - legacy
   * - ``cross``
     - ``StreamingCross``
     - a, b
     - dead_band=...
     - value
     - legacy
   * - ``option_greeks``
     - ``StreamingOptionGreeks``
     - option_price, underlying, time_to_expiry
     - strike, option_type=..., model=..., rate=..., carry=...
     - iv, delta, gamma, vega, theta, rho
     - legacy
   * - ``streak``
     - ``StreamingStreak``
     - condition
     - n
     - value
     - legacy

.. rubric:: utilities (3)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``check_cross``
     - ``StreamingCheckCross``
     - fast, slow
     - -
     - value
     - windowed
   * - ``funding_cumulative_pnl``
     - ``StreamingFundingCumulativePnl``
     - position_size, funding_rate
     - -
     - value
     - native
   * - ``rsi_threshold_signals``
     - ``StreamingRsiThresholdSignals``
     - close
     - timeperiod=14, oversold=30.0, overbought=70.0
     - value
     - native

.. rubric:: volatility (3)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``atr``
     - ``StreamingATR``
     - high, low, close
     - period=14
     - value
     - legacy
   * - ``natr``
     - ``StreamingNatr``
     - high, low, close
     - timeperiod=14
     - value
     - native
   * - ``trange``
     - ``StreamingTrange``
     - high, low, close
     - -
     - value
     - windowed

.. rubric:: volume (4)

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 26 14 10

   * - Name
     - Python class
     - Inputs
     - Params (defaults)
     - Outputs
     - Kind
   * - ``ad``
     - ``StreamingAd``
     - high, low, close, volume
     - -
     - value
     - native
   * - ``adosc``
     - ``StreamingAdosc``
     - high, low, close, volume
     - fastperiod=3, slowperiod=10
     - value
     - native
   * - ``mfi``
     - ``StreamingMfi``
     - high, low, close, volume
     - timeperiod=14
     - value
     - native
   * - ``obv``
     - ``StreamingObv``
     - close, volume
     - -
     - value
     - native

.. END GENERATED STREAM CATALOG

API reference
-------------

.. automodule:: ferro_ta.streaming
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
