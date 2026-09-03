Benchmarks
==========

The benchmark suite is meant to support a narrow claim: ferro-ta is often
faster on selected indicators, and the evidence is published in a reproducible
form.

What is published
-----------------

The authoritative benchmark workflow lives in ``benchmarks/``:

- Cross-library speed suite: ``benchmarks/test_speed.py``
- Cross-library accuracy suite: ``benchmarks/test_accuracy.py``
- TA-Lib head-to-head script: ``benchmarks/bench_vs_talib.py``
- Backtesting engine benchmark: ``benchmarks/bench_backtest.py``
- Table generation from benchmark JSON: ``benchmarks/benchmark_table.py``
- Perf-contract artifact bundle: ``benchmarks/run_perf_contract.py``

Backtesting engine
------------------

The backtester is built for the vectorized case: you already hold a signal
array (or an OHLCV frame plus signals) and you want the equity curve, the trade
list and the summary statistics.

What that buys you, structurally:

- **The signal → equity loop runs entirely in Rust.** ``backtest_core`` and
  ``backtest_ohlcv_core`` are single O(n) passes over the bars with the GIL
  released; there is no Python on the hot path. Commission and slippage are
  applied inside the same pass rather than as a follow-up Python step
  (``crates/ferro_ta_core/src/backtest.rs``, ``src/backtest/mod.rs``).
- **All 23 performance metrics come from one call.**
  ``compute_performance_metrics`` computes the whole set in a single traversal,
  so asking for twenty-three statistics costs about what asking for one costs.
  The corollary matters: against a NumPy snippet that computes only Sharpe and
  max drawdown, the full call is *slower*, and the checked-in artifact records
  that ratio explicitly. You are buying the other twenty-one metrics.
- **Monte Carlo and multi-asset runs are Rayon-parallel** with deterministic
  LCG seeding and the GIL released, so a bootstrap over many simulations scales
  across cores instead of serialising on the interpreter.
- **Walk-forward index generation is O(number of folds)**, not O(bars) — it
  produces slice boundaries, not copies.

When it is the right tool: signal-array backtests, parameter sweeps, bootstrap
confidence intervals, and multi-asset portfolio runs whose per-bar logic is
expressible as arrays.

When it is not: path-dependent strategies that must make a decision inside the
bar loop in Python, or anything that needs a broker event model. For those, use
a dedicated event-driven engine — ``ferro_ta.analysis.backtest`` says as much in
its own module docstring.

For numbers, read the artifact rather than this page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``benchmarks/artifacts/latest/bench_backtest_results.json`` carries per-size
timings for every path above, together with the machine, commit, Python version
and Rust toolchain that produced them. Regenerate it with:

.. code-block:: bash

   python benchmarks/bench_backtest.py --sizes 10000 100000 \
       --json /tmp/bench_backtest_results.json

Add ``--skip-competitors`` to run only the ferro-ta paths, and ``--assets`` /
``--sims`` to change the multi-asset and Monte Carlo shapes. Absolute figures
move a great deal between builds and machines — a fresh local run of the same
script has disagreed with the committed artifact by more than 2× on several
rows, including flipping the sign of the multi-asset parallel-vs-loop
comparison — which is exactly why they are not reproduced here.

.. note::

   Earlier revisions of this page carried a table ranking the backtester
   against named third-party backtesting libraries. Most of those numbers were
   not reproducible from anything in this repository:
   ``benchmarks/bench_backtest.py`` has never contained an event-driven-library
   comparison, and its only optional competitor path covers a single vectorized
   library. Comparative numbers now live in the benchmark artifacts under
   ``benchmarks/``, where they carry their own provenance and are regenerated
   rather than transcribed.

Latest checked-in TA-Lib artifact
---------------------------------

The current checked-in TA-Lib comparison artifact benchmarks contiguous
``float64`` arrays at 10k and 100k bars on an ``Apple M3 Max`` with 14 logical
cores, about 38.7 GB RAM, ``CPython 3.13.5``, and ``Rust 1.91.1`` using the
default release profile (``lto = true``, ``codegen-units = 1``).

Summary transcribed from the ``summary`` block of
``benchmarks/artifacts/latest/benchmark_vs_talib.json`` (generated
2026-03-24). "Wins", "ties" and "losses" use that artifact's own tie band of
0.95–1.05; a tie is therefore *not* a ferro-ta win.

.. list-table::
   :header-rows: 1

   * - Size
     - Rows
     - Wins
     - Ties
     - Losses
     - Median speedup
     - TA-Lib wins or ties
   * - ``10,000``
     - 12
     - 5
     - 4
     - 3
     - ``1.0383x``
     - ``EMA``, ``RSI``, ``ATR``, ``STOCH``, ``ADX``, ``OBV``, ``MFI``
   * - ``100,000``
     - 12
     - 7
     - 3
     - 2
     - ``1.1748x``
     - ``EMA``, ``ATR``, ``STOCH``, ``ADX``, ``OBV``

Examples from the 100k-bar run:

.. list-table::
   :header-rows: 1

   * - Indicator
     - ferro-ta
     - TA-Lib
     - Speedup
     - Read
   * - ``MFI``
     - ``0.1736 ms``
     - ``0.5635 ms``
     - ``3.2460x``
     - clear ferro-ta win
   * - ``BBANDS``
     - ``0.2213 ms``
     - ``0.4350 ms``
     - ``1.9657x``
     - clear ferro-ta win
   * - ``SMA``
     - ``0.0758 ms``
     - ``0.1466 ms``
     - ``1.9340x``
     - clear ferro-ta win
   * - ``MACD``
     - ``0.4458 ms``
     - ``0.6455 ms``
     - ``1.4480x``
     - ferro-ta win
   * - ``ATR``
     - ``0.5035 ms``
     - ``0.5014 ms``
     - ``0.9958x``
     - tie on this machine
   * - ``ADX``
     - ``0.8133 ms``
     - ``0.5973 ms``
     - ``0.7344x``
     - TA-Lib win
   * - ``STOCH``
     - ``1.7758 ms``
     - ``0.8185 ms``
     - ``0.4609x``
     - TA-Lib win

Methodology notes
-----------------

- The head-to-head script uses the same synthetic OHLCV generator, the same
  parameters, and the same contiguous ``float64`` array layout for both
  libraries.
- Reported speedup is ``TA-Lib median time / ferro-ta median time``.
- The script uses 1 warmup run and 7 measured runs per case, and now records
  the full per-run timing samples, not just one selected number.
- Published JSON artifacts include machine/runtime metadata, git metadata, Rust
  toolchain and build-profile metadata, per-run variance statistics, and
  Python-tracked peak allocation snapshots.
- Allocation snapshots are based on ``tracemalloc`` and capture Python-tracked
  allocations only; they are not full native RSS profiles.
- If your workload uses non-contiguous arrays, different dtypes, or different
  batch sizes, benchmark that exact workload. Those factors can materially
  change the result.

Reproduce the TA-Lib comparison
-------------------------------

.. code-block:: bash

   pip install ta-lib
   python benchmarks/bench_vs_talib.py --sizes 10000 100000 --json benchmark_vs_talib.json

The JSON output is the main artifact to review when publishing performance
claims.

Cross-library suite
-------------------

Run the broader speed suite on 100,000 bars:

.. code-block:: bash

   uv run pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json -v

That run writes ``benchmarks/results.json``; render it as a markdown table
with:

.. code-block:: bash

   uv run python benchmarks/benchmark_table.py

The rendered table is checked in at ``benchmarks/README.md``. Per-indicator
throughput figures are deliberately not repeated on this page: the same
indicator measures very differently across the harnesses in this repository
(the pytest suite reports median µs per call, ``bench_vs_talib.py`` and
``bench_vs_openalgo.py`` each report their own ``M bars/s``), so a transcribed
number is ambiguous about which harness produced it. Read the artifact that
matches the harness you care about.

Perf-contract artifacts
-----------------------

Use the perf-contract runner when you want a compact, machine-readable artifact
bundle for single-series latency, batch throughput, streaming throughput, and
hotspot attribution:

.. code-block:: bash

   uv run python benchmarks/run_perf_contract.py \
       --output-dir benchmarks/artifacts/latest --skip-simd --skip-talib

.. warning::

   Pass ``--skip-simd`` unless you specifically want the portable-vs-SIMD
   comparison. Without it the runner calls ``benchmarks/bench_simd.py``, which
   runs ``maturin develop --release`` **twice** — once with
   ``--no-default-features`` and once with the default features — replacing
   whatever build you currently have installed. ``--skip-talib`` skips the
   TA-Lib suite, which needs ``ta-lib`` present.

The committed ``perf-contract/*.json`` files are a **local snapshot, not the
enforced gate.** CI regenerates that directory from this same runner on its own
hardware and then checks the fresh output, so treat the committed copies as an
example of the format and of one machine's result rather than as the numbers CI
is asserting. ``benchmarks/README.md`` documents this in more detail.

See ``benchmarks/README.md`` for the detailed benchmark playbook and the
checked-in comparison tables.
