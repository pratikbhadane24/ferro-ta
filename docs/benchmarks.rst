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
- Table generation from benchmark JSON: ``benchmarks/benchmark_table.py``
- Perf-contract artifact bundle: ``benchmarks/run_perf_contract.py``

Latest checked-in TA-Lib artifact
---------------------------------

The current checked-in TA-Lib comparison artifact benchmarks contiguous
``float64`` arrays at 10k and 100k bars on an ``Apple M3 Max`` with 14 logical
cores, about 38.7 GB RAM, ``CPython 3.13.5``, and ``Rust 1.91.1`` using the
default release profile (``lto = true``, ``codegen-units = 1``).

Summary from ``benchmarks/artifacts/latest/benchmark_vs_talib.json``:

.. list-table::
   :header-rows: 1

   * - Size
     - Rows
     - ferro-ta wins
     - Median speedup
     - TA-Lib wins or ties
   * - ``10,000``
     - 12
     - 6
     - ``1.0850x``
     - ``EMA``, ``RSI``, ``ATR``, ``STOCH``, ``ADX``, ``OBV``
   * - ``100,000``
     - 12
     - 6
     - ``1.0784x``
     - ``EMA``, ``RSI``, ``ATR``, ``STOCH``, ``ADX``, ``OBV``

Examples from the 100k-bar run:

.. list-table::
   :header-rows: 1

   * - Indicator
     - ferro-ta
     - TA-Lib
     - Speedup
     - Read
   * - ``SMA``
     - ``0.0985 ms``
     - ``0.2241 ms``
     - ``2.2751x``
     - clear ferro-ta win
   * - ``BBANDS``
     - ``0.2122 ms``
     - ``0.4966 ms``
     - ``2.3402x``
     - clear ferro-ta win
   * - ``MACD``
     - ``0.5152 ms``
     - ``0.7111 ms``
     - ``1.3801x``
     - ferro-ta win
   * - ``STOCH``
     - ``1.7064 ms``
     - ``0.7603 ms``
     - ``0.4455x``
     - TA-Lib win
   * - ``ADX``
     - ``0.7910 ms``
     - ``0.5769 ms``
     - ``0.7294x``
     - TA-Lib win
   * - ``ATR``
     - ``0.5087 ms``
     - ``0.5147 ms``
     - ``1.0118x``
     - tie on this machine

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

Selected throughput examples from the checked-in table:

.. list-table::
   :header-rows: 1

   * - Indicator
     - Throughput
   * - ``ADD``
     - 1.9 G bars/s
   * - ``CDLENGULFING``
     - 454 M bars/s
   * - ``EMA``
     - 444 M bars/s
   * - ``SMA``
     - 259 M bars/s
   * - ``RSI``
     - 145 M bars/s
   * - ``ATR``
     - 70 M bars/s
   * - ``MACD``
     - 104 M bars/s
   * - ``STOCH``
     - 33 M bars/s

Perf-contract artifacts
-----------------------

Use the perf-contract runner when you want a compact, machine-readable artifact
bundle for single-series latency, batch throughput, streaming throughput, and
hotspot attribution:

.. code-block:: bash

   uv run python benchmarks/run_perf_contract.py --output-dir benchmarks/artifacts/latest

See ``benchmarks/README.md`` for the detailed benchmark playbook and the
checked-in comparison tables.
